from __future__ import annotations

import asyncio
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Iterable,
    Protocol,
    Sequence,
    runtime_checkable,
)

from minimal_harness.agent.protocol import Agent
from minimal_harness.agent.registry import AgentRegistryProtocol, Session
from minimal_harness.tool.base import StreamingTool

if TYPE_CHECKING:
    from minimal_harness.client.built_in.memory import PersistentMemory
    from minimal_harness.llm import LLMProvider
    from minimal_harness.memory import ExtendedInputContentPart
    from minimal_harness.types import AgentEvent


@runtime_checkable
class AgentRuntimeProtocol(Protocol):
    def get_running_session_ids(self) -> list[str]: ...
    def is_session_running(self, session_id: str) -> bool: ...
    def run(
        self,
        user_input: Iterable[ExtendedInputContentPart],
        stop_event: asyncio.Event | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[AgentEvent]: ...
    def create_session(
        self,
        config: dict[str, Any],
        tools: Sequence[StreamingTool],
        memory: PersistentMemory,
        agent_factory: Callable[..., Agent],
    ) -> Session: ...
    def load_session(
        self,
        session_id: str,
        config: dict[str, Any],
        tools: Sequence[StreamingTool],
        agent_factory: Callable[..., Agent],
        memory_dir: str | None = None,
    ) -> Session: ...
    def get_session(self, session_id: str) -> Session | None: ...
    def list_sessions(self) -> list[Session]: ...
    def create_handoff_tool(self) -> StreamingTool: ...

    def create_discover_agents_tool(self) -> StreamingTool: ...

    def inject_runtime_tools(
        self,
        session: Session,
        *,
        tool_names: tuple[str, ...] = ("handoff", "discover_agents"),
    ) -> None: ...

    def register_agent(
        self,
        name: str,
        description: str,
        system_prompt: str,
        llm_provider: "LLMProvider",
        tools: Sequence[StreamingTool],
        agent_factory: Callable[..., Agent] | None = None,
    ) -> str: ...


class AgentRuntime:
    def __init__(self, registry: AgentRegistryProtocol) -> None:
        self.registry = registry
        self._sessions: dict[str, Session] = {}
        self._running_tasks: dict[str, asyncio.Task[None]] = {}

    def create_handoff_tool(self) -> StreamingTool:
        runtime = self

        async def handoff_fn(
            target_session_id: str, context_summary: str, task_description: str
        ) -> AsyncIterator[Any]:
            combined = f"Context: {context_summary}\n\nTask: {task_description}"
            session = runtime._sessions.get(target_session_id)
            if session is None:
                yield {
                    "status": "error",
                    "message": f"Session {target_session_id} not found",
                }
                return

            async def _background_run(
                user_input: "Iterable[ExtendedInputContentPart]",
            ) -> None:
                async for _ in runtime.run(
                    user_input=user_input,
                    session_id=target_session_id,
                ):
                    pass

            task = asyncio.create_task(
                _background_run([{"type": "text", "text": combined}])
            )
            runtime._running_tasks[target_session_id] = task
            task.add_done_callback(
                lambda t: runtime._running_tasks.pop(target_session_id, None)
            )

            yield {
                "status": "handoff",
                "message": "Task handed off to another agent. Results will be delivered when ready.",
            }

        return StreamingTool(
            name="handoff",
            description="Hand off a task to another agent session. Use discover_agents first to find available sessions.",
            parameters={
                "type": "object",
                "properties": {
                    "target_session_id": {
                        "type": "string",
                        "description": "The session ID of the target agent to hand off to.",
                    },
                    "context_summary": {
                        "type": "string",
                        "description": "Summary of the current context and conversation state.",
                    },
                    "task_description": {
                        "type": "string",
                        "description": "Description of the task to hand off to the next agent.",
                    },
                },
                "required": [
                    "target_session_id",
                    "context_summary",
                    "task_description",
                ],
            },
            fn=handoff_fn,
        )

    def create_discover_agents_tool(self) -> StreamingTool:
        runtime = self

        async def discover_fn() -> AsyncIterator[Any]:
            registered = [
                {"name": m.name, "description": m.description, "type": "registered"}
                for m in runtime.registry.get_all()
            ]
            sessions = [
                {
                    "session_id": s.session_id,
                    "name": s.name,
                    "type": "session",
                    "running": runtime.is_session_running(s.session_id),
                }
                for s in runtime.list_sessions()
            ]
            yield {
                "status": "ok",
                "agents": registered + sessions,
            }

        return StreamingTool(
            name="discover_agents",
            description="Discover available agents and active sessions in the system.",
            parameters={
                "type": "object",
                "properties": {},
            },
            fn=discover_fn,
        )

    def get_running_session_ids(self) -> list[str]:
        return list(self._running_tasks.keys())

    def is_session_running(self, session_id: str) -> bool:
        return session_id in self._running_tasks

    async def run(
        self,
        user_input: "Iterable[ExtendedInputContentPart]",
        stop_event: asyncio.Event | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator["AgentEvent"]:
        if session_id is None:
            return
        session = self._sessions.get(session_id)
        if session is None:
            return
        self.inject_runtime_tools(session)
        async for event in session.agent.run(
            user_input=user_input,
            stop_event=stop_event,
            memory=session.memory,
            tools=session.tools,
        ):
            if session.event_queue is not None and not session.event_queue.full():
                session.event_queue.put_nowait(event)
            yield event

    def create_session(
        self,
        config: dict[str, Any],
        tools: Sequence[StreamingTool],
        memory: PersistentMemory,
        agent_factory: Callable[..., Agent],
    ) -> Session:
        llm = self._create_llm_provider(config)
        agent = agent_factory(llm_provider=llm, tools=list(tools), memory=memory)
        session = Session(
            session_id=memory._session_id,
            name=memory.title or "New Chat",
            agent=agent,
            memory=memory,
            tools=list(tools),
        )
        self._sessions[session.session_id] = session
        return session

    def load_session(
        self,
        session_id: str,
        config: dict[str, Any],
        tools: Sequence[StreamingTool],
        agent_factory: Callable[..., Agent],
        memory_dir: str | None = None,
    ) -> Session:
        from pathlib import Path

        from minimal_harness.client.built_in.memory import PersistentMemory

        directory = Path(memory_dir) if memory_dir else None
        memory = PersistentMemory.from_session(session_id, memory_dir=directory)
        llm = self._create_llm_provider(config)
        agent = agent_factory(llm_provider=llm, tools=list(tools), memory=memory)
        session = Session(
            session_id=memory._session_id,
            name=memory.title or "Untitled",
            agent=agent,
            memory=memory,
            tools=list(tools),
        )
        self._sessions[session.session_id] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[Session]:
        return list(self._sessions.values())

    def inject_runtime_tools(
        self,
        session: Session,
        *,
        tool_names: tuple[str, ...] = ("handoff", "discover_agents"),
    ) -> None:
        existing = {t.name for t in session.tools}
        tools_to_add: list[StreamingTool] = []
        if "handoff" in tool_names and "handoff" not in existing:
            tools_to_add.append(self.create_handoff_tool())
        if "discover_agents" in tool_names and "discover_agents" not in existing:
            tools_to_add.append(self.create_discover_agents_tool())
        if tools_to_add:
            session.tools.extend(tools_to_add)

    def register_agent(
        self,
        name: str,
        description: str,
        system_prompt: str,
        llm_provider: "LLMProvider",
        tools: Sequence[StreamingTool],
        agent_factory: Callable[..., Agent] | None = None,
    ) -> str:
        from minimal_harness.agent.simple import SimpleAgent
        from minimal_harness.client.built_in.memory import PersistentMemory

        factory = agent_factory or SimpleAgent
        memory = PersistentMemory(system_prompt=system_prompt)
        agent = factory(llm_provider=llm_provider, tools=list(tools), memory=memory)
        session = Session(
            session_id=memory._session_id,
            name=name,
            agent=agent,
            memory=memory,
            tools=list(tools),
        )
        self._sessions[session.session_id] = session
        self.registry.register(agent=agent, name=name, description=description)
        return session.session_id

    def _create_llm_provider(self, config: dict[str, Any]) -> "LLMProvider":
        from anthropic import AsyncAnthropic
        from openai import AsyncOpenAI

        from minimal_harness.llm import AnthropicLLMProvider, OpenAILLMProvider

        provider = config.get("provider", "openai")
        kwargs: dict[str, Any] = {}
        if config.get("base_url"):
            kwargs["base_url"] = config["base_url"]
        if config.get("api_key"):
            kwargs["api_key"] = config["api_key"]

        if provider == "anthropic":
            return AnthropicLLMProvider(
                client=AsyncAnthropic(**kwargs),
                model=config.get("model", ""),
            )
        return OpenAILLMProvider(
            client=AsyncOpenAI(**kwargs), model=config.get("model", "")
        )
