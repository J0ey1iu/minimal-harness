from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
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
from minimal_harness.agent.registry import AgentRegistryProtocol
from minimal_harness.tool.base import StreamingTool, Tool
from minimal_harness.tool.registry import ToolRegistry

if TYPE_CHECKING:
    from minimal_harness.agent.session import Session
    from minimal_harness.llm import LLMProvider
    from minimal_harness.memory import ExtendedInputContentPart, Memory
    from minimal_harness.types import AgentEvent


DEFAULT_QUEUE_SIZE = 1000


@dataclass
class _HandoffMeta:
    name: str
    default_tools: list[str] | None = None
    event_queue: asyncio.Queue["AgentEvent"] = field(
        default_factory=lambda: asyncio.Queue(maxsize=DEFAULT_QUEUE_SIZE)
    )


@runtime_checkable
class AgentRuntimeProtocol(Protocol):
    """Runtime for orchestrating multi-agent conversations."""

    def run(
        self,
        agent: Agent,
        user_input: Iterable[ExtendedInputContentPart],
        memory: Memory,
        tools: Sequence[Tool],
        stop_event: asyncio.Event | None = None,
    ) -> AsyncIterator[AgentEvent]: ...

    def register_agent(
        self,
        name: str,
        description: str,
        system_prompt: str,
        llm_provider: "LLMProvider",
        tools: Sequence[Tool],
        agent_factory: Callable[..., Agent] | None = None,
        memory_factory: Callable[[str, str, str], Memory] | None = None,
        default_tools: Sequence[str] | None = None,
    ) -> str: ...

    def create_handoff_tool(self) -> Tool: ...
    def create_discover_agents_tool(self) -> Tool: ...

    def run_background(
        self,
        session_id: str,
        agent: Agent,
        user_input: Iterable[ExtendedInputContentPart],
        memory: Memory,
        tools: Sequence[Tool],
        stop_event: asyncio.Event | None = None,
    ) -> None: ...

    def get_session(self, session_id: str) -> Session | None: ...
    def is_background_task_running(self, session_id: str) -> bool: ...
    def drain_handoff_events(self, session_id: str) -> list[AgentEvent]: ...

    @property
    def registered_agents(self) -> list[Session]: ...


class AgentRuntime:
    def __init__(
        self,
        registry: AgentRegistryProtocol,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.registry = registry
        self._tool_registry = tool_registry or ToolRegistry()
        self._sessions: dict[str, Session] = {}
        self._handoff_meta: dict[str, _HandoffMeta] = {}
        self._background_tasks: dict[str, asyncio.Task[None]] = {}
        self._on_handoff_event: Callable[[str], None] | None = None

    def _resolve_tools(self, tool_names: Sequence[str]) -> list[Tool]:
        return [
            t for name in tool_names if (t := self._tool_registry.get(name)) is not None
        ]

    def set_on_handoff_event(self, callback: Callable[[str], None] | None) -> None:
        self._on_handoff_event = callback

    async def run(
        self,
        agent: Agent,
        user_input: "Iterable[ExtendedInputContentPart]",
        memory: "Memory",
        tools: "Sequence[Tool]",
        stop_event: asyncio.Event | None = None,
    ) -> AsyncIterator["AgentEvent"]:
        async for event in agent.run(
            user_input=user_input,
            stop_event=stop_event,
            memory=memory,
            tools=tools,
        ):
            yield event

    async def run_session(
        self,
        session: Session,
        user_input: "Iterable[ExtendedInputContentPart]",
        stop_event: asyncio.Event | None = None,
    ) -> AsyncIterator["AgentEvent"]:
        async for event in self.run(
            agent=session.agent,
            user_input=user_input,
            memory=session.memory,
            tools=session.tools,
            stop_event=stop_event or session.stop_event,
        ):
            yield event

    def run_background(
        self,
        session_id: str,
        agent: Agent,
        user_input: "Iterable[ExtendedInputContentPart]",
        memory: "Memory",
        tools: "Sequence[Tool]",
        stop_event: asyncio.Event | None = None,
    ) -> None:
        runtime = self
        self.inject_runtime_tools(list(tools))

        async def _task() -> None:
            async for event in runtime.run(
                agent=agent,
                user_input=user_input,
                memory=memory,
                tools=tools,
                stop_event=stop_event,
            ):
                meta = runtime._handoff_meta.get(session_id)
                if meta is not None and not meta.event_queue.full():
                    meta.event_queue.put_nowait(event)
                if runtime._on_handoff_event is not None:
                    runtime._on_handoff_event(session_id)
            runtime._background_tasks.pop(session_id, None)

        task = asyncio.create_task(_task())
        self._background_tasks[session_id] = task
        task.add_done_callback(lambda t: self._background_tasks.pop(session_id, None))
        if self._on_handoff_event is not None:
            self._on_handoff_event(session_id)

    def _get_handoff_target_by_name(
        self, name: str
    ) -> tuple[Session, _HandoffMeta] | None:
        for sid, session in self._sessions.items():
            meta = self._handoff_meta.get(sid)
            if meta is not None and meta.name == name:
                return session, meta
        return None

    def create_handoff_tool(self) -> StreamingTool:
        runtime = self

        async def handoff_fn(
            target_agent_name: str, context_summary: str, task_description: str
        ) -> AsyncIterator[Any]:
            result = runtime._get_handoff_target_by_name(target_agent_name)
            if result is None:
                yield {
                    "status": "error",
                    "message": f"Handoff target '{target_agent_name}' not found",
                }
                return

            session, meta = result
            if meta.default_tools:
                missing = [
                    n
                    for n in meta.default_tools
                    if runtime._tool_registry.get(n) is None
                ]
                if missing:
                    yield {
                        "status": "error",
                        "message": (
                            f"Delegation failed: target agent '{meta.name}' "
                            f"missing required tools: {', '.join(missing)}"
                        ),
                    }
                    return

            combined = f"Context: {context_summary}\n\nTask: {task_description}"

            runtime.run_background(
                session_id=session.session_id,
                agent=session.agent,
                user_input=[{"type": "text", "text": combined}],
                memory=session.memory,
                tools=session.tools,
                stop_event=session.stop_event,
            )

            yield {
                "status": "handoff",
                "message": "Task handed off to another agent. Results will be delivered when ready.",
            }

        return StreamingTool(
            name="handoff",
            description="Hand off a task to another agent. Use discover_agents first to find available agents.",
            parameters={
                "type": "object",
                "properties": {
                    "target_agent_name": {
                        "type": "string",
                        "description": "The name of the target agent to hand off to.",
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
                    "target_agent_name",
                    "context_summary",
                    "task_description",
                ],
            },
            fn=handoff_fn,
        )

    def create_discover_agents_tool(self) -> StreamingTool:
        runtime = self

        async def discover_fn() -> AsyncIterator[Any]:
            agents = [
                {
                    "name": meta.name,
                    "description": (
                        meta2.description
                        if (meta2 := runtime.registry.get(meta.name))
                        else ""
                    ),
                    "running": sid in runtime._background_tasks,
                }
                for sid, meta in runtime._handoff_meta.items()
            ]
            yield {
                "status": "ok",
                "agents": agents,
            }

        return StreamingTool(
            name="discover_agents",
            description="Discover available agents that can accept handoffs.",
            parameters={
                "type": "object",
                "properties": {},
            },
            fn=discover_fn,
        )

    def is_background_task_running(self, session_id: str) -> bool:
        return session_id in self._background_tasks

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    @property
    def registered_agents(self) -> list[Session]:
        return list(self._sessions.values())

    def drain_handoff_events(self, session_id: str) -> list["AgentEvent"]:
        meta = self._handoff_meta.get(session_id)
        if meta is None:
            return []
        events: list[AgentEvent] = []
        while not meta.event_queue.empty():
            try:
                events.append(meta.event_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return events

    def inject_runtime_tools(
        self,
        tools: list[Tool],
        *,
        tool_names: tuple[str, ...] = ("handoff", "discover_agents"),
    ) -> None:
        existing = {t.name for t in tools}
        if "handoff" in tool_names and "handoff" not in existing:
            tools.append(self.create_handoff_tool())
        if "discover_agents" in tool_names and "discover_agents" not in existing:
            tools.append(self.create_discover_agents_tool())

    def register_agent(
        self,
        name: str,
        description: str,
        system_prompt: str,
        llm_provider: "LLMProvider",
        tools: Sequence[Tool],
        agent_factory: Callable[..., Agent] | None = None,
        memory_factory: Callable[[str, str, str], Memory] | None = None,
        default_tools: Sequence[str] | None = None,
    ) -> str:
        import uuid

        from minimal_harness.agent.session import ConversationSession
        from minimal_harness.agent.simple import SimpleAgent
        from minimal_harness.memory import ConversationMemory

        factory = agent_factory or SimpleAgent
        mem_factory = memory_factory or (
            lambda sp, _, __: ConversationMemory(system_prompt=sp)
        )
        default_tools_list = list(default_tools) if default_tools else []
        session_tools = (
            self._resolve_tools(default_tools_list)
            if default_tools_list
            else list(tools)
        )
        session_id = uuid.uuid4().hex
        memory = mem_factory(system_prompt, name, session_id)
        agent = factory(llm_provider=llm_provider, tools=session_tools, memory=memory)
        session = ConversationSession(
            session_id=session_id,
            agent=agent,
            memory=memory,
            tools=session_tools,
            name=name,
        )
        self._sessions[session_id] = session
        self._handoff_meta[session_id] = _HandoffMeta(
            name=name,
            default_tools=default_tools_list or None,
        )
        self.registry.register(agent=agent, name=name, description=description)
        return session_id
