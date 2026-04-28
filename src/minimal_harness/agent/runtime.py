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
from minimal_harness.agent.registry import AgentRegistryProtocol, HandoffTarget
from minimal_harness.tool.base import StreamingTool
from minimal_harness.tool.registry import ToolRegistry

if TYPE_CHECKING:
    from minimal_harness.client.built_in.memory import PersistentMemory
    from minimal_harness.llm import LLMProvider
    from minimal_harness.memory import ExtendedInputContentPart
    from minimal_harness.types import AgentEvent


@runtime_checkable
class AgentRuntimeProtocol(Protocol):
    def run(
        self,
        agent: Agent,
        user_input: Iterable[ExtendedInputContentPart],
        memory: PersistentMemory,
        tools: Sequence[StreamingTool],
        stop_event: asyncio.Event | None = None,
    ) -> AsyncIterator[AgentEvent]: ...

    def run_background(
        self,
        session_id: str,
        agent: Agent,
        user_input: Iterable[ExtendedInputContentPart],
        memory: PersistentMemory,
        tools: Sequence[StreamingTool],
    ) -> None: ...

    def register_agent(
        self,
        name: str,
        description: str,
        system_prompt: str,
        llm_provider: "LLMProvider",
        tools: Sequence[StreamingTool],
        agent_factory: Callable[..., Agent] | None = None,
        default_tools: Sequence[str] | None = None,
    ) -> str: ...

    def create_handoff_tool(self) -> StreamingTool: ...
    def create_discover_agents_tool(self) -> StreamingTool: ...

    def inject_runtime_tools(
        self,
        tools: list[StreamingTool],
        *,
        tool_names: tuple[str, ...] = ("handoff", "discover_agents"),
    ) -> None: ...

    def get_handoff_target(self, session_id: str) -> HandoffTarget | None: ...
    def list_handoff_targets(self) -> list[HandoffTarget]: ...
    def is_background_task_running(self, session_id: str) -> bool: ...
    def drain_handoff_events(self, session_id: str) -> list[AgentEvent]: ...
    def set_on_handoff_event(self, callback: Callable[[str], None] | None) -> None: ...


class AgentRuntime:
    def __init__(
        self,
        registry: AgentRegistryProtocol,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.registry = registry
        self._tool_registry = tool_registry or ToolRegistry()
        self._handoff_targets: dict[str, HandoffTarget] = {}
        self._background_tasks: dict[str, asyncio.Task[None]] = {}
        self._on_handoff_event: Callable[[str], None] | None = None

    def _resolve_tools(self, tool_names: Sequence[str]) -> list[StreamingTool]:
        return [
            t for name in tool_names if (t := self._tool_registry.get(name)) is not None
        ]

    def set_on_handoff_event(self, callback: Callable[[str], None] | None) -> None:
        self._on_handoff_event = callback

    async def run(
        self,
        agent: Agent,
        user_input: "Iterable[ExtendedInputContentPart]",
        memory: "PersistentMemory",
        tools: "Sequence[StreamingTool]",
        stop_event: asyncio.Event | None = None,
    ) -> AsyncIterator["AgentEvent"]:
        async for event in agent.run(
            user_input=user_input,
            stop_event=stop_event,
            memory=memory,
            tools=tools,
        ):
            yield event

    def run_background(
        self,
        session_id: str,
        agent: Agent,
        user_input: "Iterable[ExtendedInputContentPart]",
        memory: "PersistentMemory",
        tools: "Sequence[StreamingTool]",
    ) -> None:
        runtime = self
        self.inject_runtime_tools(list(tools))

        async def _task() -> None:
            async for event in runtime.run(
                agent=agent,
                user_input=user_input,
                memory=memory,
                tools=tools,
            ):
                target = runtime._handoff_targets.get(session_id)
                if target is not None and not target.event_queue.full():
                    target.event_queue.put_nowait(event)
                if runtime._on_handoff_event is not None:
                    runtime._on_handoff_event(session_id)
            runtime._background_tasks.pop(session_id, None)

        task = asyncio.create_task(_task())
        self._background_tasks[session_id] = task
        task.add_done_callback(lambda t: self._background_tasks.pop(session_id, None))
        if self._on_handoff_event is not None:
            self._on_handoff_event(session_id)

    def create_handoff_tool(self) -> StreamingTool:
        runtime = self

        async def handoff_fn(
            target_session_id: str, context_summary: str, task_description: str
        ) -> AsyncIterator[Any]:
            target = runtime._handoff_targets.get(target_session_id)
            if target is None:
                yield {
                    "status": "error",
                    "message": f"Handoff target {target_session_id} not found",
                }
                return

            if target.default_tools:
                missing = [
                    n
                    for n in target.default_tools
                    if runtime._tool_registry.get(n) is None
                ]
                if missing:
                    yield {
                        "status": "error",
                        "message": (
                            f"Delegation failed: target agent '{target.name}' "
                            f"missing required tools: {', '.join(missing)}"
                        ),
                    }
                    return

            combined = f"Context: {context_summary}\n\nTask: {task_description}"

            runtime.run_background(
                session_id=target_session_id,
                agent=target.agent,
                user_input=[{"type": "text", "text": combined}],
                memory=target.memory,
                tools=target.tools,
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
                    "running": s.session_id in runtime._background_tasks,
                }
                for s in runtime._handoff_targets.values()
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

    def is_background_task_running(self, session_id: str) -> bool:
        return session_id in self._background_tasks

    def get_handoff_target(self, session_id: str) -> HandoffTarget | None:
        return self._handoff_targets.get(session_id)

    def list_handoff_targets(self) -> list[HandoffTarget]:
        return list(self._handoff_targets.values())

    def drain_handoff_events(self, session_id: str) -> list["AgentEvent"]:
        target = self._handoff_targets.get(session_id)
        if target is None:
            return []
        events: list[AgentEvent] = []
        while not target.event_queue.empty():
            try:
                events.append(target.event_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return events

    def inject_runtime_tools(
        self,
        tools: list[StreamingTool],
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
        tools: Sequence[StreamingTool],
        agent_factory: Callable[..., Agent] | None = None,
        default_tools: Sequence[str] | None = None,
    ) -> str:
        from minimal_harness.agent.simple import SimpleAgent
        from minimal_harness.client.built_in.memory import PersistentMemory

        factory = agent_factory or SimpleAgent
        default_tools_list = list(default_tools) if default_tools else []
        session_tools = (
            self._resolve_tools(default_tools_list)
            if default_tools_list
            else list(tools)
        )
        memory = PersistentMemory(system_prompt=system_prompt, agent_name=name)
        agent = factory(llm_provider=llm_provider, tools=session_tools, memory=memory)
        handoff_target = HandoffTarget(
            session_id=memory._session_id,
            name=name,
            agent=agent,
            memory=memory,
            tools=session_tools,
            default_tools=default_tools_list or None,
        )
        self._handoff_targets[handoff_target.session_id] = handoff_target
        self.registry.register(agent=agent, name=name, description=description)
        return handoff_target.session_id
