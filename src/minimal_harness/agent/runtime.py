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

from minimal_harness.tool.base import StreamingTool

if TYPE_CHECKING:
    from minimal_harness.agent.protocol import Agent
    from minimal_harness.agent.registry import AgentRegistryProtocol
    from minimal_harness.memory import ExtendedInputContentPart, Memory
    from minimal_harness.tool.base import Tool
    from minimal_harness.types import AgentEvent


@runtime_checkable
class AgentRuntimeProtocol(Protocol):
    """Async task manager for running agents.

    The runtime is responsible for creating agent discovery and handoff
    tools and injecting them before each agent run.
    """

    def run(
        self,
        agent: Agent,
        memory: Memory | None,
        tools: Sequence[Tool],
        user_input: Iterable[ExtendedInputContentPart],
    ) -> tuple[asyncio.Task, asyncio.Event, asyncio.Queue[AgentEvent | None]]: ...


class AgentRuntime:
    """Async task manager backed by an AgentRegistry.

    Creates agent discovery and handoff tools from the registry and
    injects them before each agent run.  An optional ``on_handoff``
    callback receives notification when the handoff tool starts a
    sub-run so the caller can track it.

    Usage::

        task, stop_event, queue = runtime.run(agent, memory, tools, user_input)
        while True:
            event = await queue.get()
            if event is None:
                break
            # process event
    """

    def __init__(
        self,
        agent_registry: AgentRegistryProtocol,
        *,
        on_handoff: Callable[
            [str, asyncio.Task, asyncio.Event, asyncio.Queue[AgentEvent | None]],
            None,
        ]
        | None = None,
    ) -> None:
        self._agent_registry = agent_registry
        self._on_handoff = on_handoff

    @property
    def on_handoff(
        self,
    ) -> (
        Callable[
            [str, asyncio.Task, asyncio.Event, asyncio.Queue[AgentEvent | None]], None
        ]
        | None
    ):
        return self._on_handoff

    @on_handoff.setter
    def on_handoff(
        self,
        value: Callable[
            [str, asyncio.Task, asyncio.Event, asyncio.Queue[AgentEvent | None]], None
        ]
        | None,
    ) -> None:
        self._on_handoff = value

    def run(
        self,
        agent: Agent,
        memory: Memory | None,
        tools: Sequence[Tool],
        user_input: Iterable[ExtendedInputContentPart],
    ) -> tuple[asyncio.Task, asyncio.Event, asyncio.Queue[AgentEvent | None]]:
        tools = self._inject_runtime_tools(list(tools))
        stop_event = asyncio.Event()
        event_queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()

        async def _run() -> None:
            try:
                async for event in agent.run(
                    user_input=user_input,
                    stop_event=stop_event,
                    memory=memory,
                    tools=tools,
                ):
                    await event_queue.put(event)
            finally:
                await event_queue.put(None)

        task = asyncio.create_task(_run())
        return task, stop_event, event_queue

    def _inject_runtime_tools(self, tools: list[Tool]) -> list[Tool]:
        existing = {t.name for t in tools}
        if "handoff" not in existing:
            tools.append(self._make_handoff_tool())
        if "discover_agents" not in existing:
            tools.append(self._make_discover_agents_tool())
        return tools

    def _make_handoff_tool(self) -> StreamingTool:
        agent_registry = self._agent_registry
        on_handoff = self._on_handoff

        async def handoff_fn(
            target_agent_name: str, context_summary: str, task_description: str
        ) -> AsyncIterator[Any]:
            metadata = agent_registry.get(target_agent_name)
            if metadata is None:
                yield {
                    "status": "error",
                    "message": f"Handoff target '{target_agent_name}' not found",
                }
                return

            combined = f"Context: {context_summary}\n\nTask: {task_description}"
            task, stop_event, event_queue = self.run(
                agent=metadata.agent,
                memory=None,
                tools=[],
                user_input=[{"type": "text", "text": combined}],
            )

            if on_handoff is not None:
                on_handoff(target_agent_name, task, stop_event, event_queue)

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

    def _make_discover_agents_tool(self) -> StreamingTool:
        agent_registry = self._agent_registry

        async def discover_fn() -> AsyncIterator[Any]:
            agents_list = [
                {
                    "name": m.name,
                    "description": m.description,
                    "running": False,
                }
                for m in agent_registry.get_all()
            ]
            yield {
                "status": "ok",
                "agents": agents_list,
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
