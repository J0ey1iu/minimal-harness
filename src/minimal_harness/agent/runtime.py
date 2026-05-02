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

from minimal_harness.memory import ConversationMemory
from minimal_harness.tool.base import StreamingTool
from minimal_harness.types import (
    AgentEnd,
    AgentEvent,
    LLMChunk,
    LLMEnd,
    ToolEnd,
    ToolProgress,
    ToolStart,
)

if TYPE_CHECKING:
    from minimal_harness.agent.protocol import Agent
    from minimal_harness.agent.registry import AgentRegistryProtocol
    from minimal_harness.memory import ExtendedInputContentPart, Memory
    from minimal_harness.tool.base import Tool


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
        agent_name: str | None = None,
    ) -> tuple[asyncio.Task, asyncio.Event, asyncio.Queue[AgentEvent | None]]: ...


class AgentRuntime:
    """Async task manager backed by an AgentRegistry.

    Creates agent discovery and handoff tools from the registry and
    injects them before each agent run.

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
    ) -> None:
        self._agent_registry = agent_registry

    def run(
        self,
        agent: Agent,
        memory: Memory | None,
        tools: Sequence[Tool],
        user_input: Iterable[ExtendedInputContentPart],
        agent_name: str | None = None,
    ) -> tuple[asyncio.Task, asyncio.Event, asyncio.Queue[AgentEvent | None]]:
        tools = self._inject_runtime_tools(list(tools), agent_name=agent_name)
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

    def _inject_runtime_tools(
        self, tools: list[Tool], agent_name: str | None = None
    ) -> list[Tool]:
        existing = {t.name for t in tools}
        if "handoff" not in existing:
            tools.append(
                _make_handoff_tool(
                    agent_registry=self._agent_registry,
                    run_fn=self.run,
                    delegating_agent_name=agent_name,
                )
            )
        if "discover_agents" not in existing:
            tools.append(_make_discover_agents_tool(self._agent_registry))
        return tools


def _make_handoff_tool(
    agent_registry: AgentRegistryProtocol,
    run_fn: Callable[
        ...,
        tuple[asyncio.Task, asyncio.Event, asyncio.Queue[AgentEvent | None]],
    ],
    delegating_agent_name: str | None = None,
) -> StreamingTool:
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
        if delegating_agent_name:
            combined = f"[Delegated by {delegating_agent_name}]{combined}"

        handoff_memory = ConversationMemory()
        task, stop_event, event_queue = run_fn(
            agent=metadata.agent,
            memory=handoff_memory,
            tools=list(metadata.tools),
            user_input=[{"type": "text", "text": combined}],
            agent_name=target_agent_name,
        )

        yield {
            "status": "handoff_started",
            "message": f"Starting delegated task to {target_agent_name}...",
        }

        final_result = None
        result_text = ""
        while True:
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if stop_event.is_set():
                    yield {
                        "status": "error",
                        "message": "Delegated task was interrupted",
                    }
                    break
                continue

            if event is None:
                break

            if isinstance(event, LLMChunk):
                content = event.chunk.content if event.chunk else ""
                if content:
                    result_text += content
                    yield {
                        "status": "progress",
                        "type": "text",
                        "content": content,
                    }
            elif isinstance(event, ToolStart):
                yield {
                    "status": "progress",
                    "type": "tool_start",
                    "tool_name": event.tool_call["function"]["name"],
                }
            elif isinstance(event, ToolProgress):
                yield {
                    "status": "progress",
                    "type": "tool_progress",
                    "tool_name": event.tool_call["function"]["name"],
                    "chunk": event.chunk,
                }
            elif isinstance(event, ToolEnd):
                result_str = event.result
                if isinstance(result_str, str):
                    result_text += (
                        f"\n[Tool: {event.tool_call['function']['name']} completed]"
                    )
                yield {
                    "status": "progress",
                    "type": "tool_end",
                    "tool_name": event.tool_call["function"]["name"],
                    "result": result_str,
                }
                final_result = result_str
            elif isinstance(event, LLMEnd):
                if event.content:
                    result_text = str(event.content)
            elif isinstance(event, AgentEnd):
                result_text = event.response or result_text

        yield {
            "status": "handoff_complete",
            "message": "Delegated task completed",
            "result": result_text or final_result,
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


def _make_discover_agents_tool(
    agent_registry: AgentRegistryProtocol,
) -> StreamingTool:
    async def discover_fn() -> AsyncIterator[Any]:
        agents_list = [
            {
                "name": m.name,
                "description": m.description,
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
