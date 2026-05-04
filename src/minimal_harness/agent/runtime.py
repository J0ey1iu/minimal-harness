from __future__ import annotations

import asyncio
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Iterable,
    Protocol,
    runtime_checkable,
)

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
    from minimal_harness.memory import ExtendedInputContentPart
    from minimal_harness.tool.base import StreamingTool, Tool, ToolRegistryProtocol

    LLMProviderFactory = Callable[[], Any]


@runtime_checkable
class AgentRuntimeProtocol(Protocol):
    """Async task manager for running agents.

    Implementations MUST provide agent_registry, memory_store, and tool_registry
    (as ``_agent_registry``, ``_memory_store``, ``_tool_registry`` attributes)
    so that ``run()`` can resolve everything from IDs alone.
    """

    _agent_registry: AgentRegistryProtocol
    _memory_store: Any
    _tool_registry: ToolRegistryProtocol

    def run(
        self,
        user_input: Iterable[ExtendedInputContentPart],
        agent_metadata_id: str,
        memory_id: str,
        agent_type: str | None = None,
    ) -> tuple[asyncio.Task, asyncio.Event, asyncio.Queue[AgentEvent | None]]: ...


class AgentRuntime:
    """Async task manager backed by registries and stores.

    Creates agent discovery and handoff tools from the registry and
    injects them before each agent run. Uses MemoryStore and ToolRegistry
    to look up the memory and tools needed for an agent run.

    Usage::

        task, stop_event, queue = runtime.run(
            user_input=[...],
            agent_metadata_id="general_assistant",
            memory_id="abc123",
        )
        while True:
            event = await queue.get()
            if event is None:
                break
            # process event
    """

    def __init__(
        self,
        agent_registry: AgentRegistryProtocol,
        memory_store: Any,
        tool_registry: Any,
        llm_provider_factory: LLMProviderFactory | None = None,
    ) -> None:
        self._agent_registry = agent_registry
        self._memory_store = memory_store
        self._tool_registry = tool_registry
        self._llm_provider_factory = llm_provider_factory

    def _create_agent(self, agent_type: str) -> Agent:
        if self._llm_provider_factory is None:
            raise RuntimeError("llm_provider_factory is required but was not provided")
        from minimal_harness.agent.simple import SimpleAgent

        if agent_type == "simple":
            llm_provider = self._llm_provider_factory()
            return SimpleAgent(llm_provider=llm_provider)
        raise ValueError(f"Unknown agent type: {agent_type}")

    def run(
        self,
        user_input: Iterable[ExtendedInputContentPart],
        agent_metadata_id: str,
        memory_id: str,
        agent_type: str | None = None,
    ) -> tuple[asyncio.Task, asyncio.Event, asyncio.Queue[AgentEvent | None]]:
        metadata = self._agent_registry.get(agent_metadata_id)
        if metadata is None:
            raise ValueError(
                f"Agent metadata '{agent_metadata_id}' not found in registry"
            )

        memory = self._memory_store.get_memory(memory_id)
        if memory is None:
            raise ValueError(f"Memory '{memory_id}' not found in store")

        agent_type = agent_type or metadata.agent_type

        tools: list[Tool] = [
            t
            for n in metadata.tool_names
            if (t := self._tool_registry.get(n)) is not None
        ]

        tools = self._inject_runtime_tools(tools, agent_metadata_id=agent_metadata_id)

        agent = self._create_agent(agent_type=agent_type)

        stop_event = asyncio.Event()
        event_queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()

        async def _run() -> None:
            try:
                async for event in agent.run(
                    user_input=user_input,
                    stop_event=stop_event,
                    memory=memory,
                    tools=tools,
                    system_prompt=metadata.system_prompt,
                ):
                    await event_queue.put(event)
            finally:
                await event_queue.put(None)

        task = asyncio.create_task(_run())
        return task, stop_event, event_queue

    def _inject_runtime_tools(
        self, tools: list[Tool], agent_metadata_id: str
    ) -> list[Tool]:
        existing = {t.name for t in tools}
        if "handoff" not in existing:
            tools.append(
                _make_handoff_tool(
                    agent_registry=self._agent_registry,
                    memory_store=self._memory_store,
                    run_fn=self.run,
                    delegating_agent_id=agent_metadata_id,
                )
            )
        if "discover_agents" not in existing:
            tools.append(_make_discover_agents_tool(self._agent_registry))
        return tools


def _make_handoff_tool(
    agent_registry: AgentRegistryProtocol,
    memory_store: Any,
    run_fn: Callable[
        ...,
        tuple[asyncio.Task, asyncio.Event, asyncio.Queue[AgentEvent | None]],
    ],
    delegating_agent_id: str | None = None,
) -> StreamingTool:
    from minimal_harness.tool.base import StreamingTool

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
        if delegating_agent_id:
            combined = f"[Delegated by {delegating_agent_id}]{combined}"

        handoff_memory_id = uuid.uuid4().hex
        memory_store.create_memory(
            memory_id=handoff_memory_id,
            agent_name=target_agent_name,
        )

        try:
            task, stop_event, event_queue = run_fn(
                user_input=[{"type": "text", "text": combined}],
                agent_metadata_id=metadata.metadata_id,
                memory_id=handoff_memory_id,
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
        finally:
            memory_store.delete_memory(handoff_memory_id)

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
    from minimal_harness.tool.base import StreamingTool

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
