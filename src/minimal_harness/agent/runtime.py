from __future__ import annotations

import asyncio
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Protocol,
    runtime_checkable,
)

from minimal_harness.types import AgentEvent

if TYPE_CHECKING:
    from minimal_harness.agent.protocol import Agent
    from minimal_harness.agent.registry import AgentRegistryProtocol
    from minimal_harness.llm.llm import LLMProvider
    from minimal_harness.memory import ExtendedInputContentPart
    from minimal_harness.memory_store import MemoryStoreProtocol
    from minimal_harness.tool.base import Tool
    from minimal_harness.tool.registry import ToolRegistryProtocol

AgentFactory = Callable[..., "Agent"]


@runtime_checkable
class AgentRuntimeProtocol(Protocol):
    """Async task manager for running agents.

    Implementations MUST provide agent_registry, memory_store, and tool_registry
    attributes so that ``run()`` can resolve everything from IDs alone.
    """

    agent_registry: AgentRegistryProtocol
    memory_store: MemoryStoreProtocol
    tool_registry: ToolRegistryProtocol

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
    injects them before each agent run. Uses MemoryStoreProtocol and ToolRegistry
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
        memory_store: MemoryStoreProtocol,
        tool_registry: ToolRegistryProtocol,
        agent_factory: AgentFactory | None = None,
        llm_provider_factory: Callable[[], LLMProvider] | None = None,
    ) -> None:
        self.agent_registry = agent_registry
        self.memory_store = memory_store
        self.tool_registry = tool_registry
        self._agent_factory = agent_factory
        self._llm_provider_factory = llm_provider_factory
        self._register_runtime_tools()

    def _create_agent(self, agent_type: str) -> Agent:
        if self._agent_factory is not None:
            return self._agent_factory(agent_type=agent_type)
        if self._llm_provider_factory is None:
            raise RuntimeError("llm_provider_factory is required but was not provided")
        from minimal_harness.agent.simple import SimpleAgent
        from minimal_harness.settings import Settings

        if agent_type == "simple":
            llm_provider = self._llm_provider_factory()
            return SimpleAgent(
                llm_provider=llm_provider,
                max_iterations=Settings.max_iterations(),
            )
        raise ValueError(f"Unknown agent type: {agent_type}")

    def run(
        self,
        user_input: Iterable[ExtendedInputContentPart],
        agent_metadata_id: str,
        memory_id: str,
        agent_type: str | None = None,
    ) -> tuple[asyncio.Task, asyncio.Event, asyncio.Queue[AgentEvent | None]]:
        metadata = self.agent_registry.get(agent_metadata_id)
        if metadata is None:
            raise ValueError(
                f"Agent metadata '{agent_metadata_id}' not found in registry"
            )

        memory = self.memory_store.get_memory(memory_id)
        if memory is None:
            raise ValueError(f"Memory '{memory_id}' not found in store")

        agent_type = agent_type or metadata.agent_type

        tools: list[Tool] = [
            t
            for n in metadata.tool_names
            if (t := self.tool_registry.get(n)) is not None
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

    def _register_runtime_tools(self) -> None:
        if self.tool_registry.get("handoff") is None:
            self.tool_registry.register(
                _make_handoff_tool(
                    agent_registry=self.agent_registry,
                    memory_store=self.memory_store,
                    run_fn=self.run,
                    delegating_agent_id=None,
                )
            )
        if self.tool_registry.get("discover_agents") is None:
            self.tool_registry.register(_make_discover_agents_tool(self.agent_registry))

    def _inject_runtime_tools(
        self, tools: list[Tool], agent_metadata_id: str
    ) -> list[Tool]:
        existing = {t.name for t in tools}
        runtime_tool_names = {"handoff", "discover_agents"}
        for name in runtime_tool_names:
            if name not in existing:
                t = self.tool_registry.get(name)
                if t is not None:
                    tools.append(t)
        return tools


def _make_handoff_tool(
    agent_registry: AgentRegistryProtocol,
    memory_store: Any,
    run_fn: Callable[
        ...,
        tuple[asyncio.Task, asyncio.Event, asyncio.Queue[AgentEvent | None]],
    ],
    delegating_agent_id: str | None = None,
) -> Tool:
    from minimal_harness.tool.built_in.runtime_tools import make_handoff_tool

    return make_handoff_tool(
        agent_registry=agent_registry,
        memory_store=memory_store,
        run_fn=run_fn,
        delegating_agent_id=delegating_agent_id,
    )


def _make_discover_agents_tool(
    agent_registry: AgentRegistryProtocol,
) -> Tool:
    from minimal_harness.tool.built_in.runtime_tools import make_discover_agents_tool

    return make_discover_agents_tool(agent_registry=agent_registry)
