from __future__ import annotations

import asyncio
import contextvars
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Iterable,
    Protocol,
    Sequence,
    runtime_checkable,
)

from minimal_harness.types import AgentEvent

if TYPE_CHECKING:
    from minimal_harness.agent.middleware import Middleware
    from minimal_harness.agent.protocol import Agent
    from minimal_harness.agent.registry import AgentRegistryProtocol
    from minimal_harness.llm.llm import LLMProvider
    from minimal_harness.memory import ExtendedInputContentPart
    from minimal_harness.memory_store import SessionStoreProtocol
    from minimal_harness.tool.base import Tool
    from minimal_harness.tool.registry import ToolRegistryProtocol

_current_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "_mh_run_context", default={}
)


def get_current_locale() -> str:
    """Get the current locale from the active run context.

    Returns the locale string (e.g. ``"zh"``, ``"en"``) if set,
    or an empty string if no locale was configured.
    """
    ctx = _current_context.get()
    return ctx.get("locale", "")


AgentFactory = Callable[..., "Agent"]


@runtime_checkable
class AgentRuntimeProtocol(Protocol):
    """Async task manager for running agents.

    Implementations MUST provide agent_registry, session_store, and tool_registry
    attributes so that ``run()`` can resolve everything from IDs alone.
    """

    agent_registry: AgentRegistryProtocol
    session_store: SessionStoreProtocol
    tool_registry: ToolRegistryProtocol

    async def run(
        self,
        user_input: Iterable[ExtendedInputContentPart],
        agent_metadata_id: str,
        memory_id: str,
        agent_type: str | None = None,
        tool_names: list[str] | None = None,
        context: dict[str, Any] | None = None,
        llm_kwargs: dict[str, Any] | None = None,
    ) -> tuple[asyncio.Task, asyncio.Event, asyncio.Queue[AgentEvent | None]]: ...


class AgentRuntime:
    """Async task manager backed by registries and stores.

    Uses SessionStoreProtocol and ToolRegistry
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
        session_store: SessionStoreProtocol,
        tool_registry: ToolRegistryProtocol,
        agent_factory: AgentFactory | None = None,
        llm_provider_factory: Callable[[], LLMProvider] | None = None,
        middleware: Sequence[Middleware] = (),
    ) -> None:
        self.agent_registry = agent_registry
        self.session_store = session_store
        self.tool_registry = tool_registry
        self._agent_factory = agent_factory
        self._llm_provider_factory = llm_provider_factory
        self._middleware = middleware

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
                middleware=self._middleware,
            )
        raise ValueError(f"Unknown agent type: {agent_type}")

    async def run(
        self,
        user_input: Iterable[ExtendedInputContentPart],
        agent_metadata_id: str,
        memory_id: str,
        agent_type: str | None = None,
        tool_names: list[str] | None = None,
        context: dict[str, Any] | None = None,
        llm_kwargs: dict[str, Any] | None = None,
    ) -> tuple[asyncio.Task, asyncio.Event, asyncio.Queue[AgentEvent | None]]:
        metadata = await self.agent_registry.get(agent_metadata_id)
        if metadata is None:
            raise ValueError(
                f"Agent metadata '{agent_metadata_id}' not found in registry"
            )

        session = await self.session_store.get_session(memory_id)
        if session is None:
            raise ValueError(f"Session '{memory_id}' not found in store")

        agent_type = agent_type or metadata.agent_type

        resolved_tool_names = (
            tool_names if tool_names is not None else metadata.tool_names
        )
        tools: list[Tool] = []
        for n in resolved_tool_names:
            t = await self.tool_registry.get(n)
            if t is not None:
                tools.append(t)

        agent = self._create_agent(agent_type=agent_type)

        base = _current_context.get()
        run_context = {**base, **(context or {})}
        token = _current_context.set(run_context)

        stop_event = asyncio.Event()
        event_queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        done_event = asyncio.Event()

        async def _run() -> None:
            try:
                locale = run_context.get("locale", "")
                run_kwargs: dict[str, Any] = {}
                if llm_kwargs is not None:
                    run_kwargs["llm_kwargs"] = llm_kwargs
                async for event in agent.run(
                    user_input=user_input,
                    stop_event=stop_event,
                    memory=session,
                    tools=tools,
                    system_prompt=metadata.resolve_system_prompt(locale),
                    context=run_context,
                    **run_kwargs,
                ):
                    await event_queue.put(event)
            finally:
                await event_queue.put(None)
                done_event.set()

        task = asyncio.create_task(_run())
        task.done_event = done_event  # type: ignore[attr-defined]
        _current_context.reset(token)
        return task, stop_event, event_queue

    async def register_runtime_tools(self) -> None:
        if await self.tool_registry.get("handoff") is None:
            await self.tool_registry.register(
                _make_handoff_tool(
                    agent_registry=self.agent_registry,
                    session_store=self.session_store,
                    run_fn=self.run,
                    delegating_agent_id=None,
                )
            )
        if await self.tool_registry.get("discover_agents") is None:
            await self.tool_registry.register(
                _make_discover_agents_tool(self.agent_registry)
            )


def _make_handoff_tool(
    agent_registry: AgentRegistryProtocol,
    session_store: Any,
    run_fn: Callable[
        ...,
        Awaitable[tuple[asyncio.Task, asyncio.Event, asyncio.Queue[AgentEvent | None]]],
    ],
    delegating_agent_id: str | None = None,
) -> Tool:
    from minimal_harness.tool.built_in.runtime_tools import make_handoff_tool

    return make_handoff_tool(
        agent_registry=agent_registry,
        session_store=session_store,
        run_fn=run_fn,
        delegating_agent_id=delegating_agent_id,
    )


def _make_discover_agents_tool(
    agent_registry: AgentRegistryProtocol,
) -> Tool:
    from minimal_harness.tool.built_in.runtime_tools import make_discover_agents_tool

    return make_discover_agents_tool(agent_registry=agent_registry)
