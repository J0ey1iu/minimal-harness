from __future__ import annotations

import asyncio
import contextvars
import time
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

from minimal_harness.agent.driver import (
    DefaultAgentDriverFactory,
    RemoteAgentDriverFactory,
)
from minimal_harness.tool.factory import DefaultToolFactory, ToolFactory
from minimal_harness.types import (
    AgentEnd,
    AgentEvent,
    AgentMetadata,
    LocalAgentBinding,
    RemoteAgentBinding,
    ToolMetadata,
)

if TYPE_CHECKING:
    from minimal_harness.agent.middleware import Middleware
    from minimal_harness.agent.protocol import Agent
    from minimal_harness.agent.registry import AgentRegistryProtocol
    from minimal_harness.llm.llm import LLMProvider
    from minimal_harness.memory import ExtendedInputContentPart
    from minimal_harness.memory_store import SessionStoreProtocol
    from minimal_harness.tool.base import Tool
    from minimal_harness.tool.factory import ToolExecutorFactory
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
    """

    def __init__(
        self,
        agent_registry: AgentRegistryProtocol,
        session_store: SessionStoreProtocol,
        tool_registry: ToolRegistryProtocol,
        agent_factory: AgentFactory | None = None,
        tool_factory: ToolFactory | None = None,
        llm_provider_factory: Callable[[], LLMProvider] | None = None,
        middleware: Sequence[Middleware] = (),
        agent_driver_factories: dict[str, RemoteAgentDriverFactory] | None = None,
        tool_executor_factories: dict[str, ToolExecutorFactory] | None = None,
    ) -> None:
        self.agent_registry = agent_registry
        self.session_store = session_store
        self.tool_registry = tool_registry
        self._agent_factory = agent_factory
        self._tool_factory: ToolFactory = tool_factory or DefaultToolFactory(
            executor_factories=tool_executor_factories
        )
        self._llm_provider_factory = llm_provider_factory
        self._middleware = middleware
        self._agent_driver_factories: dict[str, RemoteAgentDriverFactory] = {
            "default": DefaultAgentDriverFactory(),
            **(agent_driver_factories or {}),
        }

    def register_agent_driver(
        self, driver: str, factory: RemoteAgentDriverFactory
    ) -> None:
        self._agent_driver_factories[driver] = factory

    def register_tool_executor(self, driver: str, factory: ToolExecutorFactory) -> None:
        if isinstance(self._tool_factory, DefaultToolFactory):
            self._tool_factory.register_executor_factory(driver, factory)

    def _create_agent(self, metadata: AgentMetadata) -> Agent:
        match metadata.binding:
            case RemoteAgentBinding(driver=driver):
                factory = self._agent_driver_factories.get(driver)
                if factory is None:
                    raise ValueError(
                        f"No driver factory registered for remote agent driver "
                        f"'{driver}'. "
                        f"Available: {list(self._agent_driver_factories)}"
                    )
                from minimal_harness.agent.remote import RemoteAgent

                driver_instance = factory.create(metadata.binding)
                return RemoteAgent(driver=driver_instance)

            case None | LocalAgentBinding():
                pass

        if self._agent_factory is not None:
            return self._agent_factory(agent_type=metadata.agent_type)
        if self._llm_provider_factory is None:
            raise RuntimeError("llm_provider_factory is required but was not provided")
        from minimal_harness.agent.simple import SimpleAgent
        from minimal_harness.settings import Settings

        if metadata.agent_type == "simple":
            llm_provider = self._llm_provider_factory()
            return SimpleAgent(
                llm_provider=llm_provider,
                max_iterations=Settings.max_iterations(),
                middleware=self._middleware,
            )
        raise ValueError(f"Unknown agent type: {metadata.agent_type}")

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

        resolved_tool_names = (
            tool_names if tool_names is not None else metadata.tool_names
        )
        tools: list[Tool] = []
        for n in resolved_tool_names:
            tool_meta = await self.tool_registry.get(n)
            if tool_meta is not None:
                tools.append(self._tool_factory.create(tool_meta))

        agent = self._create_agent(metadata=metadata)

        base = _current_context.get()
        run_context = {**base, **(context or {})}

        stop_event = asyncio.Event()
        event_queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        done_event = asyncio.Event()

        async def _run() -> None:
            ctxtoken = _current_context.set(run_context)
            _run_start = time.time()
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
            except asyncio.CancelledError:
                event_queue.put_nowait(
                    AgentEnd(
                        response="",
                        time_taken=time.time() - _run_start,
                        interrupted=True,
                    )
                )
            except Exception as exc:
                await event_queue.put(
                    AgentEnd(
                        response="",
                        time_taken=time.time() - _run_start,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
            finally:
                _current_context.reset(ctxtoken)
                await event_queue.put(None)
                done_event.set()

        task = asyncio.create_task(_run())
        task.done_event = done_event  # type: ignore[attr-defined]
        return task, stop_event, event_queue

    async def register_runtime_tools(self) -> None:
        if await self.tool_registry.get("handoff") is None:
            tool = _make_handoff_tool(
                agent_registry=self.agent_registry,
                session_store=self.session_store,
                run_fn=self.run,
                delegating_agent_id=None,
            )
            await self.tool_registry.register(
                _streaming_tool_to_metadata(tool, is_runtime=True)
            )
        if await self.tool_registry.get("discover_agents") is None:
            tool = _make_discover_agents_tool(self.agent_registry)
            await self.tool_registry.register(
                _streaming_tool_to_metadata(tool, is_runtime=True)
            )


def _streaming_tool_to_metadata(tool: Tool, *, is_runtime: bool = True) -> ToolMetadata:
    from minimal_harness.types import LocalToolBinding

    return ToolMetadata(
        name=tool.name,
        display_name=tool.display_name,
        description=tool.description,
        parameters=tool.parameters,
        metadata_id=tool.name,
        display_name_locale=tool.display_name_locale,
        description_locale=tool.description_locale,
        binding=LocalToolBinding(fn=getattr(tool, "fn", None)) if is_runtime else None,
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
