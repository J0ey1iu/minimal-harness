from __future__ import annotations

import asyncio
import contextvars
import logging
import time
import uuid
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

from minimal_harness.agent.factory import AgentFactory
from minimal_harness.tool.factory import DefaultToolFactory, ToolFactory
from minimal_harness.agent._compaction import build_summarizer
from minimal_harness.types import (
    AgentEnd,
    AgentEvent,
    AgentMetadata,
    CompactionConfig,
    CompactionEnd,
    CompactionEvent,
    CompactionSettings,
    ToolCompactionConfig,
    ToolCompactionSettings,
)

if TYPE_CHECKING:
    from minimal_harness.agent.middleware import Middleware
    from minimal_harness.agent.protocol import Agent
    from minimal_harness.agent.registry import AgentRegistryProtocol
    from minimal_harness.llm.llm import LLMProvider
    from minimal_harness.memory import (
        ExtendedInputContentPart,
        Memory,
        MemoryStoreProtocol,
    )
    from minimal_harness.tool.base import Tool
    from minimal_harness.tool.factory import ToolExecutorFactory
    from minimal_harness.tool.registry import ToolRegistryProtocol

logger = logging.getLogger(__name__)

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


@runtime_checkable
class AgentRuntimeProtocol(Protocol):
    """Async task manager for running agents.

    Implementations MUST provide agent_registry, session_store, and tool_registry
    attributes so that ``run()`` can resolve everything from IDs alone.
    """

    agent_registry: AgentRegistryProtocol
    session_store: MemoryStoreProtocol
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

    def compact_session(self, memory_id: str) -> AsyncIterator[CompactionEvent]: ...


class AgentRuntime:
    """Async task manager backed by registries and stores.

    Uses MemoryStoreProtocol and ToolRegistry
    to look up the memory and tools needed for an agent run.
    """

    def __init__(
        self,
        agent_registry: AgentRegistryProtocol,
        session_store: MemoryStoreProtocol,
        tool_registry: ToolRegistryProtocol,
        llm_provider_resolver: Callable[[AgentMetadata], LLMProvider],
        agent_factory: AgentFactory | None = None,
        tool_factory: ToolFactory | None = None,
        middleware: Sequence[Middleware] = (),
        tool_executor_factories: dict[str, ToolExecutorFactory] | None = None,
        emit_message_events: bool = True,
        default_compaction_settings: CompactionSettings | None = None,
    ) -> None:
        self.agent_registry = agent_registry
        self.session_store = session_store
        self.tool_registry = tool_registry
        self._llm_provider_resolver = llm_provider_resolver
        self._tool_factory: ToolFactory = tool_factory or DefaultToolFactory(
            executor_factories=tool_executor_factories
        )
        self._middleware = middleware
        self._emit_message_events = emit_message_events
        self._default_compaction_settings: CompactionSettings = (
            default_compaction_settings
            if default_compaction_settings is not None
            else CompactionSettings()
        )
        self._agent_factory: AgentFactory = agent_factory or AgentFactory(
            llm_provider_resolver=llm_provider_resolver,
            middleware=middleware,
        )

    def register_tool_executor(self, driver: str, factory: ToolExecutorFactory) -> None:
        if isinstance(self._tool_factory, DefaultToolFactory):
            self._tool_factory.register_executor_factory(driver, factory)

    def _create_agent(self, metadata: AgentMetadata) -> Agent:
        # Build a CompactionConfig for compacting agents: the
        # summarizer is built from the runtime's LLM provider
        # resolver using the built-in ``build_summarizer``, and the
        # threshold / ``keep_recent`` come from the agent's own
        # ``CompactionSettings``. Downstream user-registered
        # factories may override this and ignore the runtime-provided
        # ``compaction_config`` kwarg.
        kwargs: dict[str, Any] = {
            "emit_message_events": self._emit_message_events,
        }
        if metadata.agent_type == "compacting":
            settings = CompactionSettings(
                {**self._default_compaction_settings, **(metadata.compaction or {})}
            )
            llm_provider = self._llm_provider_resolver(metadata)
            kwargs["compaction_config"] = CompactionConfig(
                summarizer=build_summarizer(
                    llm_provider,
                    metadata.system_prompt,
                    system_prompt_locale=metadata.system_prompt_locale,
                    summary_prompt=settings.get("compaction_prompt"),
                    summary_prompt_locale=settings.get("compaction_prompt_locale"),
                ),
                prompt_token_threshold=int(
                    settings.get("prompt_token_threshold", 8000)
                ),
                keep_recent=int(settings.get("keep_recent", 6)),
            )
        elif metadata.agent_type == "tool_compacting":
            settings = ToolCompactionSettings({**(metadata.tool_compaction or {})})
            llm_provider = self._llm_provider_resolver(metadata)
            kwargs["tool_compaction_config"] = ToolCompactionConfig(
                summarizer=build_summarizer(
                    llm_provider,
                    metadata.system_prompt,
                    system_prompt_locale=metadata.system_prompt_locale,
                    summary_prompt=settings.get("compaction_prompt"),
                    summary_prompt_locale=settings.get("compaction_prompt_locale"),
                ),
                prompt_token_threshold=int(settings.get("prompt_token_threshold", 0)),
                keep_recent=int(settings.get("keep_recent", 6)),
            )
        return self._agent_factory.create(metadata, **kwargs)

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
        correlation_id = uuid.uuid4().hex[:12]
        metadata = await self.agent_registry.get(agent_metadata_id)
        logger.info(
            "agent.run.start agent=%s session=%s tools=%s context_keys=%s",
            agent_metadata_id,
            memory_id,
            tool_names or [],
            list(context.keys()) if context else [],
        )
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
                try:
                    tools.append(self._tool_factory.create(tool_meta))
                except ValueError as e:
                    logger.warning(
                        "tool.create.error name=%s reason=%s",
                        n,
                        e,
                    )

        if not tools and resolved_tool_names:
            logger.warning(
                "tool.create.all_failed count=%d",
                len(resolved_tool_names),
            )

        agent = self._create_agent(metadata=metadata)

        base = _current_context.get()
        run_context = {**base, **(context or {}), "correlation_id": correlation_id}

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
                logger.exception("agent.run.error")
                await event_queue.put(
                    AgentEnd(
                        response="",
                        time_taken=time.time() - _run_start,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
            finally:
                elapsed = time.time() - _run_start
                logger.info("agent.run.end duration=%.2fs", elapsed)
                _current_context.reset(ctxtoken)
                await event_queue.put(None)
                done_event.set()

        task = asyncio.create_task(_run())
        task.done_event = done_event  # type: ignore[attr-defined]
        return task, stop_event, event_queue

    async def compact_session(
        self,
        memory_id: str,
    ) -> AsyncIterator[CompactionEvent]:
        """Run a manual compaction on an existing session.

        This is the public entry point used by the ``/compact`` slash
        command and any caller that wants to fold a session's history
        outside the agent loop. The output is the same
        ``CompactionStart / CompactionChunk / CompactionEnd`` event
        stream the agent emits when its threshold is crossed, so
        downstream consumers (display layer, persistence, replay) do
        not need a separate code path for manual vs. automatic
        compaction.

        The summarizer is built from the built-in ``build_summarizer``
        and the LLM provider resolved through the runtime's provider
        resolver — same wiring the agent loop uses. The threshold and
        ``keep_recent`` come from the session's owning agent's
        :class:`CompactionSettings`, falling back to the runtime's
        ``default_compaction_settings` if the agent has none.

        Concurrent calls against the same session are not safe — the
        buffer rebuild inside ``Memory.compact()`` is not atomic with
        ``Memory.add_message()``. Callers should refuse to compact
        while a run is in flight (the TUI's ``/compact`` already does
        this via ``SessionStatus.RUNNING``).
        """
        session: Memory | None = await self.session_store.get_session(memory_id)
        if session is None:
            raise ValueError(f"Session '{memory_id}' not found in store")

        # Resolve settings: agent-level CompactionSettings/ToolCompactionSettings
        # take precedence, runtime defaults are the fallback.
        # agent_type == "compacting" stores settings in metadata.compaction,
        # agent_type == "tool_compacting" stores them in metadata.tool_compaction.
        settings: CompactionSettings = CompactionSettings(
            self._default_compaction_settings
        )
        agent_name = getattr(session, "agent_name", "") or ""
        metadata = None
        if agent_name:
            metadata = await self.agent_registry.get(agent_name)
            if metadata is not None:
                if metadata.agent_type == "tool_compacting" and metadata.tool_compaction is not None:
                    # Merge tool_compaction settings on top of defaults.
                    settings = CompactionSettings(
                        {**self._default_compaction_settings, **metadata.tool_compaction}
                    )
                elif metadata.compaction is not None:
                    # Merge: agent's settings override defaults.
                    settings = CompactionSettings(
                        {**self._default_compaction_settings, **metadata.compaction}
                    )

        keep_recent = int(settings.get("keep_recent", 6))
        total_tokens = session.get_message_usage().get("total_tokens", 0)

        # Build the summarizer from the same LLM provider the agent
        # loop would use. We construct a minimal AgentMetadata-shaped
        # lookup so the resolver is happy even if the agent is no
        # longer in the registry.
        if agent_name:
            metadata = await self.agent_registry.get(agent_name)
        else:
            metadata = None
        if metadata is not None:
            llm_provider = self._llm_provider_resolver(metadata)
            system_prompt = metadata.system_prompt
        else:
            # No agent metadata — try to resolve using a stub so the
            # summarizer can still build (it only needs the LLM
            # client config, not the full agent).
            from minimal_harness.types import AgentMetadata

            stub = AgentMetadata(
                name=agent_name or "compact",
                provider="openai",
                model="",
            )
            llm_provider = self._llm_provider_resolver(stub)
            system_prompt = ""

        summarizer = build_summarizer(
            llm_provider,
            system_prompt,
            system_prompt_locale=metadata.system_prompt_locale if metadata else None,
            summary_prompt=settings.get("compaction_prompt"),
            summary_prompt_locale=settings.get("compaction_prompt_locale"),
        )

        logger.info(
            "agent.compact.manual session=%s agent=%s threshold=%s keep_recent=%d",
            memory_id,
            agent_name,
            settings.get("prompt_token_threshold"),
            keep_recent,
        )

        succeeded = False
        async for evt in session.compact(
            summarizer,
            keep_recent,
            total_tokens=total_tokens,
        ):
            if isinstance(evt, CompactionEnd) and evt.error is None and evt.summary:
                succeeded = True
            yield evt

        if succeeded:
            session.reset_message_usage()

    async def run_batch(
        self,
        user_input: Iterable[ExtendedInputContentPart],
        agent_metadata_id: str,
        memory_id: str,
        agent_type: str | None = None,
        tool_names: list[str] | None = None,
        context: dict[str, Any] | None = None,
        llm_kwargs: dict[str, Any] | None = None,
    ) -> list[AgentEvent]:
        task, stop_event, queue = await self.run(
            user_input=user_input,
            agent_metadata_id=agent_metadata_id,
            memory_id=memory_id,
            agent_type=agent_type,
            tool_names=tool_names,
            context=context,
            llm_kwargs=llm_kwargs,
        )
        events: list[AgentEvent] = []
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                events.append(event)
        finally:
            stop_event.set()
            await task
        return events
