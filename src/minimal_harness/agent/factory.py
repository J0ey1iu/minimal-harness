from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol, Sequence, runtime_checkable

from minimal_harness.types import (
    AgentMetadata,
)

if TYPE_CHECKING:
    from minimal_harness.agent.middleware import Middleware
    from minimal_harness.agent.protocol import Agent
    from minimal_harness.llm.llm import LLMProvider


@runtime_checkable
class AgentFactory(Protocol):
    """Creates a concrete ``Agent`` from ``AgentMetadata``."""

    def create(self, metadata: AgentMetadata, **kwargs: Any) -> Agent: ...


class LocalAgentFactory(Protocol):
    """Factory that creates a local ``Agent`` from metadata + provider."""

    def create(
        self,
        metadata: AgentMetadata,
        llm_provider: LLMProvider,
        middleware: Sequence[Middleware],
        **kwargs: Any,
    ) -> Agent: ...


class DefaultSimpleAgentFactory:
    """Default factory for ``agent_type="simple"`` local agents."""

    def create(
        self,
        metadata: AgentMetadata,
        llm_provider: LLMProvider,
        middleware: Sequence[Middleware],
        **kwargs: Any,
    ) -> Agent:
        from minimal_harness.agent.simple import SimpleAgent

        return SimpleAgent(
            llm_provider=llm_provider,
            max_iterations=kwargs.get("max_iterations", 100),
            middleware=middleware,
            emit_message_events=kwargs.get("emit_message_events", True),
        )


class CompactingAgentFactory:
    """Default factory for ``agent_type="compacting"`` local agents.

    Reads ``CompactionConfig`` from the ``compaction_config`` kwarg (injected
    by :class:`AgentRuntime`). Raises if the config is missing — running a
    ``compacting`` agent without summarizer/threshold is a configuration
    error, not a silent fallback.
    """

    def create(
        self,
        metadata: AgentMetadata,
        llm_provider: LLMProvider,
        middleware: Sequence[Middleware],
        **kwargs: Any,
    ) -> Agent:
        from minimal_harness.agent.compacting import CompactionAgent
        from minimal_harness.types import CompactionConfig

        config: CompactionConfig | None = kwargs.get("compaction_config")
        if config is None:
            raise ValueError(
                "agent_type='compacting' requires AgentRuntime to be "
                "constructed with a CompactionConfig (compaction_config=...)"
            )

        return CompactionAgent(
            llm_provider=llm_provider,
            summarizer=config.summarizer,
            prompt_token_threshold=config.prompt_token_threshold,
            keep_recent=config.keep_recent,
            max_iterations=kwargs.get("max_iterations", 100),
            middleware=middleware,
            emit_message_events=kwargs.get("emit_message_events", True),
        )


class DefaultAgentFactory:
    """Default ``AgentFactory`` that handles all built-in agent types.

    Resolves local bindings by dispatching to registered
    ``LocalAgentFactory`` implementations per ``agent_type``.

    LLM provider resolution is handled by ``llm_provider_resolver``,
    which receives ``AgentMetadata`` and returns an ``LLMProvider``.
    This enables per-agent provider/model selection (gateway
    service) as well as single global providers (TUI, via a lambda
    that ignores metadata).
    """

    def __init__(
        self,
        llm_provider_resolver: Callable[[AgentMetadata], LLMProvider],
        local_agent_factories: dict[str, LocalAgentFactory] | None = None,
        middleware: Sequence[Middleware] = (),
    ) -> None:
        self._llm_provider_resolver = llm_provider_resolver
        self._local_agent_factories: dict[str, LocalAgentFactory] = {
            "simple": DefaultSimpleAgentFactory(),
            "compacting": CompactingAgentFactory(),
            **(local_agent_factories or {}),
        }
        self._middleware = middleware

    def register_local_agent_factory(
        self, agent_type: str, factory: LocalAgentFactory
    ) -> None:
        self._local_agent_factories[agent_type] = factory

    def create(self, metadata: AgentMetadata, **kwargs: Any) -> Agent:
        llm_provider = self._llm_provider_resolver(metadata)

        local_factory = self._local_agent_factories.get(metadata.agent_type)
        if local_factory is None:
            raise ValueError(
                f"Unknown agent type: {metadata.agent_type}. "
                f"Available local agent types: {list(self._local_agent_factories)}"
            )

        return local_factory.create(
            metadata=metadata,
            llm_provider=llm_provider,
            middleware=self._middleware,
            **kwargs,
        )
