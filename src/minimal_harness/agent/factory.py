from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol, Sequence, runtime_checkable

from minimal_harness.agent.driver import RemoteAgentDriverFactory
from minimal_harness.types import (
    AgentMetadata,
    LocalAgentBinding,
    RemoteAgentBinding,
)

if TYPE_CHECKING:
    from minimal_harness.agent.middleware import Middleware
    from minimal_harness.agent.protocol import Agent
    from minimal_harness.llm.llm import LLMProvider


@runtime_checkable
class AgentFactory(Protocol):
    """Creates a concrete ``Agent`` from ``AgentMetadata``.

    Implement this protocol to customise how agent metadata is
    turned into an executable agent (e.g. wire up a custom
    ``RemoteAgentDriver`` or local agent type).
    """

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


class DefaultAgentFactory:
    """Default ``AgentFactory`` that handles all built-in binding types.

    Resolves local and remote bindings.  Users can register custom
    ``LocalAgentFactory`` implementations for specific agent types,
    and custom ``RemoteAgentDriverFactory`` implementations for
    specific driver names.

    LLM provider resolution is handled by ``llm_provider_resolver``,
    which receives ``AgentMetadata`` and returns an ``LLMProvider``.
    This enables per-agent provider/model selection (orchestration
    service) as well as single global providers (TUI, via a lambda
    that ignores metadata).
    """

    def __init__(
        self,
        llm_provider_resolver: Callable[[AgentMetadata], LLMProvider],
        driver_factories: dict[str, RemoteAgentDriverFactory] | None = None,
        local_agent_factories: dict[str, LocalAgentFactory] | None = None,
        middleware: Sequence[Middleware] = (),
    ) -> None:
        self._llm_provider_resolver = llm_provider_resolver
        self._driver_factories: dict[str, RemoteAgentDriverFactory] = dict(
            driver_factories or {}
        )
        self._local_agent_factories: dict[str, LocalAgentFactory] = {
            "simple": DefaultSimpleAgentFactory(),
            **(local_agent_factories or {}),
        }
        self._middleware = middleware

    def register_local_agent_factory(
        self, agent_type: str, factory: LocalAgentFactory
    ) -> None:
        self._local_agent_factories[agent_type] = factory

    def register_driver_factory(
        self, driver: str, factory: RemoteAgentDriverFactory
    ) -> None:
        self._driver_factories[driver] = factory

    def create(self, metadata: AgentMetadata, **kwargs: Any) -> Agent:
        match metadata.binding:
            case RemoteAgentBinding(driver=driver):
                factory = self._driver_factories.get(driver)
                if factory is None:
                    raise ValueError(
                        f"No driver factory registered for remote agent driver "
                        f"'{driver}'. "
                        f"Available: {list(self._driver_factories)}"
                    )
                from minimal_harness.agent.remote import RemoteAgent

                driver_instance = factory.create(metadata.binding)
                return RemoteAgent(driver=driver_instance)

            case None | LocalAgentBinding():
                pass

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
