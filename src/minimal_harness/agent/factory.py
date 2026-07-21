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

    settings_schema: dict[str, Any] = {
        "value": "simple",
        "display_name": "Simple",
        "display_name_zh": "简单",
        "settings_key": None,
        "settings_title": "",
        "settings_title_zh": "",
        "settings_fields": [],
    }

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

    settings_schema: dict[str, Any] = {
        "value": "compacting",
        "display_name": "Compacting",
        "display_name_zh": "压缩",
        "settings_key": "compaction",
        "settings_title": "Compaction Settings",
        "settings_title_zh": "压缩设置",
        "settings_fields": [
            {
                "key": "prompt_token_threshold",
                "display_name": "Prompt Token Threshold",
                "display_name_zh": "提示令牌阈值",
                "type": "number",
                "default": 8000,
                "placeholder": "8000",
                "placeholder_zh": "8000",
                "min": 0,
            },
            {
                "key": "keep_recent",
                "display_name": "Keep Recent",
                "display_name_zh": "保留最近消息数",
                "type": "number",
                "default": 6,
                "placeholder": "6",
                "placeholder_zh": "6",
                "min": 0,
            },
        ],
    }

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


class DummyAgentFactory:
    """Factory for ``agent_type="dummy"`` local agents (echo only, no LLM)."""

    settings_schema: dict[str, Any] = {
        "value": "dummy",
        "display_name": "Dummy",
        "display_name_zh": "回声",
        "settings_key": None,
        "settings_title": "",
        "settings_title_zh": "",
        "settings_fields": [],
    }

    def create(
        self,
        metadata: AgentMetadata,
        llm_provider: LLMProvider,
        middleware: Sequence[Middleware],
        **kwargs: Any,
    ) -> Agent:
        from minimal_harness.agent.dummy import DummyAgent

        return DummyAgent(
            llm_provider=llm_provider,
            max_iterations=kwargs.get("max_iterations", 100),
            middleware=middleware,
            emit_message_events=kwargs.get("emit_message_events", True),
        )


class ToolCompactingAgentFactory:
    """Factory for ``agent_type="tool_compacting"`` local agents.

    Reads ``ToolCompactionConfig`` from the ``tool_compaction_config``
    kwarg (injected by :class:`AgentRuntime`). Raises if the config is
    missing — running a ``tool_compacting`` agent without summarizer /
    threshold is a configuration error, not a silent fallback.
    """

    settings_schema: dict[str, Any] = {
        "value": "tool_compacting",
        "display_name": "Tool Compacting",
        "display_name_zh": "工具压缩",
        "settings_key": "tool_compaction",
        "settings_title": "Tool Compaction Settings",
        "settings_title_zh": "工具压缩设置",
        "settings_fields": [
            {
                "key": "round_compress",
                "display_name": "Round Compress",
                "display_name_zh": "轮次压缩",
                "type": "boolean",
                "default": True,
                "placeholder": "",
                "placeholder_zh": "",
            },
            {
                "key": "prompt_token_threshold",
                "display_name": "Prompt Token Threshold",
                "display_name_zh": "提示令牌阈值",
                "type": "number",
                "default": 0,
                "placeholder": "0 (disabled)",
                "placeholder_zh": "0 (禁用)",
                "min": 0,
            },
            {
                "key": "keep_recent",
                "display_name": "Keep Recent",
                "display_name_zh": "保留最近消息数",
                "type": "number",
                "default": 6,
                "placeholder": "6",
                "placeholder_zh": "6",
                "min": 0,
            },
        ],
    }

    def create(
        self,
        metadata: AgentMetadata,
        llm_provider: LLMProvider,
        middleware: Sequence[Middleware],
        **kwargs: Any,
    ) -> Agent:
        from minimal_harness.agent.tool_compacting import ToolCompactionAgent
        from minimal_harness.types import ToolCompactionConfig

        config: ToolCompactionConfig | None = kwargs.get("tool_compaction_config")
        if config is None:
            raise ValueError(
                "agent_type='tool_compacting' requires AgentRuntime to be "
                "constructed with a ToolCompactionConfig "
                "(tool_compaction_config=...)"
            )

        return ToolCompactionAgent(
            llm_provider=llm_provider,
            summarizer=config.summarizer,
            round_compress=config.round_compress,
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
            "dummy": DummyAgentFactory(),
            "tool_compacting": ToolCompactingAgentFactory(),
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


# ── Agent type schema discovery ──────────────────────────────────────


# Built-in factory classes and their registered type keys.
# Used by get_builtin_agent_type_schemas() to dynamically list agent types.
_BUILTIN_FACTORY_REGISTRATIONS: list[tuple[str, type]] = [
    ("simple", DefaultSimpleAgentFactory),
    ("dummy", DummyAgentFactory),
    ("compacting", CompactingAgentFactory),
    ("tool_compacting", ToolCompactingAgentFactory),
]


def get_builtin_agent_type_schemas() -> list[dict[str, Any]]:
    """Return settings schemas for all built-in agent types.

    Each factory class declares a ``settings_schema`` dict.  This function
    collects them so the management API can return the list dynamically
    without a hardcoded copy.
    """
    schemas: list[dict[str, Any]] = []
    for _type_key, factory_cls in _BUILTIN_FACTORY_REGISTRATIONS:
        schema = getattr(factory_cls, "settings_schema", None)
        if schema is not None:
            schemas.append(dict(schema))
    return schemas
