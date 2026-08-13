from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

from minimal_harness.types import AgentMetadata

if TYPE_CHECKING:
    from minimal_harness.agent.middleware import Middleware
    from minimal_harness.agent.protocol import Agent
    from minimal_harness.llm.llm import LLMProvider


# ── Built-in agent type constructors ────────────────────────────────


def _build_simple_agent(
    llm_provider: LLMProvider,
    middleware: Sequence[Middleware],
    **kwargs: Any,
) -> Agent:
    from minimal_harness.agent.simple import SimpleAgent

    return SimpleAgent(
        llm_provider=llm_provider,
        max_iterations=kwargs.get("max_iterations", 2000),
        middleware=middleware,
        emit_message_events=kwargs.get("emit_message_events", True),
    )


def _build_compacting_agent(
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
        max_iterations=kwargs.get("max_iterations", 2000),
        middleware=middleware,
        emit_message_events=kwargs.get("emit_message_events", True),
    )


def _build_dummy_agent(
    llm_provider: LLMProvider,
    middleware: Sequence[Middleware],
    **kwargs: Any,
) -> Agent:
    from minimal_harness.agent.dummy import DummyAgent

    return DummyAgent(
        llm_provider=llm_provider,
        max_iterations=kwargs.get("max_iterations", 2000),
        middleware=middleware,
        emit_message_events=kwargs.get("emit_message_events", True),
    )


def _build_tool_compacting_agent(
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
        prompt_token_threshold=config.prompt_token_threshold,
        keep_recent=config.keep_recent,
        max_iterations=kwargs.get("max_iterations", 2000),
        middleware=middleware,
        emit_message_events=kwargs.get("emit_message_events", True),
    )


_AGENT_BUILDERS: dict[str, Callable[..., Agent]] = {
    "simple": _build_simple_agent,
    "compacting": _build_compacting_agent,
    "dummy": _build_dummy_agent,
    "tool_compacting": _build_tool_compacting_agent,
}

# Schemas exposed via the management API so the frontend can render
# agent-type-specific settings forms. Kept separate from builders
# because they're a different concern (UI metadata vs construction).
_AGENT_SCHEMAS: dict[str, dict[str, Any]] = {
    "simple": {
        "value": "simple",
        "display_name": "Simple",
        "display_name_zh": "简单",
        "settings_key": None,
        "settings_title": "",
        "settings_title_zh": "",
        "settings_fields": [],
    },
    "compacting": {
        "value": "compacting",
        "display_name": "Compacting",
        "display_name_zh": "压缩",
        "settings_key": "compaction",
        "settings_title": "Compaction Settings",
        "settings_title_zh": "压缩设置",
        "settings_fields": [
            {
                "key": "compaction_prompt",
                "display_name": "Compaction Prompt",
                "display_name_zh": "压缩提示词",
                "type": "longtext",
                "default": "",
                "placeholder": "Custom summarization instruction (leave empty for default)",
                "placeholder_zh": "自定义压缩指令（留空使用默认）",
            },
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
    },
    "dummy": {
        "value": "dummy",
        "display_name": "Dummy",
        "display_name_zh": "回声",
        "settings_key": None,
        "settings_title": "",
        "settings_title_zh": "",
        "settings_fields": [],
    },
    "tool_compacting": {
        "value": "tool_compacting",
        "display_name": "Tool Compacting",
        "display_name_zh": "工具压缩",
        "settings_key": "tool_compaction",
        "settings_title": "Tool Compaction Settings",
        "settings_title_zh": "工具压缩设置",
        "settings_fields": [
            {
                "key": "compaction_prompt",
                "display_name": "Compaction Prompt",
                "display_name_zh": "压缩提示词",
                "type": "longtext",
                "default": "",
                "placeholder": "Custom summarization instruction (leave empty for default)",
                "placeholder_zh": "自定义压缩指令（留空使用默认）",
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
    },
}


class AgentFactory:
    """Builds agents from metadata by dispatching to the right agent type constructor.

    ``llm_provider_resolver`` is a callable that maps ``AgentMetadata`` to an
    ``LLMProvider`` — the gateway uses this for per-agent provider/model selection;
    simpler consumers (TUI, test fixtures) can use a lambda that ignores metadata.
    """

    def __init__(
        self,
        llm_provider_resolver: Callable[[AgentMetadata], LLMProvider],
        middleware: Sequence[Middleware] = (),
    ) -> None:
        self._llm_provider_resolver = llm_provider_resolver
        self._middleware = middleware

    def create(self, metadata: AgentMetadata, **kwargs: Any) -> Agent:
        agent_type = metadata.agent_type
        builder = _AGENT_BUILDERS.get(agent_type)
        if builder is None:
            raise ValueError(
                f"Unknown agent type: {agent_type!r}. "
                f"Available types: {list(_AGENT_BUILDERS)}."
            )
        llm_provider = self._llm_provider_resolver(metadata)
        return builder(
            llm_provider=llm_provider,
            middleware=self._middleware,
            **kwargs,
        )


def get_builtin_agent_type_schemas() -> list[dict[str, Any]]:
    """Return settings schemas for all built-in agent types.

    The management API uses this to let the frontend render
    agent-type-specific configuration forms dynamically.
    """
    return [dict(s) for s in _AGENT_SCHEMAS.values()]
