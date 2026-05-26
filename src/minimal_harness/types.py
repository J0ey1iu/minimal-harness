from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Literal,
    TypedDict,
    TypeVar,
    Union,
)

if TYPE_CHECKING:
    from minimal_harness.memory import ExtendedInputContentPart, Message

T = TypeVar("T")

ChunkCallback = Callable[[T | None, bool], Awaitable[None]]

# Callable that returns auth headers lazily at request time.
# Used by RemoteToolBinding / RemoteAgentBinding so that auth credentials
# are resolved right before each outbound HTTP call, not at binding creation.
ExtraHeadersProvider = Callable[[], Awaitable[dict[str, str]]]


# ── Bindings (execution HOW) ──────────────────────────────────────────


@dataclass
class LocalToolBinding:
    type: Literal["local"] = "local"
    fn: StreamingToolFunction | None = None


@dataclass
class ExternalScriptToolBinding:
    type: Literal["external_script"] = "external_script"
    script_path: str = ""


@dataclass
class RemoteToolBinding:
    type: Literal["remote"] = "remote"
    url: str = ""
    driver: str = "default"
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0
    extra_headers_provider: ExtraHeadersProvider | None = None

    def __post_init__(self) -> None:
        if not self.url:
            raise ValueError("url must not be empty for RemoteToolBinding")


ToolBinding = LocalToolBinding | ExternalScriptToolBinding | RemoteToolBinding


@dataclass
class LocalAgentBinding:
    type: Literal["local"] = "local"


@dataclass
class RemoteAgentBinding:
    type: Literal["remote"] = "remote"
    url: str = ""
    driver: str = "default"
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 120.0
    extra_headers_provider: ExtraHeadersProvider | None = None

    def __post_init__(self) -> None:
        if not self.url:
            raise ValueError("url must not be empty for RemoteAgentBinding")


AgentBinding = LocalAgentBinding | RemoteAgentBinding


# ── Tool Metadata ────────────────────────────────────────────────────


@dataclass
class ToolMetadata:
    """Metadata describing a tool's identity and capabilities."""

    name: str
    display_name: str = ""
    description: str = ""
    parameters: dict = field(default_factory=dict)
    metadata_id: str = ""
    display_name_locale: dict[str, str] | None = None
    description_locale: dict[str, str] | None = None
    binding: ToolBinding | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ToolMetadata.name must not be empty")
        if not self.metadata_id:
            self.metadata_id = self.name
        if not self.display_name:
            self.display_name = self.name

    def resolve_display_name(self, locale: str = "") -> str:
        if locale and self.display_name_locale and locale in self.display_name_locale:
            return self.display_name_locale[locale]
        return self.display_name or self.name

    def resolve_description(self, locale: str = "") -> str:
        if locale and self.description_locale and locale in self.description_locale:
            return self.description_locale[locale]
        return self.description


# ── Agent Metadata (extended with binding) ───────────────────────────


@dataclass
class AgentMetadata:
    """Metadata describing an agent's configuration and capabilities."""

    name: str
    display_name: str = ""
    description: str = ""
    system_prompt: str = ""
    system_prompt_locale: dict[str, str] | None = None
    agent_type: str = "simple"
    tool_names: list[str] = field(default_factory=list)
    metadata_id: str = ""
    display_name_locale: dict[str, str] | None = None
    description_locale: dict[str, str] | None = None
    binding: AgentBinding | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("AgentMetadata.name must not be empty")
        if not self.metadata_id:
            self.metadata_id = self.name
        if not self.display_name:
            self.display_name = self.name

    def resolve_display_name(self, locale: str = "") -> str:
        if locale and self.display_name_locale and locale in self.display_name_locale:
            return self.display_name_locale[locale]
        return self.display_name or self.name

    def resolve_description(self, locale: str = "") -> str:
        if locale and self.description_locale and locale in self.description_locale:
            return self.description_locale[locale]
        return self.description

    def resolve_system_prompt(self, locale: str = "") -> str:
        if locale and self.system_prompt_locale and locale in self.system_prompt_locale:
            return self.system_prompt_locale[locale]
        return self.system_prompt


class ToolCallFunction(TypedDict):
    """Provider-agnostic representation of a tool invocation."""

    name: str
    arguments: str


class ToolCall(TypedDict):
    """Provider-agnostic tool call produced by an LLM.

    Both OpenAI and Anthropic providers map their native tool-use
    representations into this unified shape.
    """

    id: str
    type: str
    function: ToolCallFunction


class TokenUsage(TypedDict):
    """Token consumption for a single LLM turn."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ToolResult:
    """Wraps a tool execution result, separating LLM-facing content from
    UI-only metadata that should not consume LLM context window.

    ``content``: Goes into the LLM conversation context (semantic payload).
    ``meta``:   Optional dict of UI/viz data; preserved in SSE events and
                 persisted messages, but never included in LLM context.

    Example::

        yield ToolResult(
            content="Today's weather in Shanghai is sunny, 25 C",
            meta={
                "chart_data": {"labels": [...], "values": [...]},
                "html": "<div class='weather-card'>...</div>",
            },
        )
    """

    content: Any
    meta: dict | None = None


ToolResultCallback = Callable[[ToolCall, Any], Awaitable[None]]
StreamingToolFunction = Callable[..., AsyncIterator[Any]]


@dataclass
class AgentStart:
    user_input: Iterable[ExtendedInputContentPart]
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentEnd:
    response: str
    time_taken: float | None = None
    exceeded: bool = False
    interrupted: bool = False
    error: str | None = None


@dataclass
class ToolCallDelta:
    """Partial update for a tool call within a streaming chunk."""

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str | None = None


@dataclass
class LLMChunkDelta:
    """Provider-agnostic representation of a single streaming chunk delta."""

    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[ToolCallDelta] | None = None


@dataclass
class LLMChunk:
    chunk: LLMChunkDelta | None


@dataclass
class LLMStart:
    messages: list["Message"]
    tools: Any


@dataclass
class LLMEnd:
    content: str | None
    reasoning_content: str | None
    tool_calls: list[ToolCall]
    usage: TokenUsage | None


@dataclass
class ExecutionStart:
    tool_calls: list[ToolCall]


@dataclass
class ExecutionEnd:
    results: list[tuple[ToolCall, Any]]


@dataclass
class ToolStart:
    tool_call: ToolCall


@dataclass
class ToolProgress:
    tool_call: ToolCall
    chunk: Any


@dataclass
class ToolEnd:
    tool_call: ToolCall
    result: Any


@dataclass
class MemoryUpdate:
    usage: TokenUsage


@dataclass
class MessageEvent:
    """Emitted by agents to communicate conversation messages to downstream services.

    Each instance carries a single ``Message`` dict (role, content, tool_calls, etc.)
    that was added to the agent's internal conversation memory. Downstream services
    (e.g. orchestration) collect these to persist session history without needing to
    reverse-engineer conversation structure from low-level ``LLMStart``/``LLMEnd`` events.
    """

    message: dict[str, Any]


ToolEvent = Union[ToolStart, ToolProgress, ToolEnd]


AgentEvent = Union[
    AgentStart,
    AgentEnd,
    ExecutionEnd,
    ExecutionStart,
    LLMChunk,
    LLMEnd,
    LLMStart,
    MemoryUpdate,
    MessageEvent,
    ToolEnd,
    ToolProgress,
    ToolStart,
]
