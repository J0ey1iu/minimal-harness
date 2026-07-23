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

# Compaction summarizer: takes the messages to fold plus the existing
# summary (None on the first compaction), and yields the new summary as
# streaming text chunks. ``CompactionAgent`` collects the chunks into a
# single string and applies it to memory.
CompactionSummarizer = Callable[["list[Message]", "str | None"], AsyncIterator[str]]

# Callable that returns auth headers lazily at request time.
# Used by RemoteToolBinding so that auth credentials
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
    verify_ssl: bool = False

    def __post_init__(self) -> None:
        if not self.url:
            raise ValueError("url must not be empty for RemoteToolBinding")


ToolBinding = LocalToolBinding | ExternalScriptToolBinding | RemoteToolBinding


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
    provider: str = "openai"
    model: str = ""
    llm_config: dict[str, Any] = field(default_factory=dict)
    compaction: CompactionSettings | None = None
    tool_compaction: ToolCompactionSettings | None = None

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
    stop: bool = False


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
    error: str | None = None


@dataclass
class ExecutionStart:
    tool_calls: list[ToolCall]


@dataclass
class ExecutionEnd:
    results: list[tuple[ToolCall, Any]]
    error: str | None = None
    should_stop: bool = False
    response_text: str | None = None


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
    (e.g. gateway) collect these to persist session history without needing to
    reverse-engineer conversation structure from low-level ``LLMStart``/``LLMEnd`` events.
    """

    message: dict[str, Any]


@dataclass
class CompactionStart:
    """Emitted right before ``Memory.compact()`` starts streaming the summary.

    Carries the input slice to the summarizer (count only) plus the previous
    summary (if any) so observers can render a status panel without buffering
    the dropped messages themselves.
    """

    dropped_message_count: int
    existing_summary: str | None
    keep_recent: int
    total_tokens: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class CompactionChunk:
    """A single streaming delta from the compaction summarizer.

    ``delta`` is the new fragment just produced; ``accumulated`` is the full
    summary text so far (a convenience field — clients can also accumulate
    on their own).
    """

    delta: str
    accumulated: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class CompactionEnd:
    """Emitted after ``Memory.compact()`` finishes (success or failure).

    On failure, ``error`` is set and ``dropped_message_count`` is 0 — the
    memory buffer is left in its pre-compaction state.
    """

    summary: str
    dropped_message_count: int
    new_offset: int
    duration: float
    error: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class CompactionConfig:
    """Runtime-injected configuration for ``agent_type="compacting"`` agents.

    The user-supplied ``summarizer`` is a streaming async generator that
    yields the new summary text chunk by chunk. ``prompt_token_threshold``
    is checked against ``LLMEnd.usage["prompt_tokens"]`` after every LLM
    call — when exceeded, ``Memory.compact()`` runs before the next
    iteration. ``keep_recent`` controls how many tail messages are kept
    verbatim.
    """

    summarizer: "CompactionSummarizer"
    prompt_token_threshold: int
    keep_recent: int = 6


class CompactionSettings(TypedDict, total=False):
    """JSON-serialisable compaction configuration on ``AgentMetadata``.

    This is the serialisable counterpart of :class:`CompactionConfig`:
    it carries the threshold and ``keep_recent`` knobs that come from
    ``agents.json``, but **not** the runtime ``summarizer`` (which is
    a streaming async generator and cannot be serialised). Consumers
    that build a full :class:`CompactionConfig` read the
    ``CompactionSettings`` from metadata, then inject their own
    summarizer at factory time.

    All keys are optional — see
    :class:`CompactionConfig` for defaults.
    """

    prompt_token_threshold: int
    keep_recent: int


class ToolCompactionSettings(TypedDict, total=False):
    """JSON-serialisable tool compaction configuration on ``AgentMetadata``.

    Serialisable counterpart of :class:`ToolCompactionConfig`:
    carries *prompt_token_threshold* and *keep_recent* knobs from
    the agent definition, but **not** the runtime ``summarizer``.
    Consumers build a full :class:`ToolCompactionConfig` at factory
    time.

    All keys are optional — see
    :class:`ToolCompactionConfig` for defaults.
    """

    prompt_token_threshold: int
    keep_recent: int


@dataclass
class ToolCompactionConfig:
    """Runtime-injected configuration for ``agent_type="tool_compacting"`` agents.

    *summarizer* is a streaming async generator that yields summary
    text chunks. *prompt_token_threshold* and *keep_recent* control
    full conversation compaction (same behaviour as
    :class:`CompactionConfig`).

    The agent always discards ``role="tool"`` messages from the
    forward buffer -- no configuration needed for that behaviour.
    """

    summarizer: "CompactionSummarizer"
    prompt_token_threshold: int = 0
    keep_recent: int = 6


CompactionEvent = Union[CompactionStart, CompactionChunk, CompactionEnd]

ToolEvent = Union[ToolStart, ToolProgress, ToolEnd]


AgentEvent = Union[
    AgentStart,
    AgentEnd,
    CompactionChunk,
    CompactionEnd,
    CompactionStart,
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
