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
    Protocol,
    TypedDict,
    TypeVar,
    Union,
    runtime_checkable,
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


@runtime_checkable
class ContextProvider(Protocol):
    """Resolve structured request context at outbound call time.

    The returned dict is merged into the tool request body as the
    ``context`` field. Kept separate from :data:`ExtraHeadersProvider`:
    headers carry credentials (``Authorization``, cookies), context
    carries structured identity / trace / locale data.
    """

    async def __call__(self) -> dict[str, Any]: ...


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
    context_provider: ContextProvider | None = None
    verify_ssl: bool = False

    def __post_init__(self) -> None:
        if not self.url:
            raise ValueError("url must not be empty for RemoteToolBinding")


@dataclass
class MCPToolBinding:
    """Binding for a Model Context Protocol (MCP) stdio server.

    The server runs as a subprocess speaking JSON-RPC 2.0 over stdio
    (``initialize`` handshake, ``tools/list``, ``tools/call``). The
    subprocess lifecycle is owned by :class:`MCPManager`
    (:mod:`minimal_harness.tool.mcp`) — one process per ``server_slug``,
    reused across calls. Resolution goes through the ``"mcp"`` executor
    driver on :class:`~minimal_harness.tool.factory.DefaultToolFactory`.

    RFC #57 (mhc-desktop) added this binding; it is additive and does
    not change how existing local/script/remote bindings resolve.
    """

    type: Literal["mcp"] = "mcp"
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    server_slug: str = ""
    client_name: str = ""
    client_version: str = ""

    def __post_init__(self) -> None:
        if not self.command:
            raise ValueError("command must not be empty for MCPToolBinding")


ToolBinding = (
    LocalToolBinding | ExternalScriptToolBinding | RemoteToolBinding | MCPToolBinding
)


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
    max_tool_rounds: int = 2000

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
    # Canonical id of the last assistant message added during this run
    # (``msg-{seq}``, stamped by ``Memory.add_message``). Lets streaming
    # consumers commit the buffered assistant turn with the same id the
    # session reload will return, without a round-trip.
    message_id: str | None = None


# ── Controller events ─────────────────────────────────────────────
# Controller 层独立于 Agent 事件层：Controller 包裹 Agent 做多轮编排，
# 自有协议（见 agent/controller.py）与事件。三个事件覆盖完整生命周期，
# 具体 Controller 类型由 ``controller_type`` 字段区分，不新增事件类。


@dataclass
class ControllerStart:
    controller_type: str  # "default" / "goal" / "timer" / …
    user_input: Any
    timestamp: float = field(default_factory=time.time)


@dataclass
class ControllerContinue:
    controller_type: str
    next_prompt: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ControllerEnd:
    controller_type: str
    response: str
    time_taken: float | None = None
    exceeded: bool = False
    interrupted: bool = False
    error: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolCallDelta:
    """Partial update for a tool call within a streaming chunk."""

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str | None = None


@dataclass
class ToolArgsStart:
    """A tool call's arguments started streaming (delta-event granularity).

    RFC #60: emitted under ``emit_delta_events=True`` from the streamed
    ``tool_calls`` fragments so SSE consumers can render a **pending
    capsule** while the model is still generating the arguments.
    ``call_id`` follows the §3 contract: it is the stable id reused
    across ``ToolArgsStart`` / ``ToolArgsDelta`` and — when the
    provider's fragments carry an id — equals the final ``tc["id"]``
    seen at execution time (positionally aligned).

    ``name`` may arrive a fragment later, so it can be empty here and
    filled by a later ``ToolArgsStart``/chunk; ``kind`` mirrors the call
    type (``"function"`` | ``"mcp"`` | ...).
    """

    call_id: str
    name: str = ""
    kind: str = "function"
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolArgsDelta:
    """One incremental fragment of a streamed tool-call's arguments.

    Consumers concatenate ``arguments_chunk`` across deltas to rebuild
    the full JSON arguments string (RFC #60 §2).
    """

    call_id: str
    arguments_chunk: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolRoundStart:
    """A tool round is about to execute (delta-event granularity).

    RFC #57 delta events: part of the optional ``emit_delta_events``
    event stream (off by default — the base turn-level events are
    unchanged when disabled). ``kinds`` mirrors each call's ``type``
    field (``"function"``) so SSE consumers can render per-call
    capsules without re-deriving them.
    """

    ids: list[str]
    names: list[str]
    kinds: list[str]
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolRoundEnd:
    """A tool round finished (one of ``ok`` / ``cancelled`` / error)."""

    ok: bool
    cancelled: bool = False
    count: int = 0
    timestamp: float = field(default_factory=time.time)


class ToolResultTrimmer:
    """Deterministic head/tail byte trimming of tool results entering Memory.

    RFC #60 §5: a 200 KB ``cmd`` run pins ~50 K tokens of context until
    the next fold. Trimming is cheap, deterministic and needs no LLM call;
    it composes with the ``tool_compacting`` agent's summarization (the
    expensive, lossy second line).

    ``max_bytes == 0`` keeps today's behaviour byte-for-byte (default). When
    enabled, a result longer than ``max_bytes`` is reduced to
    ``head + recovery_hint + tail``; the hint tells the model how to
    re-fetch the middle. ``per_tool`` overlays per tool name (e.g. ``cmd``
    capped at 32 KB, ``read_file`` exempt) — the ``max_bytes`` of the
    overlay replaces the global one, other fields fall back to the global.

    Only the copy entering ``Memory`` is trimmed; consumers still receive
    the full result via ``ToolEnd``/``ToolProgress``.
    """

    __slots__ = ("max_bytes", "head_ratio", "tail_ratio", "recovery_hint", "per_tool")

    def __init__(
        self,
        max_bytes: int = 0,
        head_ratio: float = 0.5,
        tail_ratio: float = 0.5,
        recovery_hint: str | None = None,
        per_tool: dict[str, "ToolResultTrimmer"] | None = None,
    ) -> None:
        self.max_bytes = max_bytes
        self.head_ratio = head_ratio
        self.tail_ratio = tail_ratio
        self.recovery_hint = recovery_hint
        self.per_tool = per_tool or {}

    def effective_for(self, tool_name: str) -> "ToolResultTrimmer":
        """Return the trimmer to apply for *tool_name* (overlay merged)."""
        overlay = self.per_tool.get(tool_name)
        if overlay is None:
            return self
        return ToolResultTrimmer(
            max_bytes=overlay.max_bytes or self.max_bytes,
            head_ratio=overlay.head_ratio,
            tail_ratio=overlay.tail_ratio,
            recovery_hint=overlay.recovery_hint or self.recovery_hint,
        )

    def trim(self, content: str) -> str:
        """Trim *content* deterministically; returns it unchanged if short enough.

        Byte-based (not char-based) head/tail cut; a cut landing mid-UTF-8
        sequence drops that single grapheme rather than corrupting the
        output (``errors="ignore"``).
        """
        if self.max_bytes <= 0:
            return content
        data = content.encode("utf-8", "ignore")
        if len(data) <= self.max_bytes:
            return content
        head_n = max(0, int(self.max_bytes * self.head_ratio))
        tail_n = max(0, self.max_bytes - head_n)
        head = data[:head_n].decode("utf-8", "ignore")
        tail = data[-tail_n:].decode("utf-8", "ignore") if tail_n else ""
        lost = len(data) - head_n - tail_n
        hint = self.recovery_hint or (
            f"... [output trimmed: middle {lost} bytes removed] ..."
        )
        return f"{head}{hint}{tail}"


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
    # Canonical id of the assistant message this LLM turn just produced
    # (``msg-{seq}``, stamped by ``Memory.add_message``). Lets streaming
    # consumers locate the turn's message before AgentEnd arrives.
    message_id: str | None = None


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

    Each instance carries a single ``Message`` dict (role, content, tool_calls, …)
    that was added to the agent's internal conversation memory. Downstream services
    (e.g. gateway) collect these to persist session history without needing to
    reverse-engineer conversation structure from low-level ``LLMStart``/``LLMEnd``
    events.

    ``message`` carries the canonical ``id`` (``msg-{seq}``) stamped by
    ``Memory.add_message`` when the message entered the session — the same
    value read-side adapters return after a reload.

    Ordering contract: for a given LLM turn the ``MessageEvent``(s) for the
    turn's messages are emitted *before* the turn's ``LLMEnd`` (the agent
    persists the messages first, then broadcasts ``LLMEnd`` with the
    assistant message's ``message_id``). Consumers must not assume
    ``LLMEnd`` precedes the turn's ``MessageEvent``.
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
    # Canonical id of the summary message (``role="compaction"``) stamped by
    # ``Memory.compact()`` — the id the summary row will have after reload.
    message_id: str | None = None


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
    soft_limit_ratio: float = 0.0
    max_context_tokens: int = 0
    estimate_leading_edge: bool = True
    anchor_keep_recent_on: Literal["last_tool_round", "last_user", "tail"] = "tail"


class CompactionSettings(TypedDict, total=False):
    """JSON-serialisable compaction configuration on ``AgentMetadata``.

    This is the serialisable counterpart of :class:`CompactionConfig`:
    it carries the threshold and ``keep_recent`` knobs that come from
    ``agents.json``, but **not** the runtime ``summarizer`` (which is
    a streaming async generator and cannot be serialised). Consumers
    that build a full :class:`CompactionConfig` read the
    ``CompactionSettings`` from metadata, then inject their own
    summarizer at factory time.

    ``compaction_prompt`` is an optional user-customisable summarization
    instruction that replaces the built-in ``DEFAULT_SUMMARY_REQUEST``.
    When not set (or empty), the built-in default is used.

    ``compaction_prompt_locale`` is a JSON dict mapping locale codes
    to locale-specific versions of the compaction prompt, e.g.
    ``{"zh": "请用中文总结", "en": "Summarize in English"}``.
    It follows the same i18n pattern as ``system_prompt_locale``.

    All keys are optional — see
    :class:`CompactionConfig` for defaults.
    """

    prompt_token_threshold: int
    keep_recent: int
    compaction_prompt: str
    compaction_prompt_locale: str
    soft_limit_ratio: float
    max_context_tokens: int
    estimate_leading_edge: bool
    anchor_keep_recent_on: str


class ToolCompactionSettings(TypedDict, total=False):
    """JSON-serialisable tool compaction configuration on ``AgentMetadata``.

    Serialisable counterpart of :class:`ToolCompactionConfig`:
    carries *prompt_token_threshold* and *keep_recent* knobs from
    the agent definition, but **not** the runtime ``summarizer``.
    Consumers build a full :class:`ToolCompactionConfig` at factory
    time.

    ``compaction_prompt`` is an optional user-customisable summarization
    instruction that replaces the built-in ``DEFAULT_SUMMARY_REQUEST``.
    When not set (or empty), the built-in default is used.

    ``compaction_prompt_locale`` is a JSON dict mapping locale codes
    to locale-specific versions of the compaction prompt, e.g.
    ``{"zh": "请用中文总结", "en": "Summarize in English"}``.
    It follows the same i18n pattern as ``system_prompt_locale``.

    All keys are optional — see
    :class:`ToolCompactionConfig` for defaults.
    """

    prompt_token_threshold: int
    keep_recent: int
    compaction_prompt: str
    compaction_prompt_locale: str


@dataclass
class ToolCompactionConfig:
    """Runtime-injected configuration for ``agent_type="tool_compacting"`` agents.

    *summarizer* is a streaming async generator that yields summary
    text chunks. *prompt_token_threshold* and *keep_recent* control
    full conversation compaction (same behaviour as
    :class:`CompactionConfig`).

    The agent always discards ``role="tool"`` messages from the
    forward buffer -- no configuration needed for that behaviour.

    RFC #57 additions mirror :class:`CompactionConfig`
    (``soft_limit_ratio`` / ``max_context_tokens`` /
    ``estimate_leading_edge`` / ``anchor_keep_recent_on``), all defaulting
    to the pre-existing behaviour.
    """

    summarizer: "CompactionSummarizer"
    prompt_token_threshold: int = 0
    keep_recent: int = 6
    soft_limit_ratio: float = 0.0
    max_context_tokens: int = 0
    estimate_leading_edge: bool = True
    anchor_keep_recent_on: Literal["last_tool_round", "last_user", "tail"] = "tail"


CompactionEvent = Union[CompactionStart, CompactionChunk, CompactionEnd]

ToolEvent = Union[ToolStart, ToolProgress, ToolEnd]


ControllerEvent = Union[ControllerStart, ControllerContinue, ControllerEnd]

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
    ToolArgsDelta,
    ToolArgsStart,
    ToolRoundEnd,
    ToolRoundStart,
    ToolStart,
]
