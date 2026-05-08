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
    TypedDict,
    TypeVar,
    Union,
)

if TYPE_CHECKING:
    from minimal_harness.memory import ExtendedInputContentPart, Message

T = TypeVar("T")

ChunkCallback = Callable[[T | None, bool], Awaitable[None]]


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

    def __post_init__(self) -> None:
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
    ToolEnd,
    ToolProgress,
    ToolStart,
]
