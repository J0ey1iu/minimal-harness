from __future__ import annotations

from dataclasses import dataclass
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

AgentStartCallback = Callable[
    [
        Iterable["ExtendedInputContentPart"],
    ],
    Awaitable[None],
]
AgentEndCallback = Callable[[str], Awaitable[None]]


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
ToolStartCallback = ToolResultCallback
ToolEndCallback = ToolResultCallback
ExecutionStartCallback = Callable[[list[ToolCall]], Awaitable[None]]
ToolFunction = Callable[..., Awaitable[Any]]
StreamingToolFunction = Callable[..., AsyncIterator[Any]]
UserInputCallback = Callable[[str], Awaitable[Any]]
ProgressCallback = Callable[[ToolCall, Any], Awaitable[None]]


@dataclass
class AgentStart:
    user_input: Iterable[ExtendedInputContentPart]


@dataclass
class AgentEnd:
    response: str


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
    is_done: bool


@dataclass
class LLMStart:
    messages: list["Message"]
    tools: Any


@dataclass
class LLMEnd:
    content: str | None
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
