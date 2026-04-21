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
    from minimal_harness.memory import ExtendedInputContentPart

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
    name: str
    arguments: str


class ToolCall(TypedDict):
    id: str
    type: str
    function: ToolCallFunction


class TokenUsage(TypedDict):
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
class LLMChunk:
    chunk: Any | None
    is_done: bool


@dataclass
class LLMStart:
    messages: Any
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


ToolEvent = Union[ToolStart, ToolProgress, ToolEnd]


AgentEvent = Union[
    AgentStart,
    AgentEnd,
    ExecutionEnd,
    ExecutionStart,
    LLMChunk,
    LLMEnd,
    LLMStart,
    ToolEnd,
    ToolProgress,
    ToolStart,
]
