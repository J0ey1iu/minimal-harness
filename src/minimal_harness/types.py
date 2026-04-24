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
    from minimal_harness.client.events import Event
    from minimal_harness.memory import ExtendedInputContentPart, Message

T = TypeVar("T")


def _client_events():
    """Lazy import of client.events to avoid circular imports."""
    import minimal_harness.client.events as _ce
    return _ce

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

    def to_client_event(self) -> "Event":
        return _client_events().AgentStartEvent(user_input=self.user_input)


@dataclass
class AgentEnd:
    response: str

    def to_client_event(self) -> "Event":
        return _client_events().AgentEndEvent(response=self.response)


@dataclass
class LLMChunk:
    chunk: Any | None
    is_done: bool

    def to_client_event(self) -> "Event":
        return _client_events().LLMChunkEvent(chunk=self.chunk, is_done=self.is_done)


@dataclass
class LLMStart:
    messages: list["Message"]
    tools: Any

    def to_client_event(self) -> "Event":
        return _client_events().LLMStartEvent(messages=self.messages, tools=self.tools)


@dataclass
class LLMEnd:
    content: str | None
    tool_calls: list[ToolCall]
    usage: TokenUsage | None

    def to_client_event(self) -> "Event":
        return _client_events().LLMEndEvent(
            content=self.content,
            tool_calls=self.tool_calls,
            usage=self.usage,
        )


@dataclass
class ExecutionStart:
    tool_calls: list[ToolCall]

    def to_client_event(self) -> "Event":
        return _client_events().ExecutionStartEvent(tool_calls=self.tool_calls)


@dataclass
class ExecutionEnd:
    results: list[tuple[ToolCall, Any]]

    def to_client_event(self) -> "Event":
        return _client_events().ExecutionEndEvent(results=self.results)


@dataclass
class ToolStart:
    tool_call: ToolCall

    def to_client_event(self) -> "Event":
        return _client_events().ToolStartEvent(self.tool_call, None)


@dataclass
class ToolProgress:
    tool_call: ToolCall
    chunk: Any

    def to_client_event(self) -> "Event":
        return _client_events().ToolProgressEvent(tool_call=self.tool_call, chunk=self.chunk)


@dataclass
class ToolEnd:
    tool_call: ToolCall
    result: Any

    def to_client_event(self) -> "Event":
        return _client_events().ToolEndEvent(tool_call=self.tool_call, result=self.result)


@dataclass
class MemoryUpdate:
    usage: TokenUsage

    def to_client_event(self) -> "Event":
        return _client_events().MemoryUpdateEvent(usage=self.usage)


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
