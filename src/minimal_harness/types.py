from typing import Any, AsyncIterator, Awaitable, Callable, TypedDict, TypeVar

T = TypeVar("T")

ChunkCallback = Callable[[T | None, bool], Awaitable[None]]


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
