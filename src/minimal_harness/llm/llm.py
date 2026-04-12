from typing import Protocol, AsyncIterator, Any, Callable, Awaitable, TypeVar, TypedDict

from minimal_harness.memory import Message
from minimal_harness.tool import Tool


T = TypeVar("T")

ChunkCallback = Callable[[T | None, bool], Awaitable[None]]


class ToolCallFunction(TypedDict):
    name: str
    arguments: str


class ToolCall(TypedDict):
    id: str
    type: str
    function: ToolCallFunction


ToolResultCallback = Callable[[ToolCall, Any], Awaitable[None]]


class TokenUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall]
    finish_reason: str | None
    usage: TokenUsage | None

    def __init__(
        self,
        content: str | None,
        tool_calls: list[ToolCall],
        finish_reason: str | None,
        usage: TokenUsage | None = None,
    ):
        self.content = content
        self.tool_calls = tool_calls
        self.finish_reason = finish_reason
        self.usage = usage


class Stream[T]:
    def __init__(self, agen: AsyncIterator[T]):
        self._agen = agen
        self._response: LLMResponse | None = None

    def __aiter__(self) -> AsyncIterator:
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._agen.__anext__()

            if isinstance(chunk, LLMResponse):
                self._response = chunk
                raise StopAsyncIteration

            return chunk

        except StopAsyncIteration:
            raise

    @property
    def response(self) -> LLMResponse:
        if self._response is None:
            raise RuntimeError("Stream not exhausted yet")
        return self._response


class LLMProvider(Protocol):
    async def chat(
        self,
        messages: list[Message],
        tools: list[Tool],
        on_chunk: ChunkCallback | None,
    ) -> Stream: ...
