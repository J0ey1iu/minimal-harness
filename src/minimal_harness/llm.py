from typing import Protocol, AsyncIterator, Any, Callable, Awaitable, TypedDict

from minimal_harness.memory import Message
from minimal_harness.tool import Tool


ChunkCallback = Callable[[Any, bool], Awaitable[None]]


class ToolCallFunction(TypedDict):
    name: str
    arguments: str


class ToolCall(TypedDict):
    id: str
    type: str
    function: ToolCallFunction


class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall]
    finish_reason: str | None


class Stream:
    def __init__(self, agen: AsyncIterator):
        self._agen = agen
        self._response: LLMResponse | None = None

    def __aiter__(self) -> AsyncIterator:
        return self

    async def __anext__(self) -> Any:
        try:
            return await self._agen.__anext__()
        except StopIteration as e:
            self._response = e.value
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
