import asyncio
from typing import AsyncIterator, Protocol, Sequence, TypeVar

from minimal_harness.memory import Message
from minimal_harness.tool.base import StreamingTool
from minimal_harness.types import (
    ChunkCallback,
    LLMChunkDelta,
    TokenUsage,
    ToolCall,
    ToolCallFunction,
    ToolResultCallback,
)

T = TypeVar("T")

__all__ = [
    "ChunkCallback",
    "LLMProvider",
    "LLMResponse",
    "Stream",
    "TokenUsage",
    "ToolCall",
    "ToolCallFunction",
    "ToolResultCallback",
]


class LLMResponse:
    content: str | None
    reasoning_content: str | None
    tool_calls: list[ToolCall]
    finish_reason: str | None
    usage: TokenUsage | None

    def __init__(
        self,
        content: str | None,
        reasoning_content: str | None,
        tool_calls: list[ToolCall],
        finish_reason: str | None,
        usage: TokenUsage | None = None,
    ):
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls
        self.finish_reason = finish_reason
        self.usage = usage


class Stream[T]:
    def __init__(self, agen: AsyncIterator[T | LLMResponse]):
        self._agen = agen
        self._response: LLMResponse | None = None

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        chunk = await self._agen.__anext__()

        if isinstance(chunk, LLMResponse):
            self._response = chunk
            raise StopAsyncIteration

        return chunk

    @property
    def response(self) -> LLMResponse:
        if self._response is None:
            raise RuntimeError("Stream not exhausted yet")
        return self._response


class LLMProvider(Protocol):
    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[StreamingTool],
        stop_event: asyncio.Event | None = None,
    ) -> Stream[LLMChunkDelta]: ...
