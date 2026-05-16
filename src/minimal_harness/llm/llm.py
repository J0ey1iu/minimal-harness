import asyncio
from typing import Any, AsyncIterator, Callable, Protocol, Sequence, TypeVar

from minimal_harness.memory import Message
from minimal_harness.tool.base import Tool
from minimal_harness.types import (
    ChunkCallback,
    LLMChunkDelta,
    TokenUsage,
    ToolCall,
    ToolCallFunction,
)

T = TypeVar("T")

LLMProviderFactory = Callable[[], "LLMProvider"]

__all__ = [
    "ChunkCallback",
    "LLMProvider",
    "LLMProviderFactory",
    "LLMResponse",
    "Stream",
    "TokenUsage",
    "ToolCall",
    "ToolCallFunction",
]


async def await_with_interrupt(
    coro,
    stop_event: asyncio.Event | None,
    poll_interval: float = 0.2,
):
    """Await *coro* while polling *stop_event* every *poll_interval* seconds.

    If *stop_event* is set before *coro* completes, the underlying task is
    cancelled and ``asyncio.CancelledError`` is raised.
    """
    if stop_event is None:
        return await coro
    task = asyncio.ensure_future(coro)
    while not stop_event.is_set():
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=poll_interval)
        except asyncio.TimeoutError:
            continue
    task.cancel()
    raise asyncio.CancelledError()


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
        tools: Sequence[Tool],
        stop_event: asyncio.Event | None = None,
        **kwargs: Any,
    ) -> Stream[LLMChunkDelta]: ...
