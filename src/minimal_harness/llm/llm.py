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
    "LLMProviderRegistry",
    "ProviderFactory",
    "LLMResponse",
    "STREAM_IDLE_TIMEOUT",
    "STREAM_STALL_RETRIES",
    "Stream",
    "StreamStalledError",
    "TokenUsage",
    "ToolCall",
    "ToolCallFunction",
    "anext_with_timeout",
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


# Chunk-level stall watchdog defaults.  A streamed LLM response that produces
# no chunk for this many seconds is considered stalled.  Byte-level read
# timeouts alone don't catch this: providers can keep the connection alive
# with SSE keep-alive bytes while emitting no content, which resets the
# socket read timeout indefinitely.
STREAM_IDLE_TIMEOUT = 20.0
STREAM_STALL_RETRIES = 2  # reconnect + re-stream attempts after the first stall


class StreamStalledError(Exception):
    """Raised when an LLM stream yields no chunk within the idle timeout."""

    def __init__(self, timeout: float) -> None:
        super().__init__(f"LLM stream stalled: no chunk for {timeout:g}s")


async def anext_with_timeout(agen: Any, timeout: float) -> Any:
    """Return the next item of *agen*, or raise :class:`StreamStalledError`.

    Shared by the OpenAI and Anthropic providers so the chunk-level stall
    detector lives in one place.  ``StopAsyncIteration`` propagates as-is.
    """
    try:
        return await asyncio.wait_for(agen.__anext__(), timeout=timeout)
    except asyncio.TimeoutError:
        raise StreamStalledError(timeout) from None


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


class ProviderFactory:
    """Registry of named LLM provider constructors.

    Users register custom provider factories via ``register(name, factory)``
    without subclassing or config files.  Built-in ``"openai"`` and
    ``"anthropic"`` are pre-registered.

    The ``create(name, cfg)`` method looks up the factory, passes *cfg*
    straight through, and returns an ``LLMProvider`` instance.
    """

    def __init__(self) -> None:
        self._registry: dict[str, Callable[[dict[str, Any]], LLMProvider]] = {}

    def register(
        self,
        name: str,
        factory: Callable[[dict[str, Any]], LLMProvider],
    ) -> None:
        self._registry[name] = factory

    def create(self, provider: str, cfg: dict[str, Any]) -> LLMProvider:
        factory = self._registry.get(provider)
        if factory is None:
            raise ValueError(
                f"LLM provider '{provider}' is not registered. "
                f"Available: {list(self._registry)}"
            )
        return factory(cfg)

    def list_providers(self) -> list[str]:
        return list(self._registry)

    def is_registered(self, name: str) -> bool:
        return name in self._registry


LLMProviderRegistry = ProviderFactory  # retained for backward compat
