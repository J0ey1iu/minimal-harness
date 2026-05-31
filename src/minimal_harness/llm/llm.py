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


class LLMProviderRegistry:
    """Registry for named LLM provider constructors.

    Users register custom providers via ``register(name, factory)`` without
    subclassing or config files.  Built-in ``"openai"`` and ``"anthropic"``
    are pre-registered.

    Each provider may have a *default config* dict set via
    ``set_default_config``.  Credentials (``api_key``, ``base_url``) are
    locked to the provider defaults — they can only be set at registration
    time and cannot be overridden per-agent.  Other parameters (``model``,
    ``temperature``, ``max_tokens``, etc.) from the per-call *cfg* are
    merged in and take precedence over defaults.
    """

    def __init__(self) -> None:
        self._registry: dict[str, Callable[[dict[str, Any]], LLMProvider]] = {}
        self._defaults: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        factory: Callable[[dict[str, Any]], LLMProvider],
    ) -> None:
        self._registry[name] = factory

    def set_default_config(self, name: str, cfg: dict[str, Any]) -> None:
        self._defaults[name] = cfg

    def get_default_config(self, name: str) -> dict[str, Any]:
        return self._defaults.get(name, {})

    def create(self, provider: str, cfg: dict[str, Any]) -> LLMProvider:
        factory = self._registry.get(provider)
        if factory is None:
            raise ValueError(
                f"LLM provider '{provider}' is not registered. "
                f"Available: {list(self._registry)}"
            )
        defaults = self._defaults.get(provider, {})
        merged = {**defaults, **cfg}
        for cred in ("api_key", "base_url"):
            if cred in defaults:
                merged[cred] = defaults[cred]
        return factory(merged)

    def list_providers(self) -> list[str]:
        return list(self._registry)

    def is_registered(self, name: str) -> bool:
        return name in self._registry

    def clone_factory(self, source: str, target: str) -> None:
        """Register *target* using the same factory as *source*."""
        factory = self._registry.get(source)
        if factory is None:
            raise ValueError(f"Source provider '{source}' is not registered")
        self._registry[target] = factory
