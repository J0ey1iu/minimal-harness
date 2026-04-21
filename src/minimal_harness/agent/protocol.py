import asyncio
from typing import AsyncIterator, Awaitable, Callable, Iterable, Protocol, Sequence

from minimal_harness.memory import ExtendedInputContentPart, InputContentPart, Memory
from minimal_harness.tool.base import StreamingTool
from minimal_harness.types import AgentEvent

InputContentConversionFunction = Callable[
    [Iterable[ExtendedInputContentPart]], Awaitable[Iterable[InputContentPart]]
]


class Agent(Protocol):
    async def run(
        self,
        user_input: Iterable[ExtendedInputContentPart],
        stop_event: asyncio.Event | None = None,
        memory: Memory | None = None,
        tools: Sequence[StreamingTool] | None = None,
    ) -> AsyncIterator[AgentEvent]: ...
