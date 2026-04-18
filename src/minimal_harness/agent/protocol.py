import asyncio
from typing import Any, Awaitable, Callable, Iterable, Protocol

from minimal_harness.memory import ExtendedInputContentPart, InputContentPart
from minimal_harness.types import (
    ChunkCallback,
    ExecutionStartCallback,
    ProgressCallback,
    ToolEndCallback,
    ToolStartCallback,
    UserInputCallback,
)

InputContentConversionFunction = Callable[
    [Iterable[ExtendedInputContentPart]], Awaitable[Iterable[InputContentPart]]
]


class Agent(Protocol):
    async def run(
        self,
        user_input: Iterable[ExtendedInputContentPart],
        custom_input_conversion: InputContentConversionFunction | None = None,
        on_tool_start: ToolStartCallback | None = None,
        on_tool_end: ToolEndCallback | None = None,
        on_execution_start: ExecutionStartCallback | None = None,
        wait_for_user_input: UserInputCallback | None = None,
        on_tool_progress: ProgressCallback | None = None,
        on_chunk: ChunkCallback[Any] | None = None,
        stop_event: asyncio.Event | None = None,
    ) -> str: ...
