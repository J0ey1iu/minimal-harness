from typing import Awaitable, Callable, Iterable, Protocol

from minimal_harness.llm import ChunkCallback
from minimal_harness.memory import ExtendedInputContentPart, InputContentPart
from minimal_harness.tool_executor import ToolEndCallback, ToolStartCallback

InputContentConversionFunction = Callable[
    [Iterable[ExtendedInputContentPart]], Awaitable[Iterable[InputContentPart]]
]


class Agent(Protocol):
    async def run(
        self,
        user_input: Iterable[InputContentPart],
        on_chunk: ChunkCallback | None = None,
        on_tool_start: ToolStartCallback | None = None,
        on_tool_end: ToolEndCallback | None = None,
    ) -> str: ...
