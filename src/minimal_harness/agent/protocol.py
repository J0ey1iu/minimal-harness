from typing import Awaitable, Callable, Iterable, Protocol

from minimal_harness.llm import ChunkCallback, ToolResultCallback
from minimal_harness.memory import ExtendedInputContentPart, InputContentPart

InputContentConversionFunction = Callable[
    [Iterable[ExtendedInputContentPart]], Awaitable[Iterable[InputContentPart]]
]


class Agent(Protocol):
    async def run(
        self,
        user_input: Iterable[InputContentPart],
        on_chunk: ChunkCallback | None = None,
        on_tool_result: ToolResultCallback | None = None,
    ) -> str: ...
