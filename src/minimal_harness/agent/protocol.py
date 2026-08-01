import asyncio
from typing import Any, AsyncIterator, Awaitable, Callable, Iterable, Protocol, Sequence

from minimal_harness.memory import ExtendedInputContentPart, InputContentPart, Memory
from minimal_harness.tool.base import Tool
from minimal_harness.types import AgentEvent

InputContentConversionFunction = Callable[
    [Iterable[ExtendedInputContentPart]], Awaitable[Iterable[InputContentPart]]
]


class Agent(Protocol):
    def run(
        self,
        user_input: Iterable[ExtendedInputContentPart],
        stop_event: asyncio.Event | None = None,
        memory: Memory | None = None,
        tools: Sequence[Tool] | None = None,
        system_prompt: str = "",
        context: dict | None = None,
        llm_kwargs: dict | None = None,
        user_message_meta: dict | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AgentEvent]: ...
