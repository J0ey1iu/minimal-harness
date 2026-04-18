"""Framework client that emits events to a queue for decoupled consumption."""

import asyncio
from typing import AsyncIterator, Iterable, Sequence

from minimal_harness.agent import OpenAIAgent
from minimal_harness.agent.protocol import InputContentConversionFunction
from minimal_harness.llm.openai import OpenAILLMProvider
from minimal_harness.memory import ExtendedInputContentPart, Memory
from minimal_harness.tool.base import StreamingTool
from minimal_harness.types import (
    AgentEvent,
    AgentStart,
    Chunk,
    Done,
    ExecutionStart,
    Stopped,
    ToolEnd,
    ToolStart,
)

from .events import (
    AgentStartEvent,
    ChunkEvent,
    DoneEvent,
    Event,
    ExecutionStartEvent,
    StoppedEvent,
    ToolEndEvent,
    ToolStartEvent,
)


def _agent_event_to_client_event(event: AgentEvent) -> Event:
    if isinstance(event, AgentStart):
        return AgentStartEvent(event.user_input)
    elif isinstance(event, Chunk):
        return ChunkEvent(event.chunk, event.is_done)
    elif isinstance(event, ExecutionStart):
        return ExecutionStartEvent(event.tool_calls)
    elif isinstance(event, ToolStart):
        return ToolStartEvent(event.tool_call, None)
    elif isinstance(event, ToolEnd):
        return ToolEndEvent(event.tool_call, event.result)
    elif isinstance(event, Done):
        return DoneEvent(event.response)
    elif isinstance(event, Stopped):
        return StoppedEvent(event.response)
    else:
        msg = f"Unknown agent event type: {type(event)}"
        raise ValueError(msg)


class FrameworkClient:
    def __init__(
        self,
        llm_provider: OpenAILLMProvider,
        tools: Sequence[StreamingTool] | None = None,
        memory: Memory | None = None,
        max_iterations: int = 50,
    ) -> None:
        self._llm_provider = llm_provider
        self._tools = tools
        self._memory = memory
        self._max_iterations = max_iterations
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._stop_event: asyncio.Event | None = None
        self._agent_task: asyncio.Task[None] | None = None

    async def run(
        self,
        user_input: Iterable[ExtendedInputContentPart],
        custom_input_conversion: InputContentConversionFunction | None = None,
        stop_event: asyncio.Event | None = None,
    ) -> AsyncIterator[Event]:
        self._stop_event = stop_event
        agent = OpenAIAgent(
            llm_provider=self._llm_provider,
            tools=self._tools,
            max_iterations=self._max_iterations,
            memory=self._memory,
        )
        try:
            async for event in agent.run(
                user_input=user_input,
                custom_input_conversion=custom_input_conversion,
                stop_event=self._stop_event,
            ):
                yield _agent_event_to_client_event(event)
        except asyncio.CancelledError:
            pass

    def stop(self) -> None:
        if self._stop_event:
            self._stop_event.set()
