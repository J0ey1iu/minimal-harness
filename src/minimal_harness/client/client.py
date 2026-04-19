"""Framework client that emits events to a queue for decoupled consumption."""

import asyncio
from typing import AsyncIterator, Iterable

from minimal_harness.agent import OpenAIAgent
from minimal_harness.agent.protocol import InputContentConversionFunction
from minimal_harness.memory import ExtendedInputContentPart
from minimal_harness.types import (
    AgentEnd,
    AgentEvent,
    AgentStart,
    ExecutionEnd,
    ExecutionStart,
    LLMChunk,
    LLMEnd,
    LLMStart,
    ToolEnd,
    ToolProgress,
    ToolStart,
)

from .events import (
    AgentEndEvent,
    AgentStartEvent,
    Event,
    ExecutionEndEvent,
    ExecutionStartEvent,
    LLMChunkEvent,
    LLMEndEvent,
    LLMStartEvent,
    ToolEndEvent,
    ToolProgressEvent,
    ToolStartEvent,
)


def _agent_event_to_client_event(event: AgentEvent) -> Event:
    if isinstance(event, AgentStart):
        return AgentStartEvent(event.user_input)
    elif isinstance(event, AgentEnd):
        return AgentEndEvent(event.response)
    elif isinstance(event, ExecutionEnd):
        return ExecutionEndEvent(event.results)
    elif isinstance(event, ExecutionStart):
        return ExecutionStartEvent(event.tool_calls)
    elif isinstance(event, LLMChunk):
        return LLMChunkEvent(event.chunk, event.is_done)
    elif isinstance(event, LLMStart):
        return LLMStartEvent()
    elif isinstance(event, LLMEnd):
        return LLMEndEvent(event.content, event.tool_calls, event.usage)
    elif isinstance(event, ToolStart):
        return ToolStartEvent(event.tool_call, None)
    elif isinstance(event, ToolProgress):
        return ToolProgressEvent(event.tool_call, event.chunk)
    elif isinstance(event, ToolEnd):
        return ToolEndEvent(event.tool_call, event.result)
    else:
        msg = f"Unknown agent event type: {type(event)}"
        raise ValueError(msg)


class FrameworkClient:
    def __init__(self, agent: OpenAIAgent) -> None:
        self._agent = agent
        self._stop_event: asyncio.Event | None = None

    async def run(
        self,
        user_input: Iterable[ExtendedInputContentPart],
        custom_input_conversion: InputContentConversionFunction | None = None,
        stop_event: asyncio.Event | None = None,
    ) -> AsyncIterator[Event]:
        self._stop_event = stop_event
        try:
            async for event in self._agent.run(
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
