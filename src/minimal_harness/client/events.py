from dataclasses import dataclass
from typing import Any, Iterable

from minimal_harness.memory import ExtendedInputContentPart
from minimal_harness.types import (
    AgentEnd,
    AgentStart,
    ExecutionEnd,
    ExecutionStart,
    LLMChunk,
    LLMChunkDelta,
    LLMEnd,
    LLMStart,
    MemoryUpdate,
    TokenUsage,
    ToolCall,
    ToolEnd,
    ToolProgress,
    ToolStart,
)


def to_client_event(
    event: AgentEnd
    | AgentStart
    | ExecutionEnd
    | ExecutionStart
    | LLMChunk
    | LLMEnd
    | LLMStart
    | MemoryUpdate
    | ToolEnd
    | ToolProgress
    | ToolStart,
) -> "Event":
    if isinstance(event, AgentStart):
        return AgentStartEvent(user_input=event.user_input)
    if isinstance(event, AgentEnd):
        return AgentEndEvent(response=event.response)
    if isinstance(event, LLMChunk):
        return LLMChunkEvent(chunk=event.chunk, is_done=event.is_done)
    if isinstance(event, LLMStart):
        return LLMStartEvent(messages=event.messages, tools=event.tools)
    if isinstance(event, LLMEnd):
        return LLMEndEvent(
            content=event.content, tool_calls=event.tool_calls, usage=event.usage
        )
    if isinstance(event, ExecutionStart):
        return ExecutionStartEvent(tool_calls=event.tool_calls)
    if isinstance(event, ExecutionEnd):
        return ExecutionEndEvent(results=event.results)
    if isinstance(event, ToolStart):
        return ToolStartEvent(event.tool_call, None)
    if isinstance(event, ToolProgress):
        return ToolProgressEvent(tool_call=event.tool_call, chunk=event.chunk)
    if isinstance(event, ToolEnd):
        return ToolEndEvent(tool_call=event.tool_call, result=event.result)
    if isinstance(event, MemoryUpdate):
        return MemoryUpdateEvent(usage=event.usage)
    raise TypeError(f"Unknown event type: {type(event).__name__}")


@dataclass
class LLMChunkEvent:
    """Streaming chunk from LLM."""

    chunk: LLMChunkDelta | None
    is_done: bool


@dataclass
class LLMStartEvent:
    """When LLM starts processing."""

    messages: Any
    tools: Any


@dataclass
class LLMEndEvent:
    """When LLM finishes processing."""

    content: str | None
    tool_calls: list[ToolCall]
    usage: TokenUsage | None


@dataclass
class AgentStartEvent:
    """When agent starts running."""

    user_input: Iterable[ExtendedInputContentPart]


@dataclass
class AgentEndEvent:
    """When agent finishes running."""

    response: str


@dataclass
class ExecutionStartEvent:
    """Before tool execution begins."""

    tool_calls: list[ToolCall]


@dataclass
class ExecutionEndEvent:
    """After tool execution ends."""

    results: list[tuple[ToolCall, Any]]


@dataclass
class ToolStartEvent:
    """When a tool starts executing."""

    tool_call: ToolCall
    _: Any  # deprecated, kept for compatibility


@dataclass
class ToolProgressEvent:
    """Progress update during streaming tool execution."""

    tool_call: ToolCall
    chunk: Any


@dataclass
class ToolEndEvent:
    """When a tool finishes executing."""

    tool_call: ToolCall
    result: Any


@dataclass
class MemoryUpdateEvent:
    usage: TokenUsage


Event = (
    AgentStartEvent
    | AgentEndEvent
    | ExecutionEndEvent
    | ExecutionStartEvent
    | LLMChunkEvent
    | LLMEndEvent
    | LLMStartEvent
    | MemoryUpdateEvent
    | ToolEndEvent
    | ToolProgressEvent
    | ToolStartEvent
)
