from dataclasses import dataclass
from typing import Any, Iterable

from minimal_harness.memory import ExtendedInputContentPart
from minimal_harness.types import TokenUsage, ToolCall


@dataclass
class LLMChunkEvent:
    """Streaming chunk from LLM."""

    chunk: Any | None
    is_done: bool


@dataclass
class LLMStartEvent:
    """When LLM starts processing."""

    pass


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


Event = (
    AgentStartEvent
    | AgentEndEvent
    | LLMChunkEvent
    | ExecutionStartEvent
    | LLMEndEvent
    | LLMStartEvent
    | ToolEndEvent
    | ToolProgressEvent
    | ToolStartEvent
)
