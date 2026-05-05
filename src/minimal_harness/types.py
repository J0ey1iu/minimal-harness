from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    TypedDict,
    TypeVar,
    Union,
)

if TYPE_CHECKING:
    from minimal_harness.memory import ExtendedInputContentPart, Message

T = TypeVar("T")

ChunkCallback = Callable[[T | None, bool], Awaitable[None]]


@dataclass
class AgentMetadata:
    """Metadata describing an agent's configuration and capabilities."""

    name: str
    description: str = ""
    system_prompt: str = ""
    agent_type: str = "simple"
    tool_names: list[str] = field(default_factory=list)
    metadata_id: str = ""

    def __post_init__(self) -> None:
        if not self.metadata_id:
            self.metadata_id = self.name


class ToolCallFunction(TypedDict):
    """Provider-agnostic representation of a tool invocation."""

    name: str
    arguments: str


class ToolCall(TypedDict):
    """Provider-agnostic tool call produced by an LLM.

    Both OpenAI and Anthropic providers map their native tool-use
    representations into this unified shape.
    """

    id: str
    type: str
    function: ToolCallFunction


class TokenUsage(TypedDict):
    """Token consumption for a single LLM turn."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


ToolResultCallback = Callable[[ToolCall, Any], Awaitable[None]]
StreamingToolFunction = Callable[..., AsyncIterator[Any]]


@dataclass
class AgentStart:
    user_input: Iterable[ExtendedInputContentPart]
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentEnd:
    response: str
    time_taken: float | None = None
    exceeded: bool = False
    interrupted: bool = False


@dataclass
class ToolCallDelta:
    """Partial update for a tool call within a streaming chunk."""

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str | None = None


@dataclass
class LLMChunkDelta:
    """Provider-agnostic representation of a single streaming chunk delta."""

    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[ToolCallDelta] | None = None


@dataclass
class LLMChunk:
    chunk: LLMChunkDelta | None


@dataclass
class LLMStart:
    messages: list["Message"]
    tools: Any


@dataclass
class LLMEnd:
    content: str | None
    reasoning_content: str | None
    tool_calls: list[ToolCall]
    usage: TokenUsage | None


@dataclass
class ExecutionStart:
    tool_calls: list[ToolCall]


@dataclass
class ExecutionEnd:
    results: list[tuple[ToolCall, Any]]


@dataclass
class ToolStart:
    tool_call: ToolCall


@dataclass
class ToolProgress:
    tool_call: ToolCall
    chunk: Any


@dataclass
class ToolEnd:
    tool_call: ToolCall
    result: Any


@dataclass
class MemoryUpdate:
    usage: TokenUsage


ToolEvent = Union[ToolStart, ToolProgress, ToolEnd]


AgentEvent = Union[
    AgentStart,
    AgentEnd,
    ExecutionEnd,
    ExecutionStart,
    LLMChunk,
    LLMEnd,
    LLMStart,
    MemoryUpdate,
    ToolEnd,
    ToolProgress,
    ToolStart,
]
