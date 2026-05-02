"""Client package - re-exports from types.py for backward compat."""

from minimal_harness.types import (
    AgentEnd as AgentEndEvent,
)
from minimal_harness.types import (
    AgentEvent as Event,
)
from minimal_harness.types import (
    AgentStart as AgentStartEvent,
)
from minimal_harness.types import (
    ExecutionEnd as ExecutionEndEvent,
)
from minimal_harness.types import (
    ExecutionStart as ExecutionStartEvent,
)
from minimal_harness.types import (
    LLMChunk as LLMChunkEvent,
)
from minimal_harness.types import (
    LLMEnd as LLMEndEvent,
)
from minimal_harness.types import (
    LLMStart as LLMStartEvent,
)
from minimal_harness.types import (
    MemoryUpdate as MemoryUpdateEvent,
)
from minimal_harness.types import (
    ToolEnd as ToolEndEvent,
)
from minimal_harness.types import (
    ToolProgress as ToolProgressEvent,
)
from minimal_harness.types import (
    ToolStart as ToolStartEvent,
)


def to_client_event(event: Event) -> Event:
    return event


__all__ = [
    "Event",
    "AgentStartEvent",
    "AgentEndEvent",
    "LLMChunkEvent",
    "ExecutionStartEvent",
    "ExecutionEndEvent",
    "LLMStartEvent",
    "LLMEndEvent",
    "ToolStartEvent",
    "ToolProgressEvent",
    "ToolEndEvent",
    "MemoryUpdateEvent",
    "to_client_event",
]
