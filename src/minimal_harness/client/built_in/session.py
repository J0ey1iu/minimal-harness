from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minimal_harness.session import Session


class SessionStatus(Enum):
    IDLE = auto()
    RUNNING = auto()


@dataclass
class ConversationSession:
    """Run-time binding for a single conversation.

    Wraps a L2 ``Session`` entity with the runtime information
    needed by the TUI layer (agent binding, tool resolution,
    cancellation).
    """

    session: Session
    agent_metadata_id: str
    tool_names: list[str] = field(default_factory=list)
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)

    def interrupt(self) -> None:
        self.stop_event.set()

    def reset(self) -> None:
        self.stop_event.clear()
