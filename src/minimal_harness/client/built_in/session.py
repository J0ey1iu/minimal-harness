from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol


class SessionStatus(Enum):
    IDLE = auto()
    RUNNING = auto()


class Session(Protocol):
    @property
    def session_id(self) -> str: ...
    @property
    def agent_metadata_id(self) -> str: ...
    @property
    def memory_id(self) -> str: ...
    @property
    def tool_names(self) -> list[str]: ...
    @property
    def stop_event(self) -> asyncio.Event: ...
    def interrupt(self) -> None: ...
    def reset(self) -> None: ...


@dataclass
class ConversationSession:
    session_id: str
    agent_metadata_id: str
    memory_id: str
    tool_names: list[str] = field(default_factory=list)
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    name: str = ""

    def interrupt(self) -> None:
        self.stop_event.set()

    def reset(self) -> None:
        self.stop_event.clear()
