from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minimal_harness.agent import Session


@dataclass
class TUISession:
    session: Session
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    is_streaming: bool = False

    @property
    def session_id(self) -> str:
        return self.session.session_id

    @property
    def name(self) -> str:
        return self.session.name

    def interrupt(self) -> None:
        self.stop_event.set()

    def reset(self) -> None:
        self.stop_event.clear()
        self.is_streaming = False
