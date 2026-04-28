from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, Sequence

if TYPE_CHECKING:
    from minimal_harness.agent.protocol import Agent
    from minimal_harness.memory import Memory
    from minimal_harness.tool.base import Tool


class Session(Protocol):
    @property
    def session_id(self) -> str: ...
    @property
    def agent(self) -> Agent: ...
    @property
    def memory(self) -> Memory: ...
    @property
    def tools(self) -> Sequence[Tool]: ...
    @property
    def stop_event(self) -> asyncio.Event: ...
    def interrupt(self) -> None: ...
    def reset(self) -> None: ...


@dataclass
class ConversationSession:
    session_id: str
    agent: Agent
    memory: Memory
    tools: list[Tool]
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    name: str = ""
    default_tools: list[str] | None = None
    event_queue: asyncio.Queue = field(
        default_factory=lambda: asyncio.Queue(maxsize=1000)
    )

    def interrupt(self) -> None:
        self.stop_event.set()

    def reset(self) -> None:
        self.stop_event.clear()
