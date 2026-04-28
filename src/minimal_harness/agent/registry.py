from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from minimal_harness.agent.protocol import Agent
    from minimal_harness.memory import Memory
    from minimal_harness.tool.base import Tool
    from minimal_harness.types import AgentEvent


DEFAULT_QUEUE_SIZE = 1000


@dataclass
class AgentMetadata:
    name: str
    description: str
    agent: Agent


@runtime_checkable
class AgentRegistryProtocol(Protocol):
    def register(
        self, agent: Agent, *, name: str | None = None, description: str | None = None
    ) -> None: ...
    def unregister(self, name: str) -> bool: ...
    def get(self, name: str) -> AgentMetadata | None: ...
    def get_all(self) -> list[AgentMetadata]: ...
    def names(self) -> list[str]: ...
    def clear(self) -> None: ...
    def add_listener(self, listener: Callable[[], None]) -> None: ...
    def remove_listener(self, listener: Callable[[], None]) -> None: ...


@dataclass
class HandoffTarget:
    session_id: str
    name: str
    agent: Agent
    memory: Memory
    tools: list[Tool]
    default_tools: list[str] | None = None
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    event_queue: asyncio.Queue["AgentEvent"] = field(
        default_factory=lambda: asyncio.Queue(maxsize=DEFAULT_QUEUE_SIZE)
    )

    def interrupt(self) -> None:
        self.stop_event.set()

    def reset(self) -> None:
        self.stop_event.clear()


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, AgentMetadata] = {}
        self._listeners: list[Callable[[], None]] = []

    def register(
        self,
        agent: Agent,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        agent_name = name or getattr(agent, "name", None) or agent.__class__.__name__
        agent_description = description or getattr(agent, "description", None) or ""
        self._agents[agent_name] = AgentMetadata(
            name=agent_name,
            description=agent_description,
            agent=agent,
        )
        self._notify()

    def unregister(self, name: str) -> bool:
        if name in self._agents:
            del self._agents[name]
            self._notify()
            return True
        return False

    def get(self, name: str) -> AgentMetadata | None:
        return self._agents.get(name)

    def get_all(self) -> list[AgentMetadata]:
        return list(self._agents.values())

    def names(self) -> list[str]:
        return list(self._agents.keys())

    def clear(self) -> None:
        self._agents.clear()
        self._notify()

    def add_listener(self, listener: Callable[[], None]) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[], None]) -> None:
        self._listeners.remove(listener)

    def _notify(self) -> None:
        for listener in self._listeners:
            listener()
