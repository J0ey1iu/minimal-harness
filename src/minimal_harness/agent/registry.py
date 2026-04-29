from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    from minimal_harness.agent.protocol import Agent
    from minimal_harness.tool.base import Tool


@dataclass
class AgentMetadata:
    name: str
    description: str
    agent: Agent
    tools: Sequence[Tool] = field(default_factory=list)


@runtime_checkable
class AgentRegistryProtocol(Protocol):
    def register(
        self,
        agent: Agent,
        *,
        name: str | None = None,
        description: str | None = None,
        tools: Sequence[Tool] | None = None,
    ) -> None: ...
    def unregister(self, name: str) -> bool: ...
    def get(self, name: str) -> AgentMetadata | None: ...
    def get_all(self) -> list[AgentMetadata]: ...
    def names(self) -> list[str]: ...
    def clear(self) -> None: ...
    def add_listener(self, listener: Callable[[], None]) -> None: ...
    def remove_listener(self, listener: Callable[[], None]) -> None: ...


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
        tools: Sequence[Tool] | None = None,
    ) -> None:
        agent_name = name or getattr(agent, "name", None) or agent.__class__.__name__
        agent_description = description or getattr(agent, "description", None) or ""
        self._agents[agent_name] = AgentMetadata(
            name=agent_name,
            description=agent_description,
            agent=agent,
            tools=tools or [],
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
