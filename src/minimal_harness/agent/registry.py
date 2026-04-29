from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Protocol, Sequence, runtime_checkable

from minimal_harness.registry import Registry

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


class AgentRegistry(Registry[AgentMetadata]):
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
        self._register(
            agent_name,
            AgentMetadata(
                name=agent_name,
                description=agent_description,
                agent=agent,
                tools=tools or [],
            ),
        )
