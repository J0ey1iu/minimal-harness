from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

from minimal_harness.registry import Registry

if TYPE_CHECKING:
    pass


@dataclass
class AgentMetadata:
    name: str
    description: str = ""
    system_prompt: str = ""
    agent_type: str = "simple"
    tool_names: list[str] = field(default_factory=list)
    metadata_id: str = ""

    def __post_init__(self) -> None:
        if not self.metadata_id:
            self.metadata_id = self.name


@runtime_checkable
class AgentRegistryProtocol(Protocol):
    def register(
        self,
        *,
        name: str,
        description: str = "",
        system_prompt: str = "",
        agent_type: str = "simple",
        tool_names: list[str] | None = None,
        metadata_id: str | None = None,
    ) -> AgentMetadata: ...
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
        *,
        name: str,
        description: str = "",
        system_prompt: str = "",
        agent_type: str = "simple",
        tool_names: list[str] | None = None,
        metadata_id: str | None = None,
    ) -> AgentMetadata:
        metadata = AgentMetadata(
            name=name,
            description=description,
            system_prompt=system_prompt,
            agent_type=agent_type,
            tool_names=tool_names or [],
            metadata_id=metadata_id or name,
        )
        self._register(metadata.metadata_id, metadata)
        return metadata
