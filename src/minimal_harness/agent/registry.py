from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from minimal_harness.registry import Registry
from minimal_harness.types import AgentMetadata


@runtime_checkable
class AgentRegistryProtocol(Protocol):
    """Protocol for agent metadata registration and discovery."""

    def register(self, metadata: AgentMetadata) -> AgentMetadata: ...

    def unregister(self, name: str) -> bool: ...

    def get(self, name: str) -> AgentMetadata | None: ...

    def get_all(self) -> list[AgentMetadata]: ...

    def names(self) -> list[str]: ...

    def clear(self) -> None: ...

    def add_listener(self, listener: Callable[[], None]) -> None: ...

    def remove_listener(self, listener: Callable[[], None]) -> None: ...


class AgentRegistry(Registry[AgentMetadata]):
    def __init__(self) -> None:
        super().__init__()
        self._name_to_id: dict[str, str] = {}

    def register(self, metadata: AgentMetadata) -> AgentMetadata:
        self._register(metadata.metadata_id, metadata)
        self._name_to_id[metadata.name] = metadata.metadata_id
        return metadata

    def unregister(self, name: str) -> bool:
        metadata_id = self._name_to_id.get(name, name)
        self._name_to_id.pop(name, None)
        return super().unregister(metadata_id)

    def get(self, name: str) -> AgentMetadata | None:
        metadata_id = self._name_to_id.get(name, name)
        return super().get(metadata_id)

    def clear(self) -> None:
        self._name_to_id.clear()
        super().clear()
