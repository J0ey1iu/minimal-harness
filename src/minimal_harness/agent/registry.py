from __future__ import annotations

from typing import Awaitable, Callable, Protocol, runtime_checkable

from minimal_harness.registry import Registry, RegistryChangeEvent
from minimal_harness.types import AgentMetadata


@runtime_checkable
class AgentRegistryProtocol(Protocol):
    """Protocol for agent metadata registration and discovery."""

    async def register(self, metadata: AgentMetadata) -> AgentMetadata: ...

    async def unregister(self, name: str) -> bool: ...

    async def get(self, name: str) -> AgentMetadata | None: ...

    async def get_all(self, exclude: str | None = None) -> list[AgentMetadata]: ...

    async def names(self) -> list[str]: ...

    async def clear(self) -> None: ...

    async def add_listener(
        self, listener: Callable[[RegistryChangeEvent], Awaitable[None]]
    ) -> None: ...

    async def remove_listener(
        self, listener: Callable[[RegistryChangeEvent], Awaitable[None]]
    ) -> None: ...


class AgentRegistry(Registry[AgentMetadata]):
    def __init__(self) -> None:
        super().__init__()
        self._name_to_id: dict[str, str] = {}

    async def register(self, metadata: AgentMetadata) -> AgentMetadata:
        await self._register(metadata.metadata_id, metadata)
        self._name_to_id[metadata.name] = metadata.metadata_id
        return metadata

    async def unregister(self, name: str) -> bool:
        metadata_id = self._name_to_id.get(name, name)
        self._name_to_id.pop(name, None)
        return await super().unregister(metadata_id)

    async def get(self, name: str) -> AgentMetadata | None:
        metadata_id = self._name_to_id.get(name, name)
        return await super().get(metadata_id)

    async def get_all(self, exclude: str | None = None) -> list[AgentMetadata]:
        if exclude is None:
            return await super().get_all()
        exclude_key = self._name_to_id.get(exclude, exclude)
        return await super().get_all(exclude=exclude_key)

    async def clear(self) -> None:
        self._name_to_id.clear()
        await super().clear()
