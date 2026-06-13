from __future__ import annotations

from typing import Protocol, runtime_checkable

from minimal_harness.registry import Registry
from minimal_harness.types import (
    ExternalScriptToolBinding,
    LocalToolBinding,
    RemoteToolBinding,
    ToolMetadata,
)


@runtime_checkable
class ToolRegistryProtocol(Protocol):
    """Protocol for tool registration and discovery."""

    async def register(self, metadata: ToolMetadata) -> None: ...

    async def register_from_binding(
        self,
        name: str,
        description: str,
        parameters: dict,
        binding: LocalToolBinding | ExternalScriptToolBinding | RemoteToolBinding,
        display_name: str | None = None,
        display_name_locale: dict[str, str] | None = None,
        description_locale: dict[str, str] | None = None,
    ) -> None: ...

    async def unregister(self, name: str) -> bool: ...

    async def get(self, name: str) -> ToolMetadata | None: ...

    async def get_all(self) -> list[ToolMetadata]: ...

    async def names(self) -> list[str]: ...

    async def clear(self) -> None: ...


class ToolRegistry(Registry[ToolMetadata]):
    async def register(self, metadata: ToolMetadata) -> None:
        await self._register(metadata.metadata_id, metadata)

    async def register_from_binding(
        self,
        name: str,
        description: str,
        parameters: dict,
        binding: LocalToolBinding | ExternalScriptToolBinding | RemoteToolBinding,
        display_name: str | None = None,
        display_name_locale: dict[str, str] | None = None,
        description_locale: dict[str, str] | None = None,
    ) -> None:
        metadata = ToolMetadata(
            name=name,
            display_name=display_name or name,
            description=description,
            parameters=parameters,
            display_name_locale=display_name_locale,
            description_locale=description_locale,
            binding=binding,
        )
        await self.register(metadata)
