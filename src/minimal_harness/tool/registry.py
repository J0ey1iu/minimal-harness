from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from minimal_harness.registry import Registry
from minimal_harness.types import (
    ExternalScriptToolBinding,
    LocalToolBinding,
    RemoteToolBinding,
    ToolMetadata,
)

if TYPE_CHECKING:
    pass


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


async def collect_builtin_tools(registry: ToolRegistryProtocol) -> set[str]:
    """Register all built-in tools into the given registry.

    Returns the set of built-in tool names that were registered.
    """
    from minimal_harness.tool.built_in.bash import get_tools as get_bash_tools
    from minimal_harness.tool.built_in.local_file_operation import (
        get_tools as get_local_file_operation_tools,
    )

    names: set[str] = set()
    for getter in (get_bash_tools, get_local_file_operation_tools):
        for name, tool in getter().items():
            await registry.register(
                ToolMetadata(
                    name=tool.name,
                    display_name=tool.display_name,
                    description=tool.description,
                    parameters=tool.parameters,
                    display_name_locale=tool.display_name_locale,
                    description_locale=tool.description_locale,
                    binding=LocalToolBinding(fn=getattr(tool, "fn", None)),
                    metadata_id=tool.name,
                )
            )
            names.add(name)
    return names


def get_builtin_tool_names() -> set[str]:
    """Return the set of built-in tool names (without registering them)."""
    from minimal_harness.tool.built_in.bash import get_tools as get_bash_tools
    from minimal_harness.tool.built_in.local_file_operation import (
        get_tools as get_local_file_operation_tools,
    )

    names: set[str] = set()
    for getter in (get_bash_tools, get_local_file_operation_tools):
        names.update(getter().keys())
    return names


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
