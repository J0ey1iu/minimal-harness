from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from minimal_harness.registry import Registry
from minimal_harness.tool.base import Tool, create_streaming_tool

if TYPE_CHECKING:
    from minimal_harness.tool.base import StreamingToolFunction


@runtime_checkable
class ToolRegistryProtocol(Protocol):
    """Protocol for tool registration and discovery."""

    def register(self, tool: Tool) -> None: ...

    def register_external_tool(
        self,
        name: str,
        description: str,
        parameters: dict,
        fn: StreamingToolFunction,
        uri: Path | str | None = None,
        **kwargs: Any,
    ) -> None: ...

    def unregister(self, name: str) -> bool: ...

    def get(self, name: str) -> Tool | None: ...

    def get_all(self) -> list[Tool]: ...

    def names(self) -> list[str]: ...

    def clear(self) -> None: ...


def collect_builtin_tools(registry: ToolRegistry) -> set[str]:
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
            registry.register(tool)
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


class ToolRegistry(Registry[Tool]):
    def register(self, tool: Tool) -> None:
        self._register(tool.name, tool)

    def register_external_tool(
        self,
        name: str,
        description: str,
        parameters: dict,
        fn: StreamingToolFunction,
        uri: Path | str | None = None,
        **kwargs: Any,
    ) -> None:
        tool = create_streaming_tool(name, fn, description, parameters)
        if uri is not None:
            from minimal_harness.tool.wrapper import ExternalToolWrapper

            tool.fn = ExternalToolWrapper(  # type: ignore[assignment]
                original_fn=fn,
                script_path=uri,
                tool_name=name,
                tool_description=description,
                tool_params=parameters,
            )
        self.register(tool)
