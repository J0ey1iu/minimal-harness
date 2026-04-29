from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from minimal_harness.registry import Registry
from minimal_harness.tool.base import Tool, create_streaming_tool

if TYPE_CHECKING:
    from minimal_harness.tool.base import StreamingToolFunction


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
