from __future__ import annotations

from typing import TYPE_CHECKING

from minimal_harness.tool.base import create_streaming_tool
from minimal_harness.tool.registry import ToolRegistryProtocol
from minimal_harness.types import StreamingToolFunction

if TYPE_CHECKING:
    pass


def register_tool(
    name: str | None = None,
    description: str | None = None,
    parameters: dict | None = None,
    display_name: str | None = None,
    display_name_locale: dict[str, str] | None = None,
    description_locale: dict[str, str] | None = None,
    *,
    registry: ToolRegistryProtocol,
):
    def decorator(fn: StreamingToolFunction) -> StreamingToolFunction:
        tool_name = name or fn.__name__
        tool = create_streaming_tool(
            tool_name,
            fn,
            description,
            parameters,
            display_name,
            display_name_locale=display_name_locale,
            description_locale=description_locale,
        )
        registry.register(tool)
        return fn

    return decorator
