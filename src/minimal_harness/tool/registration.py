from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from minimal_harness.tool.base import create_streaming_tool
from minimal_harness.tool.registry import ToolRegistryProtocol
from minimal_harness.types import StreamingToolFunction

if TYPE_CHECKING:
    pass


def _sync_register(registry: ToolRegistryProtocol, tool: object) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(registry.register(tool))  # type: ignore[arg-type]
    else:
        asyncio.create_task(registry.register(tool))  # type: ignore[arg-type]


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
        _sync_register(registry, tool)
        return fn

    return decorator
