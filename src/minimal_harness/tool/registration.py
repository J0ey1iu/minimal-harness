from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from minimal_harness.types import LocalToolBinding, StreamingToolFunction, ToolMetadata

if TYPE_CHECKING:
    from minimal_harness.tool.registry import ToolRegistryProtocol


def _sync_register(registry: ToolRegistryProtocol, metadata: ToolMetadata) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(registry.register(metadata))
    else:
        asyncio.create_task(registry.register(metadata))


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
        metadata = ToolMetadata(
            name=tool_name,
            display_name=display_name or tool_name,
            description=description or (fn.__doc__ or "").strip(),
            parameters=parameters or {},
            display_name_locale=display_name_locale,
            description_locale=description_locale,
            binding=LocalToolBinding(fn=fn),
        )
        _sync_register(registry, metadata)
        return fn

    return decorator
