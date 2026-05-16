from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from minimal_harness.types import LocalToolBinding, StreamingToolFunction, ToolMetadata

if TYPE_CHECKING:
    from minimal_harness.tool.registry import ToolRegistryProtocol

_PENDING_REGISTRATIONS: list[ToolMetadata] = []


def _sync_register(registry: ToolRegistryProtocol, metadata: ToolMetadata) -> None:
    """Fallback: schedule registration into a running loop or run synchronously."""
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
    registry: ToolRegistryProtocol | None = None,
):
    """Decorate an async generator function as a tool and optionally register it.

    When ``registry`` is provided the tool is registered immediately (the legacy
    approach).  The recommended pattern is to omit ``registry`` and call
    ``register_decorated_tools()`` during async setup — this guarantees
    registration completes before the first tool lookup.
    """

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
        fn._mh_tool_metadata = metadata  # type: ignore[attr-defined]
        if registry is not None:
            _sync_register(registry, metadata)
        else:
            _PENDING_REGISTRATIONS.append(metadata)
        return fn

    return decorator


async def register_decorated_tools(
    registry: ToolRegistryProtocol,
) -> list[str]:
    """Register all tools decorated with ``@register_tool`` (without a registry).

    Returns the list of registered tool names.
    """
    global _PENDING_REGISTRATIONS
    pending, _PENDING_REGISTRATIONS[:] = _PENDING_REGISTRATIONS, []
    for metadata in pending:
        await registry.register(metadata)
    return [m.name for m in pending]
