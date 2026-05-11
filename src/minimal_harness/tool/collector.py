"""Tool collection — aggregates built-in and external tools into the ToolRegistry.

This is a Layer 2 helper that knows how to discover and register
all available tools (built-in + external) into a ToolRegistry.
"""

from __future__ import annotations

import warnings
from typing import Any

from minimal_harness.tool.external_loader import load_external_tools
from minimal_harness.tool.registry import ToolRegistry, collect_builtin_tools


async def collect_tools(
    config: dict[str, Any],
    registry: ToolRegistry,
) -> None:
    if path := config.get("tools_path", "").strip():
        await load_external_tools(path, registry)
    existing = {t.name for t in await registry.get_all()}
    builtin_names = await collect_builtin_tools(registry)
    for name in existing & builtin_names:
        warnings.warn(
            f"External tool '{name}' overwrites built-in tool of the same name."
        )
