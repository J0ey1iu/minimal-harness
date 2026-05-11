"""Tool collection — delegates to Layer 2 tool collector."""

from __future__ import annotations

from typing import Any

from minimal_harness.tool.collector import collect_tools as _collect
from minimal_harness.tool.registry import ToolRegistry


async def collect_tools(
    config: dict[str, Any],
    registry: ToolRegistry,
) -> None:
    await _collect(config, registry)
