"""Tool collection — aggregates built-in and external tools."""

from __future__ import annotations

from typing import Any

from minimal_harness.tool.base import StreamingTool
from minimal_harness.tool.built_in.bash import get_tools as get_bash_tools
from minimal_harness.tool.built_in.local_file_operation import (
    get_tools as get_local_file_operation_tools,
)
from minimal_harness.tool.external_loader import load_external_tools
from minimal_harness.tool.registry import ToolRegistry


def collect_tools(
    config: dict[str, Any],
    registry: ToolRegistry,
) -> dict[str, StreamingTool]:
    import warnings

    if path := config.get("tools_path", "").strip():
        load_external_tools(path, registry)
    tools: dict[str, StreamingTool] = {}
    for getter in (get_bash_tools, get_local_file_operation_tools):
        tools.update(getter())
    for t in registry.get_all():
        if t.name in tools:
            warnings.warn(
                f"External tool '{t.name}' overwrites built-in tool of the same name."
            )
        tools[t.name] = t
    return tools
