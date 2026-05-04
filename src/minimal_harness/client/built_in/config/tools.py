"""Tool collection — aggregates built-in and external tools into the ToolRegistry."""

from __future__ import annotations

import warnings
from typing import Any

from minimal_harness.tool.built_in.bash import get_tools as get_bash_tools
from minimal_harness.tool.built_in.local_file_operation import (
    get_tools as get_local_file_operation_tools,
)
from minimal_harness.tool.external_loader import load_external_tools
from minimal_harness.tool.registry import ToolRegistry


def collect_tools(
    config: dict[str, Any],
    registry: ToolRegistry,
) -> None:
    if path := config.get("tools_path", "").strip():
        load_external_tools(path, registry)
    existing = {t.name for t in registry.get_all()}
    for getter in (get_bash_tools, get_local_file_operation_tools):
        for name, tool in getter().items():
            if name in existing:
                warnings.warn(
                    f"External tool '{name}' overwrites built-in tool of the same name."
                )
            registry.register(tool)
