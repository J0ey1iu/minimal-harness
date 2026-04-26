from __future__ import annotations

import logging
import runpy
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from minimal_harness.tool.registry import ToolRegistry

if TYPE_CHECKING:
    from minimal_harness.tool.base import StreamingToolFunction

logger = logging.getLogger(__name__)


def load_tools_from_file(path: str | Path, registry: ToolRegistry) -> list[str]:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        logger.error("Tool script not found: %s", file_path)
        return []

    captured: list[tuple[str, str, dict, StreamingToolFunction]] = []

    def capture_register_tool(
        name: str | None = None,
        description: str | None = None,
        parameters: dict | None = None,
    ) -> Callable[..., Callable[..., Any]]:
        def decorator(fn: StreamingToolFunction) -> StreamingToolFunction:
            tool_name = name or fn.__name__
            tool_desc = description or (fn.__doc__ or "").strip()
            tool_params = parameters or {}
            captured.append((tool_name, tool_desc, tool_params, fn))
            return fn

        return decorator

    def capture_register(
        name: str,
        description: str,
        parameters: dict,
        fn: StreamingToolFunction,
    ) -> None:
        captured.append((name, description, parameters, fn))

    ns: dict[str, Any] = {
        "register_tool": capture_register_tool,
        "register": capture_register,
    }

    original_sys_path = sys.path.copy()
    script_dir = str(file_path.parent)
    try:
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        original_module = sys.modules.get(file_path.stem)
        if file_path.stem in sys.modules:
            del sys.modules[file_path.stem]

        runpy.run_path(str(file_path), init_globals=ns, run_name=file_path.stem)

        if file_path.stem not in sys.modules or original_module is None:
            sys.modules.pop(file_path.stem, None)
        elif original_module is not None:
            sys.modules[file_path.stem] = original_module

    except Exception:
        logger.exception("Error loading tool script %s", file_path)
        return []
    finally:
        sys.path = original_sys_path

    loaded_names: list[str] = []
    for tool_name, tool_desc, tool_params, fn in captured:
        registry.register_external_tool(
            name=tool_name,
            description=tool_desc,
            parameters=tool_params,
            fn=fn,
            script_path=file_path,
        )
        loaded_names.append(tool_name)
        logger.info("Loaded external tool: %s", tool_name)

    return loaded_names


def load_tools_from_directory(
    path: str | Path, registry: ToolRegistry, pattern: str = "*.py"
) -> list[str]:
    dir_path = Path(path).expanduser().resolve()
    if not dir_path.is_dir():
        logger.error("Tool directory not found: %s", dir_path)
        return []

    loaded_names: list[str] = []
    for script_file in sorted(dir_path.glob(pattern)):
        loaded_names.extend(load_tools_from_file(script_file, registry))
    return loaded_names


def load_external_tools(
    tools_path: str | Path | None, registry: ToolRegistry
) -> list[str]:
    if not tools_path:
        return []

    p = Path(str(tools_path)).expanduser().resolve()
    if p.is_dir():
        return load_tools_from_directory(p, registry)
    if p.is_file():
        return load_tools_from_file(p, registry)

    logger.error("Tools path does not exist: %s", p)
    return []
