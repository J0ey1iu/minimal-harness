from __future__ import annotations

import logging
import runpy
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from minimal_harness.tool.script_parser import parse_tool_script
from minimal_harness.types import ExternalScriptToolBinding

if TYPE_CHECKING:
    from minimal_harness.tool.base import StreamingToolFunction
    from minimal_harness.tool.registry import ToolRegistryProtocol

logger = logging.getLogger(__name__)


async def _register_script_tool(
    file_path: Path,
    name: str,
    description: str,
    parameters: dict,
    registry: ToolRegistryProtocol,
    display_name: str | None = None,
    display_name_locale: dict[str, str] | None = None,
    description_locale: dict[str, str] | None = None,
) -> str | None:
    try:
        await registry.register_from_binding(
            name=name,
            description=description,
            parameters=parameters,
            binding=ExternalScriptToolBinding(script_path=str(file_path)),
            display_name=display_name,
            display_name_locale=display_name_locale,
            description_locale=description_locale,
        )
        logger.info("tool.external.loaded name=%s path=%s", name, file_path)
        return name
    except Exception:
        logger.exception(
            "tool.external.register.error name=%s path=%s", name, file_path
        )
        return None


async def load_tools_from_file(
    path: str | Path, registry: ToolRegistryProtocol
) -> list[str]:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        logger.error("tool.script.not_found path=%s", file_path)
        return []

    # ── Try new single-tool convention first (TOOL_NAME / execute()) ──
    parse_result = parse_tool_script(file_path)
    if parse_result.is_valid:
        loaded = await _register_script_tool(
            file_path=file_path,
            name=parse_result.name,
            description=parse_result.description,
            parameters=parse_result.parameters,
            registry=registry,
            display_name=parse_result.display_name,
        )
        if loaded:
            return [loaded]
        return []

    # ── Fallback: legacy register() / @register_tool() pattern ──
    captured: list[
        tuple[
            str,
            str,
            dict,
            StreamingToolFunction,
            str | None,
            dict[str, str] | None,
            dict[str, str] | None,
        ]
    ] = []

    def capture_register_tool(
        name: str | None = None,
        description: str | None = None,
        parameters: dict | None = None,
        display_name: str | None = None,
        display_name_locale: dict[str, str] | None = None,
        description_locale: dict[str, str] | None = None,
    ) -> Callable[..., Callable[..., Any]]:
        def decorator(fn: StreamingToolFunction) -> StreamingToolFunction:
            tool_name = name or fn.__name__
            tool_desc = description or (fn.__doc__ or "").strip()
            tool_params = parameters or {}
            captured.append(
                (
                    tool_name,
                    tool_desc,
                    tool_params,
                    fn,
                    display_name,
                    display_name_locale,
                    description_locale,
                )
            )
            return fn

        return decorator

    def capture_register(
        name: str,
        description: str,
        parameters: dict,
        fn: StreamingToolFunction,
        display_name: str | None = None,
        display_name_locale: dict[str, str] | None = None,
        description_locale: dict[str, str] | None = None,
    ) -> None:
        captured.append(
            (
                name,
                description,
                parameters,
                fn,
                display_name,
                display_name_locale,
                description_locale,
            )
        )

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
        logger.exception("tool.script.load.error path=%s", file_path)
        return []
    finally:
        sys.path = original_sys_path

    loaded_names: list[str] = []
    for (
        tool_name,
        tool_desc,
        tool_params,
        fn,
        tool_display_name,
        dn_locale,
        desc_locale,
    ) in captured:
        result = await _register_script_tool(
            file_path=file_path,
            name=tool_name,
            description=tool_desc,
            parameters=tool_params,
            registry=registry,
            display_name=tool_display_name,
            display_name_locale=dn_locale,
            description_locale=desc_locale,
        )
        if result:
            loaded_names.append(result)

    return loaded_names


async def load_tools_from_directory(
    path: str | Path, registry: ToolRegistryProtocol, pattern: str = "*.py"
) -> list[str]:
    dir_path = Path(path).expanduser().resolve()
    if not dir_path.is_dir():
        logger.error("tool.dir.not_found path=%s", dir_path)
        return []

    loaded_names: list[str] = []
    for script_file in sorted(dir_path.glob(pattern)):
        loaded_names.extend(await load_tools_from_file(script_file, registry))
    return loaded_names


async def load_external_tools(
    tools_path: str | Path | None, registry: ToolRegistryProtocol
) -> list[str]:
    if not tools_path:
        return []

    p = Path(str(tools_path)).expanduser().resolve()
    if p.is_dir():
        return await load_tools_from_directory(p, registry)
    if p.is_file():
        return await load_tools_from_file(p, registry)

    logger.error("tool.path.not_found path=%s", p)
    return []
