from __future__ import annotations

import asyncio
import json
import logging
import os
import runpy
import sys
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from minimal_harness.tool.base import StreamingTool
from minimal_harness.tool.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ExternalToolWrapper:
    def __init__(
        self,
        original_fn: Callable[..., AsyncIterator[Any]],
        script_path: Path,
        tool_name: str,
        tool_description: str,
        tool_params: dict[str, Any],
    ) -> None:
        self._original_fn = original_fn
        self._script_path = script_path
        self._name = tool_name
        self._description = tool_description
        self._params = tool_params
        self._interpreter: list[str] | None = None

    def _detect_interpreter(self) -> list[str]:
        if self._interpreter is not None:
            return self._interpreter

        shebang = self._script_path.read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines()[0]
        if shebang.startswith("#!") and "python" in shebang.lower():
            interp = shebang[2:].strip().split()
            if interp:
                self._interpreter = interp
                return self._interpreter

        self._interpreter = [sys.executable]
        return self._interpreter

    def _get_subprocess_env(self) -> dict[str, str] | None:
        """Return env dict with venv PATH stripped so env(1) resolves to system Python."""
        if not hasattr(sys, "base_prefix") or sys.prefix == sys.base_prefix:
            return None  # Not in a venv, no need to clean PATH

        venv_bin = str(Path(sys.prefix) / "bin")
        path = os.environ.get("PATH", "")
        parts = [p for p in path.split(os.pathsep) if p != venv_bin]
        if len(parts) == len(path.split(os.pathsep)):
            return None  # venv bin not in PATH

        env = os.environ.copy()
        env["PATH"] = os.pathsep.join(parts)
        return env

    async def _run_subprocess(self, args: dict[str, Any]) -> AsyncIterator[Any]:
        interp = self._detect_interpreter()

        script_code = self._script_path.read_text(encoding="utf-8", errors="replace")

        runner_code = f"""
import sys, json, runpy, asyncio
from pathlib import Path

script_path = {repr(str(self._script_path))}
tool_name = {repr(self._name)}
args = json.loads({repr(json.dumps(args, default=str))})

captured = {{}}
def capture_register(name=None, desc=None, params=None, fn=None, description=None, parameters=None, **kwargs):
    actual_name = name or kwargs.get("name")
    actual_desc = desc or description or kwargs.get("desc") or kwargs.get("description")
    actual_params = params or parameters or kwargs.get("params") or kwargs.get("parameters")
    actual_fn = fn or kwargs.get("fn")
    captured[actual_name] = {{"name": actual_name, "desc": actual_desc, "params": actual_params, "fn": actual_fn}}
    return actual_fn
def capture_register_tool(name=None, desc=None, params=None, description=None, parameters=None, **kwargs):
    actual_name = name or desc or kwargs.get("name")
    actual_desc = description or desc or kwargs.get("description") or kwargs.get("desc")
    actual_params = parameters or params or kwargs.get("parameters") or kwargs.get("params")
    def decorator(fn): return capture_register(actual_name, actual_desc, actual_params, fn)
    return decorator

namespace = {{"register": capture_register, "register_tool": capture_register_tool}}
original_modules = set(sys.modules.keys())
script_dir = str(Path(script_path).parent)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    exec(compile({repr(script_code)}, script_path, 'exec'), namespace)
except Exception as e:
    print(json.dumps({{"error": str(e)}}), flush=True)
    sys.exit(1)

for mod_name in list(sys.modules.keys()):
    if mod_name not in original_modules:
        sys.modules.pop(mod_name, None)

tool_entry = captured.get(tool_name)
if tool_entry is None:
    print(json.dumps({{"error": f"Tool {{tool_name}} not found in script"}}), flush=True)
    sys.exit(1)
fn = tool_entry["fn"]

try:
    gen = fn(**args)
    async def consume_async():
        import inspect
        if inspect.isasyncgen(gen):
            async for chunk in gen:
                print(json.dumps(chunk, default=str), flush=True)
        elif asyncio.iscoroutine(gen):
            result = await gen
            print(json.dumps(result, default=str), flush=True)
        else:
            for chunk in gen:
                print(json.dumps(chunk, default=str), flush=True)
    asyncio.run(consume_async())
except Exception as e:
    print(json.dumps({{"error": str(e)}}), flush=True)
"""

        proc = await asyncio.create_subprocess_exec(
            *interp,
            "-c",
            runner_code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._get_subprocess_env(),
        )

        assert proc.stdout is not None
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                decoded = line.decode("utf-8").strip()
                if decoded:
                    try:
                        yield json.loads(decoded)
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON from subprocess: %s", decoded)
        finally:
            await proc.wait()

        if proc.returncode != 0:
            assert proc.stderr is not None
            stderr_data = await proc.stderr.read()
            stderr = stderr_data.decode("utf-8") if stderr_data else ""
            logger.error("External tool subprocess failed: %s", stderr)

    def __call__(self, **kwargs: Any) -> AsyncIterator[Any]:
        return self._run_subprocess(kwargs)


def _make_register_tool(
    captured_tools: list[StreamingTool],
) -> Callable[..., Callable[..., Any]]:
    def register_tool(
        name: str | None = None,
        description: str | None = None,
        parameters: dict | None = None,
    ) -> Callable[..., Callable[..., Any]]:
        def decorator(
            fn: Callable[..., AsyncIterator[Any]],
        ) -> Callable[..., AsyncIterator[Any]]:
            tool_name = name or fn.__name__
            tool_description = description or (fn.__doc__ or "").strip()
            tool_params = parameters or {}
            captured_tools.append(
                StreamingTool(
                    name=tool_name,
                    description=tool_description,
                    parameters=tool_params,
                    fn=fn,
                )
            )
            return fn

        return decorator

    return register_tool


def _make_register(captured_tools: list[StreamingTool]) -> Callable[..., None]:
    def register(
        name: str,
        description: str,
        parameters: dict,
        fn: Callable[..., AsyncIterator[Any]],
    ) -> None:
        captured_tools.append(
            StreamingTool(
                name=name,
                description=description,
                parameters=parameters,
                fn=fn,
            )
        )

    return register


def load_tools_from_file(path: str | Path) -> list[str]:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        logger.error("Tool script not found: %s", file_path)
        return []

    captured_tools: list[StreamingTool] = []
    register_tool = _make_register_tool(captured_tools)
    register = _make_register(captured_tools)

    namespace: dict[str, Any] = {
        "register_tool": register_tool,
        "register": register,
    }

    original_sys_path = sys.path.copy()
    script_dir = str(file_path.parent)
    try:
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        original_module = sys.modules.get(file_path.stem)
        if file_path.stem in sys.modules:
            del sys.modules[file_path.stem]

        runpy.run_path(str(file_path), init_globals=namespace, run_name=file_path.stem)

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
    registry = ToolRegistry.get_instance()
    for tool in captured_tools:
        wrapped = ExternalToolWrapper(
            original_fn=tool.fn,
            script_path=file_path,
            tool_name=tool.name,
            tool_description=tool.description,
            tool_params=tool.parameters,
        )
        tool.fn = wrapped  # type: ignore[assignment]
        registry.register(tool)
        loaded_names.append(tool.name)
        logger.info("Loaded external tool: %s", tool.name)

    return loaded_names


def load_tools_from_directory(path: str | Path, pattern: str = "*.py") -> list[str]:
    dir_path = Path(path).expanduser().resolve()
    if not dir_path.is_dir():
        logger.error("Tool directory not found: %s", dir_path)
        return []

    loaded_names: list[str] = []
    for script_file in sorted(dir_path.glob(pattern)):
        loaded_names.extend(load_tools_from_file(script_file))

    return loaded_names


def load_external_tools(tools_path: str | Path | None) -> list[str]:
    if not tools_path:
        return []

    p = Path(str(tools_path)).expanduser().resolve()
    if p.is_dir():
        return load_tools_from_directory(p)
    if p.is_file():
        return load_tools_from_file(p)

    logger.error("Tools path does not exist: %s", p)
    return []
