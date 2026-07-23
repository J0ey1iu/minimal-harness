from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from minimal_harness.tool.base import ToolExecutionError

logger = logging.getLogger(__name__)


class ExternalToolWrapper:
    def __init__(
        self,
        original_fn: Callable[..., AsyncIterator[Any]],
        script_path: Path | str,
        tool_name: str,
        tool_description: str,
        tool_params: dict[str, Any],
        display_name_locale: dict[str, str] | None = None,
        description_locale: dict[str, str] | None = None,
    ) -> None:
        self._original_fn = original_fn
        self._script_path = (
            Path(script_path) if isinstance(script_path, str) else script_path
        )
        self._name = tool_name
        self._description = tool_description
        self._params = tool_params
        self._display_name_locale = display_name_locale
        self._description_locale = description_locale
        self._interpreter: list[str] | None = None

    def _detect_interpreter(self) -> list[str]:
        if self._interpreter is not None:
            return self._interpreter

        with self._script_path.open(encoding="utf-8", errors="ignore") as f:
            shebang = f.readline()
        if shebang.startswith("#!") and "python" in shebang.lower():
            interp = shebang[2:].strip().split()
            if interp:
                self._interpreter = interp
                return self._interpreter

        self._interpreter = [sys.executable]
        return self._interpreter

    def _get_subprocess_env(self) -> dict[str, str] | None:
        if not hasattr(sys, "base_prefix") or sys.prefix == sys.base_prefix:
            return None

        venv_bin = str(Path(sys.prefix) / "bin")
        path = os.environ.get("PATH", "")
        parts = [p for p in path.split(os.pathsep) if p != venv_bin]
        if len(parts) == len(path.split(os.pathsep)):
            return None

        env = os.environ.copy()
        env["PATH"] = os.pathsep.join(parts)
        return env

    async def _run_subprocess(self, args: dict[str, Any]) -> AsyncIterator[Any]:
        interp = self._detect_interpreter()

        runner_code = f"""
import sys, json, asyncio, traceback, os
from pathlib import Path

script_path = {repr(str(self._script_path))}
tool_name = {repr(self._name)}
args = json.loads({repr(json.dumps(args, default=str))})

script_dir = str(Path(script_path).parent)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
os.chdir(script_dir)

# Legacy register() support — injected as no-ops so old scripts don't crash
captured = {{}}
def _capture_register(name=None, desc=None, params=None, fn=None, description=None, parameters=None, display_name=None, display_name_locale=None, description_locale=None, **kwargs):
    actual_name = name or kwargs.get("name")
    actual_desc = desc or description or kwargs.get("desc") or kwargs.get("description")
    actual_params = params or parameters or kwargs.get("params") or kwargs.get("parameters")
    actual_fn = fn or kwargs.get("fn")
    captured[actual_name] = {{"name": actual_name, "desc": actual_desc, "params": actual_params, "fn": actual_fn}}
    return actual_fn
def _capture_register_tool(name=None, desc=None, params=None, description=None, parameters=None, display_name=None, display_name_locale=None, description_locale=None, **kwargs):
    actual_name = name or desc or kwargs.get("name")
    actual_desc = description or desc or kwargs.get("description") or kwargs.get("desc")
    actual_params = parameters or params or kwargs.get("parameters") or kwargs.get("params")
    def decorator(fn): return _capture_register(actual_name, actual_desc, actual_params, fn)
    return decorator

namespace = {{"register": _capture_register, "register_tool": _capture_register_tool}}
with open(script_path, encoding="utf-8") as f:
    exec(compile(f.read(), script_path, 'exec'), namespace)

# Prefer new execute() convention, fall back to legacy register() capture
fn = namespace.get('execute')
if fn is None:
    tool_entry = captured.get(tool_name)
    if tool_entry:
        fn = tool_entry.get("fn")
if fn is None:
    print(json.dumps({{"error": "No execute() function or register() call found for tool: " + repr(tool_name)}}), flush=True)
    sys.exit(1)

if not callable(fn):
    print(json.dumps({{"error": "execute is not a callable function"}}), flush=True)
    sys.exit(1)

try:
    gen = fn(**args)
    import inspect
    if inspect.isasyncgen(gen):
        async def consume():
            async for chunk in gen:
                print(json.dumps(chunk, default=str), flush=True)
    elif asyncio.iscoroutine(gen):
        async def consume():
            result = await gen
            print(json.dumps(result, default=str), flush=True)
    else:
        async def consume():
            for chunk in gen:
                print(json.dumps(chunk, default=str), flush=True)
    asyncio.run(consume())
except Exception as e:
    print(json.dumps({{"error": str(e), "traceback": traceback.format_exc()}}), flush=True)
    sys.exit(1)
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
                        yield decoded
        finally:
            if proc.returncode is None:
                proc.kill()
            await proc.wait()

        if proc.returncode != 0:
            assert proc.stderr is not None
            stderr_data = await proc.stderr.read()
            stderr = stderr_data.decode("utf-8") if stderr_data else ""
            logger.error(
                "tool.subprocess.error name=%s script=%s code=%d stderr=%s",
                self._name,
                self._script_path,
                proc.returncode,
                stderr,
            )
            raise ToolExecutionError(
                f"External tool subprocess failed with code {proc.returncode}",
                stderr,
            )

    def __call__(self, **kwargs: Any) -> AsyncIterator[Any]:
        return self._run_subprocess(kwargs)
