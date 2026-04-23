from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, AsyncIterator, Callable

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

        script_code = self._script_path.read_text(encoding="utf-8", errors="replace")

        runner_code = f"""
import sys, json, runpy, asyncio, traceback
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
    print(json.dumps({{"error": str(e), "traceback": traceback.format_exc()}}), flush=True)
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
                        logger.warning("Invalid JSON from subprocess: %s", decoded)
        finally:
            await proc.wait()

        if proc.returncode != 0:
            assert proc.stderr is not None
            stderr_data = await proc.stderr.read()
            stderr = stderr_data.decode("utf-8") if stderr_data else ""
            logger.error("External tool subprocess failed: %s", stderr)
            yield {"error": f"External tool subprocess failed with code {proc.returncode}", "stderr": stderr}

    def __call__(self, **kwargs: Any) -> AsyncIterator[Any]:
        return self._run_subprocess(kwargs)
