#!/usr/bin/env python3
"""
Standalone test driver for user tool scripts.

This script lets you develop and debug your tool scripts WITHOUT running the
Minimal Harness TUI. It loads your tool file, injects mock `register_tool`
and `register` helpers, and runs your tool function directly.

Usage:
  python test_user_tools.py                  # Run default tool with sample args
  python test_user_tools.py calculator       # Run specific tool
  python test_user_tools.py calculator '{"expression": "2+2"}'
  python test_user_tools.py echo_repeat '{"message": "hello", "count": 3}'

Place this file next to your tool scripts (e.g., in the examples/ directory)
and point TOOLS_FILE at your tool script.

This is the same pattern documented in docs/external-scripts-loading.md §9.3.
"""
import asyncio
import json
import sys
from collections.abc import Callable
from pathlib import Path

TOOLS_FILE = Path(__file__).parent / "user_tool_example.py"

_captured: list[tuple[str, Callable]] = []


def register_tool(name=None, description=None, parameters=None):
    def decorator(fn):
        tool_name = name or fn.__name__
        _captured.append((tool_name, fn))
        return fn

    return decorator


def register(name, description, parameters, fn):
    _captured.append((name, fn))


def load_tool(path: Path, tool_name: str) -> Callable:
    _captured.clear()
    ns: dict = {"register_tool": register_tool, "register": register}
    code = path.read_text(encoding="utf-8")
    if code.startswith("#!"):
        code = "\n".join(code.split("\n")[1:])
    exec(compile(code, str(path), "exec"), ns)
    for name, fn in _captured:
        if name == tool_name:
            _captured.clear()
            return fn
    available = [n for n, _ in _captured]
    _captured.clear()
    raise AttributeError(f"Tool '{tool_name}' not found in {path}. Available: {available}")


async def run_tool(tool_name: str, args_json: str | None = None) -> None:
    tool_fn = load_tool(TOOLS_FILE, tool_name)
    default_args = {
        "calculator": {"expression": "2 + 2"},
        "echo_repeat": {"message": "hello", "count": 3},
        "reverse_string": {"text": "Minimal Harness"},
        "always_fail": {"message": "test error"},
        "interpreter_info": {},
    }
    args = json.loads(args_json) if args_json else default_args.get(tool_name, {})
    gen = tool_fn(**args)
    async for chunk in gen:
        print(json.dumps(chunk, default=str))
        if "success" in chunk or "error" in chunk:
            break


AVAILABLE_TOOLS = ["calculator", "echo_repeat", "reverse_string", "always_fail", "interpreter_info"]


if __name__ == "__main__":
    tool_name = sys.argv[1] if len(sys.argv) > 1 else "calculator"
    args = sys.argv[2] if len(sys.argv) > 2 else None

    if tool_name not in AVAILABLE_TOOLS:
        print(f"Unknown tool: {tool_name}", file=sys.stderr)
        print(f"Available tools: {', '.join(AVAILABLE_TOOLS)}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run_tool(tool_name, args))
