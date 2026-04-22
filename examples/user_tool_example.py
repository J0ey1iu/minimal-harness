#!/usr/bin/env python3
"""
Example user tool script for Minimal Harness.

This file demonstrates how to write custom tools that the TUI can load
at runtime. You do NOT need to install minimal_harness — the framework
injects `register_tool` and `register` into your script's namespace
automatically.

Two ways to register a tool:
  1. @register_tool decorator  —  wraps an async generator function
  2. register() function call  —  manual registration with an existing function

Place this file (or a directory of such files) at the path configured in
~/.minimal_harness/config.json under the "tools_path" key.

Quick start:
  1. Copy this file to ~/.minimal_harness/tools/  (or any directory)
  2. Set "tools_path" in ~/.minimal_harness/config.json to that directory
  3. Restart the TUI — your tools will appear in the tool selector (Ctrl+T)

NOTE: The shebang (#!) line above ensures this script runs with the Python
interpreter of your choice. This is important when your tool needs access
to Python packages that are installed in a different Python environment than
the TUI itself.

WINDOWS USERS: The shebang is parsed literally and passed to the OS as-is.
`#!/usr/bin/env python3` will fail on Windows because `/usr/bin/env` does
not exist. Choose one of these alternatives instead:
  - `#!py -3.9`  (recommended — uses the Windows Python Launcher)
  - `#!python3.9`  (only works if python3.9.exe is on PATH)
  - `#!C:/Users/You/.../python.exe`  (absolute path, not portable)
  - Omit the shebang entirely to fall back to the TUI's Python
"""

import asyncio
import sys
from typing import AsyncIterator

# NOTE: ``register_tool`` and ``register`` are injected by Minimal Harness at
# runtime—they do not need to be imported.  The framework passes them into
# your script's namespace automatically before execution begins.


# ── Method 1: @register_tool decorator ──────────────────────────────────────


@register_tool(  # noqa: F821  # type: ignore[name-defined]
    name="calculator",
    description="Evaluate a simple mathematical expression and return the result",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A math expression, e.g. '2 + 3 * 4'",
            },
        },
        "required": ["expression"],
    },
)
async def calculator(expression: str) -> AsyncIterator[dict]:
    """Evaluate a mathematical expression."""
    yield {"status": "progress", "message": f"Evaluating: {expression}"}
    try:
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        yield {"success": True, "result": result}
    except Exception as exc:
        yield {"success": False, "error": str(exc)}


@register_tool(  # noqa: F821  # type: ignore[name-defined]
    name="echo_repeat",
    description="Repeat a message a given number of times with progress updates",
    parameters={
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The message to repeat",
            },
            "count": {
                "type": "integer",
                "description": "Number of times to repeat (1-10)",
            },
        },
        "required": ["message", "count"],
    },
)
async def echo_repeat(message: str, count: int) -> AsyncIterator[dict]:
    """Echo a message multiple times, yielding progress along the way."""
    for i in range(1, count + 1):
        yield {"status": "progress", "message": f"[{i}/{count}] {message}"}
        await asyncio.sleep(0.5)
    yield {"success": True, "repeated": count}


# ── Method 2: register() direct call ────────────────────────────────────────


async def reverse_string(text: str) -> AsyncIterator[dict]:
    """Reverse the given string."""
    yield {"status": "progress", "message": f"Reversing '{text}'..."}
    yield {"success": True, "result": text[::-1]}


register(  # noqa: F821  # type: ignore[name-defined]
    name="reverse_string",
    description="Reverse a given string",
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The string to reverse",
            },
        },
        "required": ["text"],
    },
    fn=reverse_string,
)


@register_tool(  # noqa: F821  # type: ignore[name-defined]
    name="interpreter_info",
    description="Report the Python interpreter and environment being used by this tool",
    parameters={
        "type": "object",
        "properties": {},
    },
)
async def interpreter_info() -> AsyncIterator[dict]:
    """Debug tool to show which Python interpreter is running."""
    yield {
        "interpreter": sys.executable,
        "version": sys.version,
        "prefix": sys.prefix,
        "executable": sys.executable,
    }