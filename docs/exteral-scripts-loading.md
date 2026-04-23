# External Scripts Loading

This document explains how `minimal_harness` dynamically loads user-provided Python scripts at runtime and registers the tools they define.

## 1. Overview

The harness allows users to write their own tools in plain Python files. When the harness starts (or on demand), it discovers these files and executes them. During execution, the user's script can register functions as tools via two helper functions injected into the script's global namespace:

- `register_tool(...)` – a decorator style registration.
- `register(...)` – an imperative registration.

Once the script finishes, every tool captured during execution is added to the global `ToolRegistry` and becomes available to the harness.

## 2. Why Subprocess Execution?

External tools are executed in a **subprocess** rather than in the harness's Python process. This is important because:

- **Python environment isolation**: The TUI may run under a different Python interpreter (e.g., via `uv run` or a virtual environment) than the user's system Python where packages like `python-pptx` are installed.
- **Access to user packages**: Subprocess execution allows external tools to access packages installed in the user's preferred Python environment.
- **Non-blocking execution**: The subprocess runs asynchronously, ensuring the event loop is not blocked while tools execute.

The `ExternalToolWrapper` class handles this by:
1. Detecting the script's Python interpreter via shebang line (e.g. `#!/usr/bin/env python3`) or falling back to `sys.executable`
2. Stripping the venv's `bin` directory from `PATH` so `env(1)` resolves to the system Python
3. Spawning an async subprocess using `asyncio.create_subprocess_exec`
4. Running the script's code via `python -c` with inline runner code that captures registered tools
5. Streaming JSON results line-by-line back to the harness

## 3. How It Works

### 3.1 Entry Points

Three public functions form the loading API:

| Function                                          | Purpose                                                             |
| ------------------------------------------------- | ------------------------------------------------------------------- |
| `load_tools_from_file(path)`                      | Load a single `.py` file.                                           |
| `load_tools_from_directory(path, pattern="*.py")` | Load every file matching `pattern` inside a directory.              |
| `load_external_tools(tools_path)`                 | Convenience dispatcher that accepts a file, a directory, or `None`. |

### 3.2 Step-by-step Execution of `load_tools_from_file`

#### Step 1 – Resolve the path

```python
file_path = Path(path).expanduser().resolve()
```

Tildes (`~`) are expanded and the path is made absolute so that later manipulations are unambiguous.

#### Step 2 – Prepare capture containers

```python
captured_tools: list[StreamingTool] = []
register_tool = _make_register_tool(captured_tools)
register      = _make_register(captured_tools)
```

Two closures are built. Both share the **same** `captured_tools` list. When the user's script calls either helper, a new `StreamingTool` object is appended to this list.

#### Step 3 – Build the script namespace

```python
namespace: dict[str, Any] = {
    "register_tool": register_tool,
    "register": register,
}
```

This dictionary becomes the *initial* global namespace of the user's script. Because the script is executed with these names pre-defined, the user can call them without importing anything.

#### Step 4 – Temporarily mutate `sys.path`

```python
original_sys_path = sys.path.copy()
script_dir = str(file_path.parent)
try:
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    ...
finally:
    sys.path = original_sys_path
```

Inserting the script's directory at the front of `sys.path` allows the script (and any helper modules placed next to it) to be imported as top-level packages during loading. The original list is restored in the `finally` block so that the harness's import environment is not permanently altered.

#### Step 5 – Clean `sys.modules`

```python
original_module = sys.modules.get(file_path.stem)
if file_path.stem in sys.modules:
    del sys.modules[file_path.stem]
```

If a module with the same name as the script (e.g. `my_tools` for `my_tools.py`) was already imported earlier, it is removed from `sys.modules`. This prevents Python's import cache from returning stale code when `runpy` eventually creates a module object for the script.

#### Step 6 – Run the script with `runpy.run_path`

```python
runpy.run_path(str(file_path), init_globals=namespace, run_name=file_path.stem)
```

`runpy.run_path` is a standard library utility that executes a file **in the current Python process**. The arguments mean:

- `str(file_path)` – the file to execute.
- `init_globals=namespace` – the dictionary that acts as the script's `__globals__`. The two registration helpers are already present.
- `run_name=file_path.stem` – the value assigned to `__name__` inside the script. This lets the user write the classic `if __name__ == "..."` guard, although it is usually unnecessary because the script is meant to run top-level statements.

During execution, every call to `register_tool` or `register` mutates the shared `captured_tools` list living in the harness.

#### Step 7 – Restore `sys.modules`

```python
if file_path.stem not in sys.modules or original_module is None:
    sys.modules.pop(file_path.stem, None)
elif original_module is not None:
    sys.modules[file_path.stem] = original_module
```

After `runpy` finishes, the module entry it created in `sys.modules` is either deleted (if there was no previous module) or restored to the original one. This keeps the interpreter's module cache clean and avoids shadowing real installed packages.

#### Step 8 – Wrap tools and register them

```python
registry = ToolRegistry.get_instance()
for tool in captured_tools:
    wrapped = ExternalToolWrapper(
        original_fn=tool.fn,
        script_path=file_path,
        tool_name=tool.name,
        tool_description=tool.description,
        tool_params=tool.parameters,
    )
    tool.fn = wrapped  # Replace with wrapped version
    registry.register(tool)
    loaded_names.append(tool.name)
```

Each tool function is wrapped in `ExternalToolWrapper` before registration. The wrapper is responsible for spawning a subprocess when the tool is called.

## 4. The Registration Helpers

### 4.1 `register_tool` (Decorator)

```python
@register_tool(name="optional_name", description="Optional description", parameters={})
async def my_tool(...) -> AsyncIterator[Any]:
    ...
```

If `name` is omitted, the function’s `__name__` is used. If `description` is omitted, the function’s docstring is used. The decorated function is returned unchanged so the user can keep using it as a normal function if desired.

### 4.2 `register` (Imperative)

```python
async def my_tool(...) -> AsyncIterator[Any]:
    ...

register("my_tool", "Does something", {}, my_tool)
```

This is useful when the user wants to register a function that was defined elsewhere or when they prefer a non-decorator style.

Both helpers create a `StreamingTool` dataclass-like object:

```python
StreamingTool(
    name=...,
    description=...,
    parameters=...,
    fn=...,          # the actual async generator / async iterator function
)
```

## 5. Example User Script

A valid external tool script looks like any ordinary Python file:

```python
# my_custom_tools.py
# Note: no imports required for registration helpers

@register_tool(name="echo", description="Echoes the input back")
async def echo(text: str):
    yield text


async def compute(x: int, y: int):
    yield x + y


register("add", "Adds two integers", {"x": {"type": "integer"}, "y": {"type": "integer"}}, compute)
```

When `load_tools_from_file("my_custom_tools.py")` is called:

1. The harness builds the two registration closures.
2. It executes `my_custom_tools.py` inside the current interpreter.
3. `echo` and `compute` are registered into `captured_tools`.
4. Both tools are added to the global `ToolRegistry`.

## 6. Directory Loading

`load_tools_from_directory` simply iterates over files in sorted order and delegates to `load_tools_from_file`:

```python
for script_file in sorted(dir_path.glob(pattern)):
    loaded_names.extend(load_tools_from_file(script_file))
```

Because each file is loaded independently, scripts in the same directory do not share a namespace, and each gets its own `sys.path` injection.

## 7. Guarantees and Limitations

### Guarantees

- **Shebang-driven interpreter**: External tools run with the Python interpreter specified by the script's shebang line (e.g. `#!/usr/bin/env python3`). If no shebang is present, `sys.executable` is used.
- **Clean `PATH`**: When the harness runs inside a virtual environment, the subprocess's `PATH` is cleaned of the venv's `bin` directory so `env(1)` resolves to the system Python rather than the venv Python.
- **Transient `sys.path`**: The script directory is only on `sys.path` during load; it is never permanently added.
- **No stale modules**: `sys.modules` is cleaned up after each load.

### Shebang on Windows

The shebang line is parsed literally and passed directly to `asyncio.create_subprocess_exec`. Because Windows does not have `/usr/bin/env`, using a Unix-style shebang like `#!/usr/bin/env python3` will raise `FileNotFoundError` on Windows.

Windows users should use one of these alternatives instead:

| Shebang                                                            | Behavior on Windows                                                                                                                             |
| ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `#!py -3.9`                                                        | Uses the Windows Python Launcher (`py.exe`) to start Python 3.9. This is the recommended approach if the launcher is installed.                 |
| `#!python3.9`                                                      | Works only if `python3.9.exe` is on the system `PATH`. Often unreliable on Windows because the executable is usually named `python.exe`.        |
| `#!C:\Users\You\AppData\Local\Programs\Python\Python39\python.exe` | Absolute path. Most reliable if you know the exact install location, but not portable across machines.                                          |
| *(no shebang)*                                                     | Falls back to `sys.executable` — the same Python interpreter running the TUI. Use this if the TUI is already running under the Python you want. |

### Limitations / Considerations

- **Subprocess isolation**: External tools run in an isolated subprocess. They cannot share memory or global state with the harness. However, they can access any Python packages installed in the interpreter environment they run under.
- **Name collisions**: if two scripts register tools with the same name, the second one will overwrite the first in `ToolRegistry` (depending on the registry implementation).

## 8. Automatic Reloading in the TUI

The built-in TUI client (`tui.py`) automatically reloads external tools **before every agent run**. This means:

- You can edit your tool scripts on disk while the TUI is running.
- As soon as you send your next message, the harness re-executes the script files, picks up any new or modified tools, and refreshes the available tool map.
- Your current tool selection is preserved: tools you had enabled stay enabled as long as they still exist after the reload.

The reload is implemented by calling `load_external_tools()` inside `_run_agent()` right before the conversation starts. Because the registry overwrites existing tools by name, updated definitions take effect immediately without restarting the application.

## 9. Developing and Debugging Tools Without the TUI

You can develop and test your tool scripts standalone—without running the TUI. This lets you verify your tool works correctly before integrating it with the harness.

### 9.1 Understanding What the Harness Provides

When the TUI loads your script, it injects two functions into your script's namespace:

- `register_tool(...)` — a decorator that registers your async generator as a tool
- `register(...)` — an imperative function to register a tool

For standalone development, you need to provide your own minimal implementations of these, then call your tool function directly.

### 9.2 Minimal Standalone Test Script

Copy your tool function(s) into a test file and add a small driver:

```python
#!/usr/bin/env python3
"""
Standalone test driver for your tool.
Run this file directly to test your tool without the TUI.
"""
import asyncio
import json
import sys

def register_tool(name=None, description=None, parameters=None):
    def decorator(fn):
        return fn
    return decorator

def register(name, description, parameters, fn):
    pass  # No-op for standalone testing

async def your_tool(arg1: str, arg2: int):
    """Your tool implementation."""
    yield {"status": "progress", "message": f"Processing {arg1}..."}
    await asyncio.sleep(0.1)
    yield {"success": True, "result": f"Done: {arg1} x {arg2}"}

if __name__ == "__main__":
    async def main():
        tool = your_tool(arg1="hello", arg2=42)
        async for chunk in tool:
            print(json.dumps(chunk, default=str))
            if "success" in chunk or "error" in chunk:
                break

    asyncio.run(main())
```

Run it with:

```bash
python your_tool_test.py
```

Expected output:

```json
{"status": "progress", "message": "Processing hello..."}
{"success": true, "result": "Done: hello x 42"}
```

### 9.3 Testing Tool Scripts That Use `@register_tool`

If your script relies on the decorator, create a test wrapper that injects the mock helpers and then imports your tool:

```python
#!/usr/bin/env python3
"""
test_my_tools.py — Test your tool scripts standalone.

Usage:
  python test_my_tools.py                  # Run all tools with sample args
  python test_my_tools.py calculator       # Run specific tool
  python test_my_tools.py calculator '{"expression": "2+2"}'
"""
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

TOOLS_FILE = Path(__file__).parent / "my_tools.py"

def register_tool(name=None, description=None, parameters=None):
    captured = {}
    def decorator(fn):
        captured["fn"] = fn
        captured["name"] = name or fn.__name__
        return fn
    decorator._captured = captured
    return decorator

def register(name, description, parameters, fn):
    pass

def load_and_get_tool(path: Path, tool_name: str):
    spec = importlib.util.spec_from_file_location("user_tool", path)
    module = importlib.util.module_from_spec(spec)
    ns = {"register_tool": register_tool, "register": register}
    spec.loader.exec_module(module, ns)
    return getattr(module, tool_name)

async def run_tool(tool_name: str, args_json: str | None = None):
    tool_fn = load_and_get_tool(TOOLS_FILE, tool_name)
    args = json.loads(args_json) if args_json else {"expression": "2 + 2"}
    gen = tool_fn(**args)
    async for chunk in gen:
        print(json.dumps(chunk, default=str))
        if "success" in chunk or "error" in chunk:
            break

if __name__ == "__main__":
    tool_name = sys.argv[1] if len(sys.argv) > 1 else "calculator"
    args = sys.argv[2] if len(sys.argv) > 2 else None
    asyncio.run(run_tool(tool_name, args))
```

### 9.4 Debugging Tips

**Shebang issues**: If your tool works in the TUI but fails standalone, check that your shebang line matches the Python environment you're using. Run `which python` and `python --version` to verify.

**Async generator must yield**: Your tool function is an async generator—it must `yield` at least one chunk. If you `return` directly, the harness will receive no output.

**JSON output format**: Each yielded dict must be JSON-serializable. Test by running:

```python
import json
chunk = {"success": True, "result": 42}
json.dumps(chunk)  # Must not raise
```

**Subprocess vs in-process**: When the TUI calls your tool, it runs in a subprocess. For debugging, you can add `print()` statements—they'll appear in the TUI logs or stderr. For cleaner output, use `yield` to emit structured debug messages.

**Windows shebang**: If you develop on Windows but deploy to Unix, remember that `#!/usr/bin/env python3` will fail on Windows. Use `#!py -3.9` or omit the shebang to fall back to the TUI's Python.

### 9.5 Verifying Tool Compatibility

Before using a tool in the TUI, verify:

1. **Syntax**: `python -m py_compile your_tool.py` — no errors
2. **Import**: `python -c "import your_tool"` — no import errors
3. **Execution**: Run your standalone test — produces valid JSON chunks
4. **Shebang**: If using a specific interpreter, `python3 your_tool.py` uses the right Python

If all four pass, your tool will work in the TUI.

## 10. Summary

`external_loader.py` implements a lightweight plugin system that loads external Python scripts and registers their tools. During the **loading phase**, `runpy.run_path` executes the script in the harness's process to capture tool definitions. During the **execution phase**, each tool call spawns a subprocess using the script's shebang-detected interpreter, ensuring the tool runs in the user's Python environment with access to their installed packages.

The subprocess receives a clean `PATH` (with any virtual environment `bin` directory removed) so that `#!/usr/bin/env python3` resolves to the system Python rather than the harness's venv Python. This solves the common problem where tools like `python-pptx` are installed in the system Python but the TUI runs under a different interpreter.
