# Writing Custom Tools

> **0.7.0 调整**：本文的"Quick Start"操作流程（`Ctrl+O` 设置 tools path、
> `Ctrl+T` 启用工具）依赖 TUI，已迁出至 [`mh-tui`](https://github.com/J0ey1iu/mh-tui)。
> TUI 侧的端到端使用步骤见 mh-tui 文档；
> **本文保留 SDK 侧的工具编写语法与外部脚本加载机制**——这部分与
> 任何应用层（包括 mh-tui、orchestration、自建服务）通用。

Minimal Harness lets you extend any application built on the SDK (TUI agent, orchestration service, custom app) with your own tools. You write a plain Python script — **no framework installation required**. The harness injects `register_tool` and `register` into your script's namespace at load time.

## Quick Start

1. Create a directory for your tools (e.g. `~/.minimal_harness/tools/`)
2. Write one or more `.py` files using the patterns below
3. Open the TUI, press **Ctrl+O**, and set **Tools Path** to your directory
4. Press **Ctrl+T** to enable your tools

Alternatively, edit `~/.minimal_harness/config.json` directly:

```json
{
  "tools_path": "~/.minimal_harness/tools"
}
```

`tools_path` can point to:

- A **directory** — all `*.py` files are loaded (alphabetically)
- A **single `.py` file** — only that file is loaded

## Tool Interface

A tool function must be an **async generator** (`async def ... -> AsyncIterator`) that yields dictionaries.

Each yielded dict is sent to the LLM as a progress event. The **last yielded dict** becomes the final result for that tool call.

### Basic skeleton

```python
from typing import AsyncIterator

async def my_tool(arg1: str) -> AsyncIterator[dict]:
    yield {"status": "progress", "message": "Working on it..."}
    # ... do your work ...
    yield {"success": True, "result": "done"}
```

## Registration Methods

### Method 1 — `@register_tool` decorator

Wraps an async generator. Optionally pass `name`, `description`, `parameters`.

```python
@register_tool(
    name="search_web",
    display_name="Search Web",
    description="Search the web for information",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
        },
        "required": ["query"],
    },
)
async def search_web(query: str) -> AsyncIterator[dict]:
    yield {"status": "progress", "message": f"Searching: {query}"}
    # ... actual search logic ...
    yield {"success": True, "results": ["result1", "result2"]}
```

If you omit `name`, the function name is used. If you omit `description`, the docstring is used. You can also set `display_name` to provide a human-readable label for the UI — if omitted, the tool's `name` is shown instead.

### Method 2 — `register()` function call

Useful when you want to define the function separately from registration.

```python
async def reverse_string(text: str) -> AsyncIterator[dict]:
    yield {"status": "progress", "message": f"Reversing '{text}'..."}
    yield {"success": True, "result": text[::-1]}

register(
    "reverse_string",
    "Reverse a given string",
    {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The string to reverse"},
        },
        "required": ["text"],
    },
    reverse_string,
    display_name="Reverse String",
)
```

## Parameters Schema

The `parameters` dict follows the **OpenAI function calling** format:

```python
parameters={
    "type": "object",
    "properties": {
        "city": {"type": "string", "description": "City name"},
        "units": {"type": "string", "description": "Temperature units", "enum": ["celsius", "fahrenheit"]},
    },
    "required": ["city"],
}
```

Supported property types: `string`, `integer`, `number`, `boolean`, `array`, `object`.

## ToolResult — Separating Content from Metadata

You can wrap your final result in a `ToolResult` object to separate LLM-facing content from UI-only metadata:

```python
from minimal_harness.types import ToolResult

async def show_profiles() -> AsyncIterator[Any]:
    yield {"status": "progress", "message": "Searching..."}
    yield ToolResult(
        content="Found 3 matching profiles: Alice, Bob, Charlie.",
        meta={
            "profiles": [{"name": "Alice", "role": "SDE"}, ...],
            "html": "<div class='profile-card'>...</div>",
        },
    )
```

- `content`: Goes into the LLM context (semantic payload)
- `meta`: UI/viz data only — never consumes LLM context window

### Stopping the Agent Loop

Set `stop=True` on a `ToolResult` to prevent the agent from continuing after this tool execution:

```python
async def place_order(order_id: str) -> AsyncIterator[Any]:
    yield {"status": "progress", "message": f"Placing order {order_id}..."}
    yield ToolResult(
        content=f"Order {order_id} confirmed. No further action needed.",
        stop=True,
    )
```

When `stop=True`, the agent loop breaks immediately after the current tool batch completes. The tool's `content` becomes the final response (`AgentEnd.response`). This is useful for tools that represent terminal actions (e.g., booking confirmed, payment processed) where the LLM should not continue generating.

## Error Handling

Raise exceptions or yield error dicts — the framework catches both:

```python
async def risky_tool(url: str) -> AsyncIterator[dict]:
    try:
        # ... potentially failing operation ...
        yield {"success": True, "data": result}
    except Exception as e:
        yield {"success": False, "error": str(e)}
```

Unhandled exceptions are caught by the framework and reported as errors automatically.

## Multiple Files

When pointing `tools_path` at a directory, every `.py` file is loaded. Files are loaded in alphabetical order. If two files register a tool with the same name, the last one wins.

## Full Example

See `examples/user_tool_example.py` for a complete, working example with both registration methods.

## Language Detection

Your tool can detect the user's current UI language by calling `get_current_locale()`. This lets you produce localized output naturally:

```python
from typing import AsyncIterator
from minimal_harness.agent.runtime import get_current_locale

@register_tool(
    name="greet",
    display_name="Greet",
    description="Greet the user in their language",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "User's name"},
        },
        "required": ["name"],
    },
)
async def greet(name: str) -> AsyncIterator[dict]:
    locale = get_current_locale()
    if locale == "zh":
        greeting = f"你好, {name}！"
    elif locale == "en":
        greeting = f"Hello, {name}!"
    else:
        greeting = f"Hi, {name}!"
    yield {"greeting": greeting}
```

The locale is injected by the runtime from the `Accept-Language` header (or equivalent). It propagates automatically to sub-tasks (e.g. handoff), so child agents and their tools see the same locale.

> **Note**: External tool scripts (loaded via `tools_path`) run in a subprocess and currently do **not** have access to `get_current_locale()`. Only in-process tools can use this function.

## Tips

- **Shebang determines interpreter**: The first line of your script (e.g. `#!/usr/bin/env python3`) controls which Python interpreter your tools use. This is important when your tools need packages installed in a different Python environment than the TUI.
  - **Windows users**: `#!/usr/bin/env python3` will fail because Windows has no `/usr/bin/env`. Use `#!py -3` (recommended, requires the Python Launcher), `#!python3`, an absolute path like `#!C:/Users/You/.../python.exe`, or omit the shebang to use the TUI's Python.   See `docs/external-scripts-loading.md` for details.
- **You can import your own packages.** Your script runs in your Python environment with your installed packages.
- **Async generators are required.** Use `async def` + `yield`. Regular `return` functions won't work.
- **Yield dicts, not strings.** Each `yield` should produce a dictionary.
- **Progress events are optional.** You can yield just a final result: `yield {"success": True}`.
- **`await` is available.** Use `await asyncio.sleep(...)`, `aiohttp`, etc. inside your tools.
