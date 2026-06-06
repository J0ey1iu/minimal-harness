# Stop Feature Design

## Overview

The TUI client supports stopping LLM generation and tool execution mid-process by pressing the **Escape** key. This document describes the architecture and implementation.

## Architecture

### Components

1. **TUIApp** (`client/built_in/app.py`) — Textual-based terminal UI application
2. **`Binding("escape", "interrupt")`** — Textual key binding that invokes `action_interrupt`
3. **`actions/interrupt.py`** — Action handler that sets the stop event
4. **`stop_event: asyncio.Event`** — A shared flag checked throughout the async pipeline to gracefully halt operations
5. **`SessionController.interrupt()`** — Coordinates interrupt across session and active runs

### Flow

```md
User presses ESC
        ↓
Textual routing triggers action_interrupt
        ↓
action_interrupt() in actions/interrupt.py:
  - checks if session is streaming
  - calls app._ctrl.interrupt()
        ↓
SessionController.interrupt():
  - calls current_session.interrupt() (sets session's stop_event)
  - calls stop_event.set() on active run
        ↓
stop_event.is_set() returns True across:
  - OpenAILLMProvider._chat() — breaks from OpenAI stream loop
  - AnthropicLLMProvider._chat() — breaks from Anthropic stream loop
  - StreamingTool.execute() — breaks from async for loop
  - SimpleAgent.run() — breaks from iteration loops
        ↓
SessionController.end_run() cleans up the run
        ↓
ChatDisplay shows "  ✗ interrupted"
```

## Implementation

### ESC Detection

Textual's binding system routes the Escape key via the app's BINDINGS:

```python
BINDINGS = [
    Binding("ctrl+o", "config", "Config"),
    Binding("ctrl+t", "tools", "Tools"),
    Binding("escape", "interrupt", "Interrupt", show=False),
    Binding("ctrl+c", "request_quit", "Quit"),
]
```

This invokes `action_interrupt()` which delegates to `actions/interrupt.py`:

```python
def action_interrupt(app: TUIApp) -> None:
    sid = app._ctrl.current_session_id
    if not sid or not app._ctrl.is_session_streaming(sid):
        return
    d = app._chat_display
    if d is None:
        return
    app._ctrl.interrupt()
    d.say("  \u2717 interrupted", "bold bright_red")
```

This works cross-platform because Textual abstracts terminal input handling.

### Stop Propagation

#### Agent Run Loop (`app.py` — `_run()` → `SessionController.start_run()`)

The agent runs as an async task. Events are drained by a periodic `_tick()` method via `drain_session_events()`. When the stop event is set, the agent's generator breaks early, the task finishes, and `drain_session_events()` returns `done=True`, which triggers cleanup.

#### LLM Streaming (`llm/openai.py`, `llm/anthropic.py`)

```python
async for chunk in stream:
    if stop_event and stop_event.is_set():
        break   # stops consuming stream chunks
```

When stopped, the stream is abandoned. Partial content is **not** captured, and token usage is **not** recorded for the interrupted turn.

#### Tool Execution

For `StreamingTool`:

```python
async for chunk in tool.fn(**args):
    if stop_event and stop_event.is_set():
        break
```

Tools check `stop_event.is_set()` at yield points and stop gracefully.

## User Experience

- Press **Escape** during LLM streaming → stops streaming, prints `✗ interrupted`
- Press **Escape** during tool execution → cancels current tool(s), returns to input loop
- The conversation memory retains any assistant content and tool results that were already yielded before the stop

## Cross-Platform

The stop feature works on all platforms supported by Textual (Unix, macOS, Windows) because Textual handles the terminal input abstraction. No platform-specific terminal mode switching is required.


## Programmatic Stop via ToolResult

In addition to the user-initiated ESC interrupt, tools can programmatically request the agent loop to stop after their execution completes. This is done by returning a `ToolResult` with `stop=True`.

### How it works

```python
from minimal_harness.types import ToolResult

async def my_tool(args):
    yield "Processing..."
    yield ToolResult(
        content="Order confirmed: ORD-12345. Agent loop will stop here.",
        stop=True,
    )
```

When `stop=True`:
1. `_execute_tools` detects the flag and sets `should_stop=True` on `ExecutionEnd`
2. The agent's main loop breaks immediately after the tool batch completes
3. The tool's `content` becomes `AgentEnd.response` — the final response to the user
4. The `stop` flag is persisted in the tool message for replay/audit

### Signal chain

```
Tool returns ToolResult(stop=True)
  → _execute_tools sets should_stop=True, captures response_text
  → ExecutionEnd(should_stop=True, response_text="...")
  → main loop breaks
  → AgentEnd(response="...")
```

### Wire protocol

For remote tools (executed via SSE), the tool endpoint should include `__stop: true` in its `tool_end` event:

```json
{"type": "tool_end", "data": {"content": "Done", "__stop": true}}
```

The harness `_unwrap_tool_result` deserializes `__stop` → `ToolResult.stop`.
