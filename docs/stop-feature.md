# Stop Feature Design

## Overview

The SimpleCli supports stopping LLM generation and tool execution mid-process by pressing the ESC key. This document describes the architecture and implementation.

## Architecture

### Components

1. **ESC Key Monitor** (`_monitor_esc_key`) — A background thread that detects ESC key presses
2. **Cbreak Mode** (`_CbreakMode`) — A context manager that puts the terminal into cbreak (character-by-character) mode so individual key presses are detectable
3. **Stop Event** (`asyncio.Event`) — A shared flag checked throughout the async pipeline to gracefully halt operations

### Flow

```md
User presses ESC
       ↓
_monitor_esc_key thread sets stop_event
       ↓
asyncio.Event.is_set() returns True across:
  - OpenAILLMProvider._chat() — breaks from OpenAI stream loop
  - ToolExecutor._execute_streaming() — raises CancelledError in async for loop
  - ToolExecutor._execute_interactive() — raises CancelledError at await points
  - OpenAIAgent.run() — breaks from iteration loops
       ↓
SimpleCli._run_async() exits agent.run()
       ↓
 Prints "[Stopped by user]"
```

## Terminal Mode Management

### Problem

In canonical (line-buffered) mode — the default — characters are not available to a program until Enter is pressed. This makes ESC detection impossible.

### Solution: Cbreak Mode

`_CbreakMode` puts the terminal into cbreak mode (character-by-character, no line buffering) during generation, and restores canonical mode when done.

**Entering cbreak:** `tty.setcbreak(fd)` — called in `_CbreakMode.__enter__`

**Restoring canonical:** `termios.tcsetattr(fd, TCSADRAIN, old_settings)` — called in `_CbreakMode.__exit__`

### User Input Handling

`input()` requires canonical mode (line-editing, echo). When an interactive tool needs user input (`wait_for_user_input` callback), `_CbreakMode.canonical_input()` temporarily restores canonical mode for the `input()` call, then re-enters cbreak mode.

The ESC monitoring thread is also paused (`esc_pause` event) during user input to avoid stdin contention.

## Stop Propagation

### LLM Streaming (`llm/openai.py`)

```python
async for chunk in stream:
    if stop_event.is_set():
        break   ← stops consuming OpenAI chunks
# final LLMResponse is NOT yielded when broken
```

When stopped, the OpenAI HTTP stream is abandoned. No `LLMResponse` is yielded — the stream loop breaks early and the generator ends. This means partial content is **not** captured, and token usage is **not** recorded.

### Tool Execution (`tool_executor.py`)

For `StreamingTool`:

```python
async for chunk in tool.fn(**args):
    if stop_event.is_set():
        raise CancelledError("Execution cancelled by user")
    if self._on_tool_progress:
        await self._on_tool_progress(chunk)
```

For `InteractiveTool`, `stop_event.is_set()` is checked before `execute_first`, before `wait_for_user_input`, and before `execute_final`.

### Agent Loop (`agent/openai.py`)

```python
async for _ in response:
    if stop_event and stop_event.is_set():
        break   ← abandons the stream, no LLMResponse

if stop_event and stop_event.is_set():
    self._memory.add_message({
        "role": "assistant",
        "content": "[Response stopped by user]",
        "tool_calls": None,
    })
    break
```

When stopped mid-stream, the agent writes `"[Response stopped by user]"` to conversation memory as the assistant's response, so the LLM sees the stop event on the next turn.

## User Experience

- Press ESC during LLM streaming → stops streaming, shows partial response, prints `[Stopped by user]`
- Press ESC during tool execution → cancels current tool(s), returns to input loop, prints `[Tool Execution Stopped]`
- `[Response stopped by user]` is added to conversation memory for context on subsequent turns

## Cross-Platform

- **Unix/macOS:** Uses `select.select()` on stdin with `tty.setcbreak()` / `termios.tcgetattr(tcsetattr)`
- **Windows:** Uses `msvcrt.kbhit()` / `msvcrt.getch()` — no terminal mode changes needed since Windows console handles this differently
