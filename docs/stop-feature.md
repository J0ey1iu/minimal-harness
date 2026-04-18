# Stop Feature Design

## Overview

The SimpleCli supports stopping LLM generation and tool execution mid-process by pressing the ESC key. This document describes the architecture and implementation.

## Architecture

### Components

1. **ESC Reader** (`_esc_reader`) — A callback registered with `loop.add_reader()` that reads ESC bytes from stdin when the terminal is in cbreak mode
2. **Cbreak Mode** (`_enter_cbreak` / `_leave_cbreak`) — Functions that put the terminal into cbreak (character-by-character) mode so individual key presses are immediately available
3. **Stop Event** (`asyncio.Event`) — A shared flag checked throughout the async pipeline to gracefully halt operations

### Flow

```md
User presses ESC
        ↓
os.read() returns b"\x1b" in _esc_reader callback
        ↓
asyncio.Event.set() called on stop_event
        ↓
stop_event.is_set() returns True across:
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

`_enter_cbreak()` puts the terminal into cbreak mode during generation, and `_leave_cbreak()` restores it when done. This makes individual key presses immediately available to `os.read()`.

**Entering cbreak:** `tty.setcbreak(fd)` — called before streaming starts

**Restoring canonical:** `termios.tcsetattr(fd, TCSADRAIN, old_settings)` + `termios.tcflush(fd, TCIFLUSH)` — called after streaming ends

### ESC Detection Mechanism

ESC detection uses `loop.add_reader(sys.stdin.fileno(), _esc_reader, stop_event)`. When a key is pressed in cbreak mode, the asyncio event loop invokes `_esc_reader` which reads one byte via `os.read()`. If that byte is `\x1b` (ESC), it calls `stop_event.set()`.

This is **asyncio event-based, not thread-based** — the event loop drives the ESC reader callback directly.

### User Input Handling

`prompt_toolkit`'s `PromptSession.prompt_async()` requires controlling the terminal. When an interactive tool needs user input (`wait_for_user_input` callback):

1. The ESC reader is removed from the event loop (`loop.remove_reader`)
2. Cbreak mode is exited (restoring canonical mode for prompt_toolkit)
3. `session.prompt_async()` reads the user's answer
4. Cbreak mode is re-entered and the ESC reader is re-registered

## Stop Propagation

### LLM Streaming (`llm/openai.py`)

```python
async for chunk in stream:
    if stop_event and stop_event.is_set():
        break   ← stops consuming OpenAI chunks
# final LLMResponse is NOT yielded when broken
```

When stopped, the OpenAI HTTP stream is abandoned. No `LLMResponse` is yielded — the stream loop breaks early and the generator ends. This means partial content is **not** captured, and token usage is **not** recorded.

### Tool Execution (`tool_executor.py`)

For `StreamingTool`:

```python
async for chunk in tool.fn(**args):
    if stop_event and stop_event.is_set():
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

- Press ESC during LLM streaming → stops streaming, prints `[Stopped by user]`
- Press ESC during tool execution → cancels current tool(s), returns to input loop, prints `[Tool Execution Stopped]`
- `[Response stopped by user]` is added to conversation memory for context on subsequent turns

## Cross-Platform

- **Unix/macOS:** Uses `tty.setcbreak()` / `termios.tcgetattr(tcsetattr)` + `loop.add_reader()` for async ESC detection
- **Windows:** Not currently implemented — `is_tty` check at `cli.py:96` skips ESC detection on Windows; Ctrl+C remains available
