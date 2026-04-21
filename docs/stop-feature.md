# Stop Feature Design

## Overview

The TUI client supports stopping LLM generation and tool execution mid-process by pressing the **Escape** key. This document describes the architecture and implementation.

## Architecture

### Components

1. **TUIApp** (`client/built_in/tui.py`) — Textual-based terminal UI application
2. **`key_escape()`** — Textual key handler invoked when ESC is pressed
3. **`stop_event: asyncio.Event`** — A shared flag checked throughout the async pipeline to gracefully halt operations
4. **`_run_agent()`** — Async worker that runs the agent and responds to stop events

### Flow

```md
User presses ESC
        ↓
TUIApp.key_escape() is called by Textual
        ↓
stop_event.set() is called
        ↓
stop_event.is_set() returns True across:
  - OpenAILLMProvider._chat() — breaks from OpenAI stream loop
  - StreamingTool.execute() — breaks from async for loop
  - OpenAIAgent.run() — breaks from iteration loops
        ↓
_run_agent() exits the async for event loop
        ↓
Prints "[Interrupted by user]"
```

## Implementation

### ESC Detection

Textual handles raw keyboard input and routes the Escape key to the `key_escape()` method:

```python
def key_escape(self) -> None:
    if self.is_streaming and self.stop_event is not None:
        self.stop_event.set()
        self._flush_streaming_to_pending()
        self._queue_message("\n  [Interrupted by user]", "bold red")
        self._refresh_display()
        self.is_streaming = False
        self.query_one("#streaming-label", Static).update("")
        self.query_one("#chat-input", Input).disabled = False
        self.query_one("#chat-input", Input).focus()
```

This works cross-platform because Textual abstracts terminal input handling.

### Stop Propagation

#### Agent Run Loop (`_run_agent` in `client/built_in/tui.py`)

```python
async for event in self.framework_client.run(
    user_input=[{"type": "text", "text": user_input}],
    stop_event=self.stop_event,
    memory=self.memory,
    tools=self.tools if self.tools else None,
):
    if self.stop_event.is_set():
        break
    self._handle_event(event)
```

When stopped, the loop breaks early. The `finally` block cleans up UI state.

#### LLM Streaming (`llm/openai.py`)

```python
async for chunk in stream:
    if stop_event and stop_event.is_set():
        break   # stops consuming OpenAI chunks
```

When stopped, the OpenAI HTTP stream is abandoned. Partial content is **not** captured, and token usage is **not** recorded for the interrupted turn.

#### Tool Execution

For `StreamingTool`:

```python
async for chunk in tool.fn(**args):
    if stop_event and stop_event.is_set():
        break
```

Tools check `stop_event.is_set()` at yield points and stop gracefully.

## User Experience

- Press **Escape** during LLM streaming → stops streaming, prints `[Interrupted by user]`
- Press **Escape** during tool execution → cancels current tool(s), returns to input loop
- The conversation memory retains any assistant content and tool results that were already yielded before the stop

## Cross-Platform

The stop feature works on all platforms supported by Textual (Unix, macOS, Windows) because Textual handles the terminal input abstraction. No platform-specific terminal mode switching is required.
