# minimal-harness

**Documentation: [/docs](./docs/)**

A lightweight Python agent harness for building LLM-powered agents with tool-calling support.

Latest version: **0.5.4**

## What This Project Is For

Minimal-harness is a lean framework for building agents that can call tools. It provides:

- **OpenAI/Anthropic-compatible API** - Works with OpenAI, Anthropic, or any OpenAI-compatible API provider
- **Multi-modal image input** - Pass image URLs or base64 data to LLM providers supporting vision
- **Tool system** - Create tools via decorators; includes built-in tools (bash, file ops)
- **Middleware hooks** - Observe and intercept the agent lifecycle (agent start/end, LLM calls, tool execution, tool policy enforcement)
- **AsyncIterator events** - Real-time async iteration for chunks, tool start/end, execution events
- **Conversation memory** - Tracks token usage across interactions, auto-persists to disk
- **ESC stop support** - Gracefully stop LLM streaming and tool execution

## Architecture

The framework uses an **event-driven architecture** with AsyncIterator-based event handling:

```
Agent (SimpleAgent) → AgentEvent (from types.py)
```

**Event flow:**

```python
async for event in agent.run(
    user_input=[{"type": "text", "text": "..."}],
    memory=memory,
    tools=tools,
):
    if isinstance(event, LLMChunk):
        # handle chunk
    elif isinstance(event, ToolEnd):
        # handle tool result
```

All event types are defined in `src/minimal_harness/types.py`. No separate client event layer exists.

## How to Build an App

### Project Structure

A typical app looks like this:

```
my-app/
├── cli.py          # Entry point
└── tools.py        # Your custom tools
```

### 1. Create Your Entry Point

```python
import argparse
import asyncio
from openai import AsyncOpenAI

from minimal_harness.agent.simple import SimpleAgent
from minimal_harness.llm.openai import OpenAILLMProvider
from minimal_harness.memory import ConversationMemory
from minimal_harness.tool.built_in.bash import get_tools as get_bash_tools
from minimal_harness.types import (
    AgentStart,
    AgentEnd,
    LLMChunk,
    ToolStart,
    ToolEnd,
)

def main():
    parser = argparse.ArgumentParser(description="My AI agent")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default="qwen3.5-27b")
    args = parser.parse_args()

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    llm_provider = OpenAILLMProvider(client=client, model=args.model)
    agent = SimpleAgent(llm_provider=llm_provider, max_iterations=50)
    memory = ConversationMemory()
    tools = list(get_bash_tools().values())

    async def run():
        stop_event = asyncio.Event()
        context = {"user_id": "abc123"}  # passed to middleware hooks
        async for event in agent.run(
            user_input=[{"type": "text", "text": "What files are in the current directory?"}],
            stop_event=stop_event,
            memory=memory,
            tools=tools,
            context=context,
        ):
            if isinstance(event, AgentStart):
                print("Agent starting...")
            elif isinstance(event, LLMChunk):
                delta = event.chunk
                if delta and delta.content:
                    print(delta.content, end="", flush=True)
            elif isinstance(event, ToolStart):
                print(f"\n[Calling tool: {event.tool_call['function']['name']}]")
            elif isinstance(event, ToolEnd):
                print(f"\n[Tool result: {str(event.result)[:100]}...]")
            elif isinstance(event, AgentEnd):
                print(f"\n[Done in {event.time_taken:.2f}s]")
                break

    asyncio.run(run())

if __name__ == "__main__":
    main()
```

### 2. Add Custom Tools

Use the `@register_tool` decorator to add your own tools. You need a `ToolRegistry` instance:

```python
from typing import AsyncIterator

from minimal_harness.tool.registration import register_tool
from minimal_harness.tool.registry import ToolRegistry

registry = ToolRegistry()

@register_tool(
    name="get_weather",
    description="Get weather for a location",
    parameters={
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
    registry=registry,
)
async def get_weather(location: str) -> AsyncIterator[dict]:
    yield {"success": True, "result": f"The weather in {location} is sunny."}
```

The decorator registers the tool with the provided registry. Pass the same registry to the harness when running.

**Localized tool output**: Tools can detect the user's language at runtime via `get_current_locale()`:

```python
from minimal_harness.agent.runtime import get_current_locale

async def my_tool() -> AsyncIterator[dict]:
    locale = get_current_locale()
    yield {"message": "你好" if locale == "zh" else "Hello"}
```

### 3. Run

```bash
python cli.py --base-url https://api.openai.com/v1 --api-key sk-... --model gpt-4o
```

Or set environment variables:

```bash
export MH_BASE_URL=https://api.openai.com/v1
export MH_API_KEY=sk-...
export MH_MODEL=gpt-4o
python cli.py
```

### Middleware Hooks

Subclass `Middleware` to observe or intercept the agent lifecycle:

```python
from minimal_harness.agent.middleware import Middleware
from minimal_harness.types import LLMEnd, ToolCall

class PolicyEnforcer(Middleware):
    async def should_allow_tool(
        self, tool_call: ToolCall, **kwargs
    ) -> bool | str:
        # Return False or a reason string to deny the tool
        if tool_call["function"]["name"] == "bash":
            return "bash is not permitted in this context"
        return True

    async def on_llm_end(self, event: LLMEnd) -> None:
        if event.usage:
            print(f"Tokens: {event.usage['total_tokens']}")
```

Pass middleware to `SimpleAgent`:

```python
agent = SimpleAgent(
    llm_provider=llm_provider,
    middleware=[PolicyEnforcer()],
    max_iterations=50,
)
```

### Multi-modal Image Input

Pass image URLs or base64-encoded image data as input content parts:

```python
user_input = [
    {"type": "text", "text": "What's in this image?"},
    {
        "type": "image",
        "image_url": {"url": "https://example.com/photo.jpg"},
    },
]
```

For local images, encode as base64:

```python
import base64

with open("photo.jpg", "rb") as f:
    data = base64.b64encode(f.read()).decode()

user_input = [
    {"type": "text", "text": "Describe this image"},
    {
        "type": "image",
        "data": data,
        "media_type": "image/jpeg",
    },
]
```

### Built-in Tools

| Tool                   | Description                                           |
| ---------------------- | ----------------------------------------------------- |
| `bash`                 | Execute shell commands with timeout and workdir support |
| `local_file_operation` | Read, write, patch, or delete files (4 universal modes) |

### Event Types

All events are defined in `minimal_harness.types` and consumed as a single `AgentEvent` union:

| Event             | Fields                                                 | Description                     |
| ----------------- | ------------------------------------------------------ | ------------------------------- |
| `AgentStart`      | `user_input`, `timestamp`                              | Agent execution started         |
| `AgentEnd`        | `response`, `time_taken`, `exceeded`                   | Agent execution completed       |
| `LLMStart`        | `messages`, `tools`                                    | LLM generation started          |
| `LLMChunk`        | `chunk: LLMChunkDelta \| None`                         | LLM output chunk received       |
| `LLMEnd`          | `content`, `reasoning_content`, `tool_calls`, `usage`  | LLM generation completed        |
| `ExecutionStart`  | `tool_calls`                                           | Tool execution started          |
| `ExecutionEnd`    | `results`                                              | Tool execution completed        |
| `ToolStart`       | `tool_call`                                            | Tool call started               |
| `ToolProgress`    | `tool_call`, `chunk`                                   | Tool intermediate progress      |
| `ToolEnd`         | `tool_call`, `result`                                  | Tool call completed with result |
| `MemoryUpdate`    | `usage`                                                | Memory token usage updated      |

`LLMChunkDelta` contains `content`, `reasoning`, and `tool_calls` fields for provider-agnostic partial deltas.

### Environment Variables

| Variable             | Description                                 |
| -------------------- | ------------------------------------------- |
| `MH_BASE_URL`        | API base URL                                |
| `MH_API_KEY`         | API key                                     |
| `MH_MODEL`           | Model name (default: qwen3.5-27b)           |
| `MH_MAX_ITERATIONS`  | Max agent loop iterations (default: 50)     |
| `MH_THEME`           | TUI theme name (default: tokyo-night)       |

### Stop Mechanism

Press **ESC** during execution to gracefully stop LLM streaming and tool execution.
