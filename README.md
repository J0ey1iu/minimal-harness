# minimal-harness

**Documentation: [/docs](./docs/)**

A lightweight Python agent harness for building LLM-powered agents with tool-calling support.

Latest version: **0.4.4**

## What This Project Is For

Minimal-harness is a lean framework for building agents that can call tools. It provides:

- **OpenAI-compatible API** - Works with any OpenAI-compatible API provider
- **Tool system** - Create tools via decorators; includes built-in tools (bash, file ops)
- **AsyncIterator events** - Real-time async iteration for chunks, tool start/end, execution events
- **Conversation memory** - Tracks token usage across interactions
- **ESC stop support** - Gracefully stop LLM streaming and tool execution

## Architecture

The framework uses an **event-driven architecture** with AsyncIterator-based event handling:

```
Agent (OpenAIAgent) → Internal Events → to_client_event() → Client-Facing Events
```

**Event flow:**

```python
async for event in agent.run(user_input=[{"type": "text", "text": "..."}]):
    if isinstance(event, LLMChunk):
        # handle chunk
    elif isinstance(event, ToolEnd):
        # handle tool result
```

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
import os
from openai import AsyncOpenAI
from minimal_harness.agent.openai import OpenAIAgent
from minimal_harness.client.events import (
    AgentStartEvent,
    AgentEndEvent,
    LLMChunkEvent,
    ToolStartEvent,
    ToolEndEvent,
)
from minimal_harness.llm.openai import OpenAILLMProvider
from minimal_harness.memory import ConversationMemory
from minimal_harness.tool.built_in.bash import get_tools as get_bash_tools

def main():
    parser = argparse.ArgumentParser(description="My AI agent")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default="qwen3.5-27b")
    args = parser.parse_args()

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    llm_provider = OpenAILLMProvider(client=client, model=args.model)
    memory = ConversationMemory(system_prompt="You are a helpful assistant.")
    agent = OpenAIAgent(
        llm_provider=llm_provider,
        tools=list(get_bash_tools().values()),
        memory=memory,
    )

    async def run():
        stop_event = asyncio.Event()
        async for event in agent.run(
            user_input=[{"type": "text", "text": "What files are in the current directory?"}],
            stop_event=stop_event,
        ):
            client_event = event.to_client_event()
            if isinstance(client_event, AgentStartEvent):
                print(f"Agent starting...")
            elif isinstance(client_event, LLMChunkEvent):
                chunk = client_event.chunk
                if chunk and chunk.choices:
                    content = chunk.choices[0].delta.content or ""
                    print(content, end="", flush=True)
            elif isinstance(client_event, ToolStartEvent):
                print(f"\n[Calling tool: {client_event.tool_call['function']['name']}]")
            elif isinstance(client_event, ToolEndEvent):
                print(f"\n[Tool result: {client_event.result[:100]}...]")
            elif isinstance(client_event, AgentEndEvent):
                break

    import asyncio
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

### Built-in Tools

| Tool                   | Description                                           |
| ---------------------- | ----------------------------------------------------- |
| `bash`                 | Execute shell commands with timeout                   |
| `local_file_operation` | Read, write, patch, or delete files (4 universal modes) |

### Event Types

| Event                 | Description                     |
| --------------------- | ------------------------------- |
| `AgentStartEvent`     | Agent execution started         |
| `AgentEndEvent`       | Agent execution completed       |
| `LLMStartEvent`       | LLM generation started          |
| `LLMChunkEvent`       | LLM output chunk received       |
| `LLMEndEvent`         | LLM generation completed        |
| `ExecutionStartEvent` | Tool execution started          |
| `ExecutionEndEvent`   | Tool execution completed        |
| `ToolStartEvent`      | Tool call started               |
| `ToolProgressEvent`   | Tool intermediate progress      |
| `ToolEndEvent`        | Tool call completed with result |

### Environment Variables

| Variable      | Description                       |
| ------------- | --------------------------------- |
| `MH_BASE_URL` | API base URL                      |
| `MH_API_KEY`  | API key                           |
| `MH_MODEL`    | Model name (default: qwen3.5-27b) |

### Stop Mechanism

Press **ESC** during execution to gracefully stop LLM streaming and tool execution.
