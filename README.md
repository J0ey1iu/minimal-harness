# minimal-harness

A lightweight Python agent harness for building LLM-powered agents with tool-calling support. An exploration of making an agent SDK as lean as possible while being effective.

## Features

- Simple `Agent` class for building LLM-powered agents
- Tool-calling support with concurrent execution
- Streaming response support via chunk callbacks
- Conversation history management with `Memory` interface
- Built-in tools: `glob` (file pattern matching) and `grep` (content search)
- Multiple LLM backends: OpenAI-compatible and LiteLLM
- Extensible LLM provider interface

## Installation

```bash
pip install -e .                    # Basic install
pip install -e ".[test]"            # With test dependencies
pip install -e ".[demo]"            # With demo dependencies
pip install -e ".[dev]"             # All dev dependencies
```

## Quick Start

```python
import asyncio
from minimal_harness import Agent, Tool, OpenAILLMProvider
from openai import AsyncOpenAI

async def get_weather(city: str) -> dict:
    return {"city": city, "temperature": "22°C", "condition": "Sunny"}

tools = [
    Tool(
        name="get_weather",
        description="Get weather for a specified city",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
        fn=get_weather,
    ),
]

client = AsyncOpenAI(api_key="your-api-key", base_url="https://aihubmix.com/v1")

async def on_chunk(chunk, is_done):
    if is_done:
        print()
        return
    delta = chunk.choices[0].delta if chunk.choices else None
    if delta and delta.content:
        print(delta.content, end="", flush=True)

llm_provider = OpenAILLMProvider(client=client, model="qwen3.5-27b", on_chunk=on_chunk)
agent = Agent(llm_provider=llm_provider, tools=tools)

result = await agent.run("What's the weather in Beijing?")
print(result)
```

## Demo

Run an interactive TUI demo:

```bash
python demo/cli.py
```

## Agent

The `Agent` class manages conversation context and tool execution.

### Constructor

```python
Agent(
    llm_provider: LLMProvider,
    tools: list[Tool] | None = None,
    max_iterations: int = 10,
    memory: Memory | None = None,
    tool_executor: ToolExecutor | None = None,
)
```

### Methods

- `run(user_input: str, on_chunk: ChunkCallback | None = None) -> str` - Run the agent with user input

## LLMProvider

The `LLMProvider` is a protocol that defines the interface for LLM backends.

### OpenAILLMProvider

```python
OpenAILLMProvider(client: AsyncOpenAI, model: str = "qwen3.5-27b")
```

### LiteLLMProvider

```python
LiteLLMProvider(model: str = "qwen3.5-27b", **kwargs)
```

## Memory

Memory classes manage conversation history.

### ConversationMemory

```python
ConversationMemory(system_prompt: str = "You are a helpful assistant.")
```

## Tool

Define tools that the agent can call.

```python
Tool(
    name: str,
    description: str,
    parameters: dict,  # OpenAI function parameters schema
    fn: Callable[..., Awaitable[Any]],  # Async function implementation
)
```

## ToolExecutor

Executes tool calls concurrently and returns results as messages.

## Built-in Tools

### glob

File pattern matching tool.

```python
from minimal_harness.tool import glob

tool = glob()
```

### grep

Content search tool.

```python
from minimal_harness.tool import grep

tool = grep()
```

## Testing

```bash
pip install -e ".[test]"
pytest
```
