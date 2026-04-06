# minimal-harness

A lightweight Python agent harness with tool-calling support.

## Features

- Simple `Agent` class for building LLM-powered agents
- Tool-calling support with concurrent execution
- Streaming response support via chunk callbacks
- Conversation history management
- Built on OpenAI's API (supports any OpenAI-compatible endpoint)

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import asyncio
from minimal_harness import Agent, Tool
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
agent = Agent(model="minimax-m2.7", tools=tools, client=client)

async def on_chunk(text: str, is_done: bool):
    if not is_done:
        print(text, end="", flush=True)
    else:
        print()

result = await agent.run("What's the weather in Beijing?", on_chunk=on_chunk)
print(result)
```

## Agent

The `Agent` class manages conversation context and tool execution.

### Constructor

```python
Agent(
    model: str = "minimax-m2.1",
    system_prompt: str = "You are a helpful assistant.",
    tools: list[Tool] | None = None,
    max_iterations: int = 10,
    client: AsyncOpenAI | None = None,
)
```

### Methods

- `run(user_input: str, on_chunk: ChunkCallback | None = None) -> str` - Run the agent with user input
- `reset()` - Clear conversation history (keeps system prompt)

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

## Testing

```bash
pip install -e ".[test]"
pytest
```
