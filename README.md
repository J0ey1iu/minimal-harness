# minimal-harness

A lightweight Python agent harness with tool-calling support.

## Features

- Simple `Agent` class for building LLM-powered agents
- Tool-calling support with concurrent execution
- Streaming response support via chunk callbacks
- Conversation history management with `Memory` interface
- Built on OpenAI's API (supports any OpenAI-compatible endpoint)
- Extensible LLM provider interface

## Installation

```bash
pip install -e .
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
llm_provider = OpenAILLMProvider(client=client, model="minimax-m2.7")
agent = Agent(llm_provider=llm_provider, tools=tools)

async def on_chunk(chunk, is_done):
    if is_done:
        print()
        return
    delta = chunk.choices[0].delta if chunk.choices else None
    if delta and delta.content:
        print(delta.content, end="", flush=True)

result = await agent.run("What's the weather in Beijing?", on_chunk=on_chunk)
print(result)
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

The `LLMProvider` is a protocol that defines the interface for LLM backends. The library includes `OpenAILLMProvider` for OpenAI-compatible endpoints.

### OpenAILLMProvider

```python
OpenAILLMProvider(client: AsyncOpenAI, model: str = "qwen3.5-27b")
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

## Testing

```bash
pip install -e ".[test]"
pytest
```
