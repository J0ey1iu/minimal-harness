import os

import pytest
import asyncio
from minimal_harness import Tool, Agent
from openai import AsyncOpenAI


async def get_weather(city: str) -> dict:
    """Simulate weather query"""
    await asyncio.sleep(0.2)
    return {"city": city, "temperature": "22°C", "condition": "Sunny"}


async def calculator(expression: str) -> dict:
    """Simple calculator"""
    result = eval(expression, {"__builtins__": {}})
    return {"expression": expression, "result": result}


@pytest.mark.asyncio
async def test():
    tools = [
        Tool(
            name="get_weather",
            description="Get weather for a specified city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
            fn=get_weather,
        ),
        Tool(
            name="calculator",
            description="Calculate mathematical expression",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Valid Python mathematical expression",
                    },
                },
                "required": ["expression"],
            },
            fn=calculator,
        ),
    ]

    agent = Agent(
        model="minimax-m2.7",
        system_prompt="You are an assistant that can check weather and do calculations.",
        tools=tools,
        client=AsyncOpenAI(
            api_key=os.getenv("AIHUBMIX_API_KEY"),
            base_url="https://aihubmix.com/v1",
        ),
    )

    # Chunk-level callback: print in real-time
    async def on_chunk(text: str, is_done: bool):
        if not is_done:
            print(text, end="", flush=True)
        else:
            print()  # newline

    print("=== Round 1 ===")
    await agent.run(
        "What's the weather like in Beijing today? Also help me calculate (3 + 5) * 12",
        on_chunk=on_chunk,
    )

    print("\n=== Round 2 (multi-turn context) ===")
    await agent.run(
        "Is the weather in that city suitable for going outside?", on_chunk=on_chunk
    )

    print("\n=== Round 3 (Long respones without tool calling) ===")
    await agent.run("What do you think about TV drama?", on_chunk=on_chunk)
