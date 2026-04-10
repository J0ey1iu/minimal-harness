import os
from typing import cast

import pytest
import asyncio
from minimal_harness import Tool
from minimal_harness.agent import OpenAIAgent
from minimal_harness.llm.openai import OpenAILLMProvider
from minimal_harness.memory import ContentPart, ConversationMemory, TextContentPart
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk


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

    client = AsyncOpenAI(
        api_key=os.getenv("AIHUBMIX_API_KEY"),
        base_url="https://aihubmix.com/v1",
    )
    llm_provider = OpenAILLMProvider(client=client, model="qwen3.5-27b")
    memory = ConversationMemory(
        system_prompt="You are an assistant that can check weather and do calculations."
    )
    agent = OpenAIAgent(
        llm_provider=llm_provider,
        tools=tools,
        memory=memory,
    )

    async def on_chunk(chunk: ChatCompletionChunk | None, is_done: bool):
        if is_done:
            print()
            return
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            print(delta.content, end="", flush=True)

    print("=== Round 1 ===")
    await agent.run(
        user_input=cast(
            list[ContentPart],
            [
                {
                    "type": "text",
                    "text": "What's the weather like in Beijing today? Also help me calculate (3 + 5) * 12",
                }
            ],
        ),
        on_chunk=on_chunk,
    )

    print("\n=== Round 2 (multi-turn context) ===")
    await agent.run(
        [
            cast(
                TextContentPart,
                {
                    "type": "text",
                    "text": "Is the weather in that city suitable for going outside?",
                },
            )
        ],
        on_chunk=on_chunk,
    )

    print("\n=== Round 3 (Long respones without tool calling) ===")
    await agent.run(
        [
            cast(
                TextContentPart,
                {
                    "type": "text",
                    "text": "What do you think about TV drama?",
                },
            )
        ],
        on_chunk=on_chunk,
    )
