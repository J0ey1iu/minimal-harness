import asyncio
import os
from typing import cast

import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk

from minimal_harness import Tool
from minimal_harness.agent import OpenAIAgent
from minimal_harness.llm.openai import OpenAILLMProvider
from minimal_harness.memory import (
    ConversationMemory,
    ExtendedInputContentPart,
    TextContentPart,
)


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

    async def on_chunk(chunk: ChatCompletionChunk | None, is_done: bool):
        if is_done:
            print()
            return
        if not chunk:
            raise
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            print(delta.content, end="", flush=True)

    async def on_tool_start(tool_call, data):
        print(f"[Tool Start] {tool_call}")

    async def on_tool_end(tool_call, result):
        print(f"[Tool End] {tool_call} -> {result}")

    async def on_execution_start(tool_calls):
        print(f"[Execution Start] {len(tool_calls)} tool(s) to execute")
        for tc in tool_calls:
            print(f"  - {tc['function']['name']}")

    client = AsyncOpenAI(
        api_key=os.getenv("AIHUBMIX_API_KEY"),
        base_url="https://aihubmix.com/v1",
    )
    llm_provider = OpenAILLMProvider(
        client=client, model="qwen3.5-27b", on_chunk=on_chunk
    )
    memory = ConversationMemory(
        system_prompt="You are an assistant that can check weather and do calculations."
    )
    agent = OpenAIAgent(
        llm_provider=llm_provider,
        tools=tools,
        memory=memory,
    )

    print("=== Round 1 ===")
    await agent.run(
        user_input=cast(
            list[ExtendedInputContentPart],
            [
                {
                    "type": "text",
                    "text": "What's the weather like in Beijing today? Also help me calculate (3 + 5) * 12",
                }
            ],
        ),
        on_tool_start=on_tool_start,
        on_tool_end=on_tool_end,
        on_execution_start=on_execution_start,
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
        on_tool_start=on_tool_start,
        on_tool_end=on_tool_end,
        on_execution_start=on_execution_start,
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
        on_tool_start=on_tool_start,
        on_tool_end=on_tool_end,
        on_execution_start=on_execution_start,
    )
