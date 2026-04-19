import asyncio
import os
from typing import AsyncIterator, cast

import pytest
from openai import AsyncOpenAI

from minimal_harness import StreamingTool
from minimal_harness.agent import OpenAIAgent
from minimal_harness.llm.openai import OpenAILLMProvider
from minimal_harness.memory import (
    ConversationMemory,
    ExtendedInputContentPart,
    TextContentPart,
)
from minimal_harness.types import (
    AgentEnd,
    ExecutionStart,
    LLMChunk,
    ToolEnd,
    ToolStart,
)


async def get_weather(city: str) -> AsyncIterator[dict]:
    """Simulate weather query"""
    await asyncio.sleep(0.2)
    yield {"city": city, "temperature": "22°C", "condition": "Sunny"}


async def calculator(expression: str) -> AsyncIterator[dict]:
    """Simple calculator"""
    result = eval(expression, {"__builtins__": {}})
    yield {"expression": expression, "result": result}


@pytest.mark.asyncio
async def test():
    tools = [
        StreamingTool(
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
        StreamingTool(
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
        api_key=os.getenv("MH_API_KEY"),
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

    async def run_and_print(user_input):
        final_response = None
        async for event in agent.run(user_input=user_input):
            if isinstance(event, LLMChunk):
                if event.chunk and not event.is_done:
                    delta = (
                        event.chunk.choices[0].delta if event.chunk.choices else None
                    )
                    if delta and delta.content:
                        print(delta.content, end="", flush=True)
            elif isinstance(event, ToolStart):
                print(f"[Tool Start] {event.tool_call}")
            elif isinstance(event, ToolEnd):
                print(f"[Tool End] {event.tool_call} -> {event.result}")
            elif isinstance(event, ExecutionStart):
                print(f"[Execution Start] {len(event.tool_calls)} tool(s) to execute")
                for tc in event.tool_calls:
                    print(f"  - {tc['function']['name']}")
            elif isinstance(event, AgentEnd):
                final_response = event.response
                print()
        return final_response

    print("=== Round 1 ===")
    await run_and_print(
        user_input=cast(
            list[ExtendedInputContentPart],
            [
                {
                    "type": "text",
                    "text": "What's the weather like in Beijing today? Also help me calculate (3 + 5) * 12",
                }
            ],
        ),
    )

    print("\n=== Round 2 (multi-turn context) ===")
    await run_and_print(
        [
            cast(
                TextContentPart,
                {
                    "type": "text",
                    "text": "Is the weather in that city suitable for going outside?",
                },
            )
        ],
    )

    print("\n=== Round 3 (Long respones without tool calling) ===")
    await run_and_print(
        [
            cast(
                TextContentPart,
                {
                    "type": "text",
                    "text": "What do you think about TV drama?",
                },
            )
        ],
    )
