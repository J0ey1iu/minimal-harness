import os

import pytest
import asyncio
from minimal_harness import Tool, Agent
from openai import AsyncOpenAI


async def get_weather(city: str) -> dict:
    """模拟天气查询"""
    await asyncio.sleep(0.2)
    return {"city": city, "temperature": "22°C", "condition": "晴"}


async def calculator(expression: str) -> dict:
    """简单计算器"""
    result = eval(expression, {"__builtins__": {}})
    return {"expression": expression, "result": result}


@pytest.mark.asyncio
async def test():
    tools = [
        Tool(
            name="get_weather",
            description="查询指定城市的天气",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                },
                "required": ["city"],
            },
            fn=get_weather,
        ),
        Tool(
            name="calculator",
            description="计算数学表达式",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "合法的 Python 数学表达式",
                    },
                },
                "required": ["expression"],
            },
            fn=calculator,
        ),
    ]

    agent = Agent(
        model="minimax-m2.7",
        system_prompt="你是一个助手，可以查天气和做计算。",
        tools=tools,
        client=AsyncOpenAI(
            api_key=os.getenv("AIHUBMIX_API_KEY"),
            base_url="https://aihubmix.com/v1",
        ),
    )

    # chunk 级回调：实时打印
    async def on_chunk(text: str, is_done: bool):
        if not is_done:
            print(text, end="", flush=True)
        else:
            print()  # 换行

    print("=== Round 1 ===")
    await agent.run(
        "北京今天天气怎么样？顺便帮我算一下 (3 + 5) * 12", on_chunk=on_chunk
    )

    print("\n=== Round 2 (多轮上下文) ===")
    await agent.run("刚才那个城市的天气适合出门吗？", on_chunk=on_chunk)

    print("\n=== Round 3 (长回答无调用) ===")
    await agent.run("What do you think about TV drama?", on_chunk=on_chunk)
