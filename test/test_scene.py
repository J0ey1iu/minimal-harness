import asyncio
import os
from typing import Iterable, cast

import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk

from minimal_harness.agent import OpenAIAgent
from minimal_harness.llm.openai import OpenAILLMProvider
from minimal_harness.memory import (
    ConversationMemory,
    ExtendedInputContentPart,
    FileContentPart,
    InputContentPart,
)
from minimal_harness.tool.base import Tool


async def write_rebate_article(contract_file_id: str) -> str:
    """Simulate weather query"""
    await asyncio.sleep(1)
    return """您的返利方案已经写好了，您可以用EDM298873934去企业文档中查看"""


@pytest.mark.asyncio
async def test_scene():

    tools = [
        Tool(
            name="write_rebate_article",
            description="写rebate方案，有标准的文档支撑，写出来的方案合规好用",
            parameters={
                "type": "object",
                "properties": {
                    "contract_file_id": {
                        "type": "string",
                        "description": "合同文件的ID",
                    },
                },
                "required": ["contract_file_id"],
            },
            fn=write_rebate_article,
        ),
    ]
    client = AsyncOpenAI(
        api_key=os.getenv("AIHUBMIX_API_KEY"),
        base_url="https://aihubmix.com/v1",
    )
    llm_provider = OpenAILLMProvider(client=client, model="qwen3.5-27b")
    memory = ConversationMemory(
        system_prompt="你是一个Rebate专员，你擅长写返利方案以及检查相关文档。"
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
        if not chunk:
            raise ValueError("chunk is None")
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            print(delta.content, end="", flush=True)

    async def custom_input_conversion(
        original: Iterable[ExtendedInputContentPart],
    ) -> Iterable[InputContentPart]:
        converted = []
        for part in original:
            if part["type"] == "file":
                converted.append(
                    {
                        "type": "text",
                        "text": f"file_id: {part['file']['file_id']}",
                    }
                )
            else:
                converted.append(part)
        return converted

    await agent.run(
        user_input=cast(
            list[ExtendedInputContentPart],
            [
                {
                    "type": "text",
                    "text": "帮我根据这篇合同写一个Rebate方案",
                },
                {
                    "type": "file",
                    "file": {"file_name": "contract.docx", "file_id": "989898"},
                },
            ],
        ),
        on_chunk=on_chunk,
        custom_input_conversion=custom_input_conversion,
    )
