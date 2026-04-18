"""Test FrameworkClient events using pytest."""

import asyncio
import os
from typing import AsyncIterator

import pytest

from minimal_harness import StreamingTool
from minimal_harness.client import FrameworkClient
from minimal_harness.client.events import (
    ChunkEvent,
    DoneEvent,
    StoppedEvent,
)
from minimal_harness.llm.openai import OpenAILLMProvider
from minimal_harness.memory import ConversationMemory


async def calculator_handler(expression: str) -> AsyncIterator[dict]:
    yield {"status": "progress", "message": f"I'm about to calculate: {expression}"}
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        yield {"success": True, "expression": expression, "result": result}
    except Exception as e:
        yield {"success": False, "error": str(e)}


calculator_tool = StreamingTool(
    name="calculator",
    description="Evaluate a mathematical expression and return the result.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate (e.g., '2+2', '10*5', '2**3')",
            },
        },
        "required": ["expression"],
    },
    fn=calculator_handler,
)


@pytest.mark.asyncio
async def test_framework_client_events():
    """Test that FrameworkClient properly emits events to the queue."""
    from openai import AsyncOpenAI

    api_key = os.getenv("MH_API_KEY")
    base_url = os.getenv("MH_BASE_URL")
    model = os.getenv("MH_MODEL", "qwen3.5-27b")

    client = AsyncOpenAI(base_url=base_url, api_key=api_key or None)
    llm_provider = OpenAILLMProvider(client=client, model=model)
    memory = ConversationMemory(system_prompt="You are a helpful assistant.")
    tools = [calculator_tool]

    framework_client = FrameworkClient(
        llm_provider=llm_provider,
        tools=tools,
        memory=memory,
    )

    events = []
    stop_event = asyncio.Event()
    output_file = "./test_client_events.txt"

    if os.path.exists(output_file):
        os.remove(output_file)

    async for event in framework_client.run(
        user_input=[{"type": "text", "text": "What is 125 * 37?"}],
        stop_event=stop_event,
    ):
        events.append(event)
        with open(output_file, "a") as f:
            f.write(str(event) + "\n\n")
        if isinstance(event, (DoneEvent, StoppedEvent)):
            break

    assert len(events) > 0, "Should have received at least one event"
    assert any(isinstance(e, ChunkEvent) for e in events), "Should have ChunkEvent"
    assert any(isinstance(e, DoneEvent) for e in events), "Should have DoneEvent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
