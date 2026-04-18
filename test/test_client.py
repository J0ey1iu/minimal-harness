"""Test FrameworkClient events using pytest."""

import pytest


@pytest.mark.asyncio
async def test_framework_client_events():
    """Test that FrameworkClient properly emits events to the queue."""
    import asyncio
    import os

    from openai import AsyncOpenAI

    from minimal_harness.client import FrameworkClient
    from minimal_harness.client.events import (
        ChunkEvent,
        DoneEvent,
        StoppedEvent,
    )
    from minimal_harness.llm.openai import OpenAILLMProvider
    from minimal_harness.memory import ConversationMemory
    from minimal_harness.tool.registry import ToolRegistry

    api_key = os.getenv("MH_API_KEY")
    base_url = os.getenv("MH_BASE_URL")
    model = os.getenv("MH_MODEL", "qwen3.5-27b")

    client = AsyncOpenAI(base_url=base_url, api_key=api_key or None)
    llm_provider = OpenAILLMProvider(client=client, model=model)
    memory = ConversationMemory(system_prompt="You are a helpful assistant.")
    tools = ToolRegistry.get_instance().get_all()

    framework_client = FrameworkClient(
        llm_provider=llm_provider,
        tools=tools,
        memory=memory,
    )

    events = []
    stop_event = asyncio.Event()

    async for event in framework_client.run(
        user_input=[{"type": "text", "text": "What is 2+2?"}],
        stop_event=stop_event,
    ):
        print(event)
        events.append(event)
        if isinstance(event, (DoneEvent, StoppedEvent)):
            break

    assert len(events) > 0, "Should have received at least one event"
    assert any(isinstance(e, ChunkEvent) for e in events), "Should have ChunkEvent"
    assert any(isinstance(e, DoneEvent) for e in events), "Should have DoneEvent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
