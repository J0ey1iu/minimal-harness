from __future__ import annotations

from typing import Any, AsyncIterator, Sequence, cast

import pytest
from minimal_harness.agent.base import BaseAgent
from minimal_harness.agent.dummy import DummyAgent
from minimal_harness.llm.llm import LLMResponse, Stream
from minimal_harness.llm.openai import _convert_messages
from minimal_harness.memory import ConversationMemory, Message
from minimal_harness.types import (
    AgentEnd,
    AgentEvent,
    LLMChunkDelta,
)


def _source_of(m: Message) -> Any:
    return cast(dict[str, Any], m).get("source")


class ScriptedLLMProvider:
    """Returns a single plain text response (no tool calls)."""

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[Any] = (),
        stop_event: Any = None,
        **kwargs: Any,
    ) -> Stream[LLMChunkDelta]:
        self.calls.append(list(messages))

        async def _gen() -> AsyncIterator[Any]:
            yield LLMResponse(
                content="ok",
                reasoning_content=None,
                tool_calls=[],
                finish_reason="stop",
                usage=None,
            )

        return Stream(_gen())


async def _run_agent(agent: Any, meta: dict | None) -> list[Message]:
    memory = ConversationMemory()
    events: list[AgentEvent] = []
    async for evt in agent.run(
        user_input=[{"type": "text", "text": "hello"}],
        stop_event=None,
        memory=memory,
        tools=[],
        system_prompt="",
        user_message_meta=meta,
    ):
        events.append(evt)
    assert isinstance(events[-1], AgentEnd)
    return memory.get_all_messages()


@pytest.mark.asyncio
async def test_user_message_meta_is_merged_into_persisted_message() -> None:
    llm = ScriptedLLMProvider()
    agent = BaseAgent(llm_provider=llm, max_iterations=1)
    msgs = await _run_agent(agent, {"source": "auto"})
    assert msgs[0]["role"] == "user"
    assert _source_of(msgs[0]) == "auto"


@pytest.mark.asyncio
async def test_default_run_has_no_marker() -> None:
    llm = ScriptedLLMProvider()
    agent = BaseAgent(llm_provider=llm, max_iterations=1)
    msgs = await _run_agent(agent, None)
    assert msgs[0]["role"] == "user"
    assert _source_of(msgs[0]) is None


@pytest.mark.asyncio
async def test_marker_is_stripped_from_llm_payload() -> None:
    llm = ScriptedLLMProvider()
    agent = BaseAgent(llm_provider=llm, max_iterations=1)
    await _run_agent(agent, {"source": "auto"})
    # The OpenAI provider converter rebuilds messages from known fields,
    # so the marker must never reach the wire.
    converted = _convert_messages(llm.calls[0])
    assert converted
    for msg in converted:
        assert "source" not in msg


@pytest.mark.asyncio
async def test_dummy_agent_honors_user_message_meta() -> None:
    llm = ScriptedLLMProvider()
    agent = DummyAgent(llm_provider=llm)
    msgs = await _run_agent(agent, {"source": "auto"})
    assert msgs[0]["role"] == "user"
    assert _source_of(msgs[0]) == "auto"
