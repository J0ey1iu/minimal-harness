"""Empty LLM responses must surface as errors, not silent stops.

Regression guard for mh-incubator #58: a provider returning no content
and no tool calls used to break the loop with response_text="" — the
replay fallback then surfaced stale text from a previous round, so a
long loop "mysteriously stopped" with no error.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Sequence

import pytest
from minimal_harness.agent.base import BaseAgent
from minimal_harness.llm.llm import LLMResponse, Stream
from minimal_harness.memory import ConversationMemory, Message, TextContentPart
from minimal_harness.types import AgentEnd, AgentEvent, LLMChunkDelta


class EmptyResponseProvider:
    """chat() 返回空 content、无 tool_calls 的 LLMResponse。"""

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[Any] = (),
        stop_event: Any = None,
        **kwargs: Any,
    ) -> Stream[LLMChunkDelta]:
        async def _gen() -> AsyncIterator[Any]:
            yield LLMResponse(
                content="",
                reasoning_content=None,
                tool_calls=[],
                finish_reason="stop",
                usage=None,
            )

        return Stream(_gen())


@pytest.mark.asyncio
async def test_empty_response_ends_run_with_error() -> None:
    agent = BaseAgent(llm_provider=EmptyResponseProvider(), max_iterations=5)
    events: list[AgentEvent] = []
    async for event in agent.run(
        user_input=[TextContentPart(type="text", text="hi")],
        memory=ConversationMemory(),
        tools=[],
    ):
        events.append(event)

    end = events[-1]
    assert isinstance(end, AgentEnd)
    assert end.error is not None
    assert "empty response" in end.error
