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


class FlakyProvider:
    """前 N 次返回空响应，之后返回正常文本 —— 用于验证空响应重试 (mh-incubator #87)。"""

    def __init__(self, empty_then: int = 1) -> None:
        self._calls = 0
        self._empty_then = empty_then

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[Any] = (),
        stop_event: Any = None,
        **kwargs: Any,
    ) -> Stream[LLMChunkDelta]:
        calls = self._calls
        self._calls += 1

        async def _gen() -> AsyncIterator[Any]:
            if calls < self._empty_then:
                yield LLMResponse(
                    content="",
                    reasoning_content=None,
                    tool_calls=[],
                    finish_reason="stop",
                    usage=None,
                )
            else:
                yield LLMResponse(
                    content="ok",
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


@pytest.mark.asyncio
async def test_transient_empty_response_is_retried() -> None:
    # 空响应是瞬时抖动时：重试后应正常产出文本，而不是中断 (mh-incubator #87)。
    agent = BaseAgent(llm_provider=FlakyProvider(empty_then=1), max_iterations=5)
    events: list[AgentEvent] = []
    async for event in agent.run(
        user_input=[TextContentPart(type="text", text="hi")],
        memory=ConversationMemory(),
        tools=[],
    ):
        events.append(event)

    end = events[-1]
    assert isinstance(end, AgentEnd)
    assert end.error is None
    assert end.response == "ok"
    # 断言确实发生了重试（provider 被调用 ≥2 次）。
    assert agent._llm_provider._calls >= 2
