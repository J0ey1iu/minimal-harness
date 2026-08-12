"""Summarizer must yield the summary exactly once.

``_summarize`` used to yield the streamed deltas AND then the final
``LLMResponse.content`` again — ``Stream`` swallows the terminal
``LLMResponse`` internally (stores it on ``.response``) instead of
yielding it, so the accumulated summary ended up doubled in the
compaction card and in the persisted summary. Regression test for that.
"""

from __future__ import annotations

from typing import Any, Sequence

import pytest

from minimal_harness.agent._compaction import build_summarizer
from minimal_harness.llm.llm import LLMResponse
from minimal_harness.types import LLMChunkDelta


class _FakeStream:
    """Mimics ``Stream``: yields deltas, swallows the final LLMResponse."""

    def __init__(self, deltas: list[LLMChunkDelta], final: LLMResponse) -> None:
        self._deltas = deltas
        self._final = final
        self._i = 0

    def __aiter__(self) -> "_FakeStream":
        return self

    async def __anext__(self) -> LLMChunkDelta:
        if self._i < len(self._deltas):
            d = self._deltas[self._i]
            self._i += 1
            return d
        raise StopAsyncIteration

    @property
    def response(self) -> LLMResponse:
        return self._final


class _FakeProvider:
    def __init__(self, deltas: list[LLMChunkDelta], final_content: str) -> None:
        self._deltas = deltas
        self._final = LLMResponse(
            content=final_content,
            reasoning_content=None,
            tool_calls=[],
            finish_reason="stop",
        )

    async def chat(
        self, messages: Sequence[Any], tools: Sequence[Any], **kwargs: Any
    ) -> _FakeStream:
        return _FakeStream(self._deltas, self._final)


FULL = "这是一段完整的摘要内容。"


async def _collect(summarizer, messages) -> str:
    return "".join([d async for d in summarizer(messages, None)])


@pytest.mark.asyncio
async def test_streaming_summarizer_yields_full_text_once() -> None:
    """Streamed deltas already cover the full text — no doubling."""
    provider = _FakeProvider(
        deltas=[
            LLMChunkDelta(content=FULL[:6]),
            LLMChunkDelta(content=FULL[6:12]),
            LLMChunkDelta(content=FULL[12:]),
        ],
        final_content=FULL,
    )
    summarizer = build_summarizer(provider, system_prompt="")
    out = await _collect(summarizer, [{"role": "user", "content": "hi"}])
    assert out == FULL


@pytest.mark.asyncio
async def test_non_streaming_summarizer_falls_back_to_final() -> None:
    """No deltas (non-streaming provider) → fall back to the final response."""
    provider = _FakeProvider(deltas=[], final_content=FULL)
    summarizer = build_summarizer(provider, system_prompt="")
    out = await _collect(summarizer, [{"role": "user", "content": "hi"}])
    assert out == FULL
