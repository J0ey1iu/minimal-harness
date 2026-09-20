"""RFC #57 delta events: max_tool_rounds cap + ToolRoundStart/End granularity."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Sequence

from minimal_harness.agent.base import BaseAgent
from minimal_harness.agent.middleware import Middleware
from minimal_harness.llm.llm import LLMResponse, Stream
from minimal_harness.memory import ConversationMemory
from minimal_harness.tool.base import StreamingTool
from minimal_harness.types import (
    AgentEnd,
    LLMChunkDelta,
    MessageEvent,
    TokenUsage,
    ToolCall,
    ToolRoundEnd,
    ToolRoundStart,
)

_CALL_A: ToolCall = {
    "id": "a1",
    "type": "function",
    "function": {"name": "t", "arguments": "{}"},
}
_CALL_B: ToolCall = {
    "id": "b1",
    "type": "function",
    "function": {"name": "t", "arguments": "{}"},
}


def _resp(
    content: str | None = None,
    calls: list[ToolCall] | None = None,
    usage: TokenUsage | None = None,
):
    return LLMResponse(content, None, calls or [], "stop", usage)


class _Scripted:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def chat(
        self,
        messages: Sequence[Any],
        tools: Sequence[Any] = (),
        stop_event: asyncio.Event | None = None,
        **kwargs: Any,
    ) -> Stream[LLMChunkDelta]:
        self.calls += 1
        resp = self._responses.pop(0)

        async def _gen() -> AsyncIterator[Any]:
            yield resp

        return Stream(_gen())


async def _tool_fn(**kwargs: Any) -> AsyncIterator[str]:
    yield "result"


def _tool() -> StreamingTool:
    return StreamingTool(name="t", description="t", parameters={}, fn=_tool_fn)


async def test_max_tool_rounds_stops_followup_generations():
    provider = _Scripted(
        [
            _resp(calls=[_CALL_A]),
            _resp(calls=[_CALL_B]),
            _resp(content="should never be reached"),
        ]
    )
    agent = BaseAgent(llm_provider=provider, max_tool_rounds=2)
    memory = ConversationMemory()
    events = [ev async for ev in agent.run([], memory=memory, tools=[_tool()])]
    assert provider.calls == 2, "cap stops follow-up AFTER the last emitted round"
    # both emitted calls still executed
    assert any(
        isinstance(e, MessageEvent) and e.message["role"] == "tool" for e in events
    )
    assert any(isinstance(e, AgentEnd) for e in events)


async def test_no_cap_keeps_iterating():
    provider = _Scripted([_resp(calls=[_CALL_A]), _resp(content="done")])
    agent = BaseAgent(llm_provider=provider)  # max_tool_rounds defaults to None
    memory = ConversationMemory()
    events = [ev async for ev in agent.run([], memory=memory, tools=[_tool()])]
    assert provider.calls == 2
    assert not any(isinstance(e, ToolRoundStart) for e in events)


async def test_delta_events_emitted_each_round():
    provider = _Scripted([_resp(calls=[_CALL_A]), _resp(content="done")])
    agent = BaseAgent(llm_provider=provider, emit_delta_events=True)
    memory = ConversationMemory()
    events = [ev async for ev in agent.run([], memory=memory, tools=[_tool()])]

    starts = [e for e in events if isinstance(e, ToolRoundStart)]
    ends = [e for e in events if isinstance(e, ToolRoundEnd)]
    assert len(starts) == 1 and len(ends) == 1
    assert starts[0].ids == ["a1"]
    assert starts[0].names == ["t"]
    assert starts[0].kinds == ["function"]
    assert ends[0].ok is True
    assert ends[0].count == 1


async def test_delta_events_off_by_default():
    provider = _Scripted([_resp(calls=[_CALL_A]), _resp(content="done")])
    agent = BaseAgent(llm_provider=provider)  # emit_delta_events default False
    memory = ConversationMemory()
    events = [ev async for ev in agent.run([], memory=memory, tools=[_tool()])]
    assert not any(isinstance(e, (ToolRoundStart, ToolRoundEnd)) for e in events)


async def test_metadata_carries_max_tool_rounds_default():
    from minimal_harness.types import AgentMetadata

    md = AgentMetadata(name="a")
    assert md.max_tool_rounds == 2000


# ── RFC #60 §1: loop-boundary middleware hooks ──────────────────────


class _HookRecorder(Middleware):
    def __init__(self) -> None:
        self.turns: list[Any] = []
        self.rounds: list[Any] = []

    async def on_turn_complete(self, memory: Any, llm_end: Any) -> None:
        self.turns.append((memory, llm_end))

    async def on_tool_round_complete(self, memory: Any, tool_ends: list[Any]) -> None:
        self.rounds.append((memory, tool_ends))


async def test_turn_and_round_boundary_hooks_fire():
    from minimal_harness.tool.base import ToolEnd

    provider = _Scripted([_resp(calls=[_CALL_A]), _resp(content="done")])
    recorder = _HookRecorder()
    agent = BaseAgent(llm_provider=provider, middleware=[recorder])
    memory = ConversationMemory()
    events = [ev async for ev in agent.run([], memory=memory, tools=[_tool()])]
    # two LLM calls → two turn-complete hooks (the second after the final turn)
    assert len(recorder.turns) == 2
    # one tool round → one round-complete hook carrying the ToolEnd
    assert len(recorder.rounds) == 1
    assert any(isinstance(e, ToolEnd) for e in recorder.rounds[0][1])
    assert any(isinstance(e, AgentEnd) for e in events)
