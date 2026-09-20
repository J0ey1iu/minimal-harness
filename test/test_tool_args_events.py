"""RFC #60 §2/§3: streamed tool-arg delta events + stable call_id contract."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Sequence

from minimal_harness.agent.base import BaseAgent
from minimal_harness.llm.llm import LLMResponse, Stream
from minimal_harness.memory import ConversationMemory
from minimal_harness.tool.base import StreamingTool
from minimal_harness.types import (
    LLMChunkDelta,
    ToolArgsDelta,
    ToolArgsStart,
    ToolCall,
    ToolCallDelta,
    ToolRoundStart,
)

_CALL0: ToolCall = {
    "id": "call_0",
    "type": "function",
    "function": {"name": "t", "arguments": '{"x": "1"}'},
}
_CALL1: ToolCall = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "t", "arguments": "{}"},
}


class _StreamingArgsProvider:
    """First call streams tool_calls fragments; second call answers."""

    def __init__(
        self, fragments: list[LLMChunkDelta], final_calls: list[ToolCall]
    ) -> None:
        self._fragments = fragments
        self._final_calls = final_calls
        self.calls = 0

    async def chat(
        self,
        messages: Sequence[Any],
        tools: Sequence[Any] = (),
        stop_event: asyncio.Event | None = None,
        **kwargs: Any,
    ) -> Stream[LLMChunkDelta]:
        self.calls += 1

        async def _gen() -> AsyncIterator[Any]:
            if self.calls == 1:
                for f in self._fragments:
                    yield f
                yield LLMResponse(
                    content=None,
                    reasoning_content=None,
                    tool_calls=self._final_calls,
                    finish_reason="tool_calls",
                )
            else:
                yield LLMResponse(
                    content="done",
                    reasoning_content=None,
                    tool_calls=[],
                    finish_reason="stop",
                )

        return Stream(_gen())


class _FinalOnlyProvider:
    """Non-streaming: chunks carry no tool_calls fragments."""

    async def chat(
        self,
        messages: Sequence[Any],
        tools: Sequence[Any] = (),
        stop_event: asyncio.Event | None = None,
        **kwargs: Any,
    ) -> Stream[LLMChunkDelta]:
        async def _gen() -> AsyncIterator[Any]:
            yield LLMChunkDelta(content="thinking")
            yield LLMResponse(
                content=None,
                reasoning_content=None,
                tool_calls=[_CALL0],
                finish_reason="tool_calls",
            )

        return Stream(_gen())


async def _run(provider: Any) -> list[Any]:
    agent = BaseAgent(
        llm_provider=provider,
        emit_delta_events=True,
        emit_message_events=False,
    )
    tool = StreamingTool(name="t", description="t", parameters={}, fn=_tool_fn)
    return [ev async for ev in agent.run([], memory=ConversationMemory(), tools=[tool])]


async def _tool_fn(**_: Any) -> AsyncIterator[str]:
    yield "ok"


def test_tool_args_events_from_streamed_fragments():
    events = asyncio.run(
        _run(
            _StreamingArgsProvider(
                fragments=[
                    LLMChunkDelta(
                        tool_calls=[
                            ToolCallDelta(
                                index=0, id="call_0", name="t", arguments='{"x":'
                            )
                        ]
                    ),
                    LLMChunkDelta(
                        tool_calls=[ToolCallDelta(index=0, arguments='"1"}')]
                    ),
                    LLMChunkDelta(
                        tool_calls=[
                            ToolCallDelta(
                                index=1, id="call_1", name="t", arguments="{}"
                            )
                        ]
                    ),
                ],
                final_calls=[_CALL0, _CALL1],
            )
        )
    )
    starts = [e for e in events if isinstance(e, ToolArgsStart)]
    deltas = [e for e in events if isinstance(e, ToolArgsDelta)]
    assert [s.call_id for s in starts] == ["call_0", "call_1"]
    assert [s.name for s in starts] == ["t", "t"]
    # incremental arguments chunks, consumer concatenates
    call0_chunks = [d.arguments_chunk for d in deltas if d.call_id == "call_0"]
    assert call0_chunks == ['{"x":', '"1"}']
    assert "".join(call0_chunks) == '{"x":"1"}'


def test_call_id_contract_spans_streaming_and_execution():
    events = asyncio.run(
        _run(
            _StreamingArgsProvider(
                fragments=[
                    LLMChunkDelta(
                        tool_calls=[ToolCallDelta(index=0, id="call_0", name="t")]
                    ),
                    LLMChunkDelta(
                        tool_calls=[ToolCallDelta(index=1, id="call_5", name="t")]
                    ),
                ],
                final_calls=[_CALL0, _CALL1],
            )
        )
    )
    starts = [e for e in events if isinstance(e, ToolArgsStart)]
    round_start = next(e for e in events if isinstance(e, ToolRoundStart))
    # RFC #60 §3: same call_id across the pending (streaming) and executing
    # phases — even when the gateway numbers calls from a non-zero index (5).
    assert [s.call_id for s in starts] == ["call_0", "call_5"]
    assert round_start.ids == ["call_0", "call_1"]


def test_no_tool_args_events_without_fragments():
    events = asyncio.run(_run(_FinalOnlyProvider()))
    assert not any(isinstance(e, (ToolArgsStart, ToolArgsDelta)) for e in events)
    # execution phase still intact
    assert any(isinstance(e, ToolRoundStart) for e in events)


async def test_tool_args_off_by_default():
    provider = _StreamingArgsProvider(
        fragments=[
            LLMChunkDelta(
                tool_calls=[
                    ToolCallDelta(index=0, id="call_0", name="t", arguments="{}")
                ]
            )
        ],
        final_calls=[_CALL0],
    )
    agent = BaseAgent(
        llm_provider=provider,
        emit_delta_events=False,  # default
        emit_message_events=False,
    )
    tool = StreamingTool(name="t", description="t", parameters={}, fn=_tool_fn)
    events = [
        ev async for ev in agent.run([], memory=ConversationMemory(), tools=[tool])
    ]
    assert not any(isinstance(e, (ToolArgsStart, ToolArgsDelta)) for e in events)
