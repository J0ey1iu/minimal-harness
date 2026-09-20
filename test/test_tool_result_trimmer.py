"""RFC #60 §5: deterministic tool-result trimming; §6 error hints."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Sequence

from minimal_harness.agent.base import BaseAgent, _serialize_tool_error
from minimal_harness.llm.llm import LLMResponse, Stream
from minimal_harness.memory import ConversationMemory
from minimal_harness.tool.base import StreamingTool, ToolExecutionError
from minimal_harness.types import (
    LLMChunkDelta,
    ToolCall,
    ToolProgress,
    ToolResultTrimmer,
)

_CALL: ToolCall = {
    "id": "c1",
    "type": "function",
    "function": {"name": "t", "arguments": "{}"},
}
_BAD_CALL: ToolCall = {
    "id": "bad",
    "type": "function",
    "function": {"name": "t", "arguments": '"bare string"'},
}


# ── trimmer unit ────────────────────────────────────────────────────


def test_trimmer_short_result_unchanged():
    t = ToolResultTrimmer(max_bytes=1000)
    assert t.trim("short") == "short"


def test_trimmer_head_tail_and_recovery_marker():
    t = ToolResultTrimmer(max_bytes=20, head_ratio=0.5, tail_ratio=0.5)
    out = t.trim("A" * 100)
    assert len(out) < 100
    assert "output trimmed" in out
    assert out.startswith("A" * 10)
    # deterministic
    assert t.trim("A" * 100) == out


def test_trimmer_custom_recovery_hint():
    t = ToolResultTrimmer(
        max_bytes=10,
        recovery_hint="\n[re-run: cmd --limit 1000 to see the middle]\n",
    )
    out = t.trim("B" * 50)
    assert "[re-run" in out


def test_trimmer_per_tool_overlay():
    t = ToolResultTrimmer(
        max_bytes=100, per_tool={"cmd": ToolResultTrimmer(max_bytes=8)}
    )
    assert t.effective_for("read_file").max_bytes == 100
    eff = t.effective_for("cmd")
    assert eff.max_bytes == 8
    # global cap (100) applies to non-overlaid tools
    trimmed = t.trim("C" * 200)
    assert trimmed != "C" * 200
    assert len(trimmed.encode()) < 200  # strictly smaller than the input
    assert "output trimmed" in trimmed
    # cmd overlay applies its own (8-byte) cap
    eff_trimmed = eff.trim("C" * 50)
    assert eff_trimmed != "C" * 50
    assert "output trimmed" in eff_trimmed


def test_trimmer_disabled_default():
    t = ToolResultTrimmer()
    assert t.trim("D" * 999) == "D" * 999


# ── trimmer end-to-end: applied to the buffer copy only ─────────────


class _ToolThenDone:
    """First LLM call emits a tool call; second emits the final answer."""

    def __init__(self, call: ToolCall) -> None:
        self._call = call
        self.calls = 0

    async def chat(
        self,
        messages: Sequence[Any],
        tools: Sequence[Any] = (),
        stop_event: asyncio.Event | None = None,
        **kwargs: Any,
    ) -> Stream[LLMChunkDelta]:
        self.calls += 1
        resp = LLMResponse(
            content=None if self.calls == 1 else "done",
            reasoning_content=None,
            tool_calls=[self._call] if self.calls == 1 else [],
            finish_reason="stop",
        )

        async def _gen() -> AsyncIterator[Any]:
            yield resp

        return Stream(_gen())


async def test_trimmer_trims_memory_but_not_progress():
    async def big_fn(**_: Any) -> AsyncIterator[str]:
        yield "FULL-" + "X" * 200

    agent = BaseAgent(
        llm_provider=_ToolThenDone(_CALL),
        tool_result_trimmer=ToolResultTrimmer(max_bytes=32),
    )
    tool = StreamingTool(name="t", description="t", parameters={}, fn=big_fn)
    memory = ConversationMemory()
    events = [ev async for ev in agent.run([], memory=memory, tools=[tool])]

    progress = [e for e in events if isinstance(e, ToolProgress)]
    assert progress and "X" * 200 in str(progress[0].chunk)
    tool_msgs = [m for m in memory.get_all_messages() if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert "output trimmed" in tool_msgs[0]["content"]
    assert "X" * 200 not in tool_msgs[0]["content"]


async def test_no_trimmer_keeps_full_result():
    async def big_fn(**_: Any) -> AsyncIterator[str]:
        yield "Y" * 300

    agent = BaseAgent(llm_provider=_ToolThenDone(_CALL))
    tool = StreamingTool(name="t", description="t", parameters={}, fn=big_fn)
    memory = ConversationMemory()
    _ = [ev async for ev in agent.run([], memory=memory, tools=[tool])]
    tool_msgs = [m for m in memory.get_all_messages() if m["role"] == "tool"]
    assert tool_msgs[0]["content"] == "Y" * 300


# ── §6 error serialization ──────────────────────────────────────────


async def test_non_object_arguments_get_actionable_error():
    agent = BaseAgent(llm_provider=_ToolThenDone(_BAD_CALL))
    tool = StreamingTool(name="t", description="t", parameters={}, fn=_tool_fn)
    memory = ConversationMemory()
    _ = [ev async for ev in agent.run([], memory=memory, tools=[tool])]
    tool_msgs = [m for m in memory.get_all_messages() if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert "bare str" in tool_msgs[0]["content"]
    assert "hint:" in tool_msgs[0]["content"]


def test_plain_exception_keeps_old_format():
    assert _serialize_tool_error(ValueError("boom")) == "[Error] boom"
    assert _serialize_tool_error(ToolExecutionError("oops")) == "[Error] oops"


def test_hinting_exception_gets_actionable_format():
    exc = ToolExecutionError(
        "tool arguments were not a JSON object (got a bare str)",
        hint="re-send arguments as a JSON object matching the declared parameters",
    )
    text = _serialize_tool_error(exc)
    assert text.startswith("[tool error] class=ToolExecutionError")
    assert "re-send arguments as a JSON object" in text


async def _tool_fn(**_: Any) -> AsyncIterator[str]:
    yield "ok"
