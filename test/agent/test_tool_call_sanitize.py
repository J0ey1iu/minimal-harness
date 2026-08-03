"""Regression tests for malformed tool-call handling.

Covers two linked defects found via user reports (issues #26/#27):

1. A truncated ``arguments`` string (broken/stopped stream) must not
   crash ``_execute_tools`` — the parse is guarded so it surfaces as a
   normal ``ToolEnd`` error instead of hanging the agent loop waiting
   for a queue sentinel that never arrives.
2. Tool calls whose arguments are not valid JSON are dropped at
   persistence time (``sanitize_tool_calls``) and again on read
   (``get_forward_messages``), so a corrupted message cannot poison the
   next LLM request.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

import pytest
from minimal_harness.agent.base import BaseAgent
from minimal_harness.memory import (
    ConversationMemory,
    assistant_message,
    sanitize_tool_calls,
)
from minimal_harness.tool.base import create_streaming_tool
from minimal_harness.types import ToolCall, ToolEnd, ToolResult


async def _echo_fn(content: str = "") -> AsyncIterator[Any]:
    yield ToolResult(content={"status": "ok", "content": content})


def _tool_call(arguments: str) -> ToolCall:
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": "echo", "arguments": arguments},
    }


# ── sanitize_tool_calls ───────────────────────────────────────────


def test_sanitize_keeps_valid_drops_truncated() -> None:
    valid = _tool_call('{"content": "hi"}')
    truncated = _tool_call('{"content": "unterminated')
    result = sanitize_tool_calls([valid, truncated])
    assert result == [valid]


def test_sanitize_all_invalid_returns_none() -> None:
    assert sanitize_tool_calls([_tool_call('{"content": "x')]) is None
    assert sanitize_tool_calls(None) is None


def test_sanitize_keeps_empty_arguments() -> None:
    # Some models emit an empty (no-arg) call — that is not corruption.
    tc = _tool_call("")
    assert sanitize_tool_calls([tc]) == [tc]


# ── get_forward_messages heals corrupted history ──────────────────


@pytest.mark.asyncio
async def test_forward_messages_heals_corrupted_tool_calls() -> None:
    mem = ConversationMemory()
    await mem.add_message({"role": "user", "content": [{"type": "text", "text": "q"}]})
    await mem.add_message(
        assistant_message("partial text", [_tool_call('{"content": "unterminated')])
    )
    fwd = mem.get_forward_messages()
    assert [m["role"] for m in fwd] == ["user", "assistant"]
    assert fwd[-1]["tool_calls"] is None
    assert fwd[-1]["content"] == "partial text"


# ── _execute_tools survives truncated arguments ───────────────────


@pytest.mark.asyncio
async def test_execute_tools_no_hang_on_truncated_args() -> None:
    agent = BaseAgent(llm_provider=None)  # type: ignore[arg-type]
    mem = ConversationMemory()
    tool = create_streaming_tool(name="echo", fn=_echo_fn)
    events = [
        e
        async for e in agent._execute_tools(
            [_tool_call('{"content": "unterminated')],
            stop_event=None,
            tools=[tool],
            memory=mem,
        )
    ]
    types = [type(e).__name__ for e in events]
    assert types == ["ExecutionStart", "ToolEnd", "MessageEvent", "ExecutionEnd"]
    tool_end = next(e for e in events if isinstance(e, ToolEnd))
    assert isinstance(tool_end.result, Exception)
    # The tool message is persisted so the LLM sees the error and can retry.
    tool_msgs = [m for m in mem.get_all_messages() if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert "[Error]" in tool_msgs[0]["content"]
