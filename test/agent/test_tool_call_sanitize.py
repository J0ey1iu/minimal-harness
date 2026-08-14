"""Regression tests for malformed tool-call handling.

Covers two linked defects found via user reports (issues #26/#27):

1. A truncated ``arguments`` string (broken/stopped stream) must not
   crash ``_execute_tools`` — the parse is guarded so it surfaces as a
   normal ``ToolEnd`` error instead of hanging the agent loop waiting
   for a queue sentinel that never arrives.
2. Tool calls whose arguments are not valid JSON are dropped at every
   LLM boundary: ``sanitize_tool_calls`` + a dangling-tool-message guard
   in ``get_forward_messages`` (main loop) and in
   ``build_chat_payload`` (compaction summarizer), so a corrupted
   message cannot poison the next LLM request. Persisted history stays
   faithful (display/export keep the failed call).
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


@pytest.mark.asyncio
async def test_forward_messages_drops_dangling_tool_message() -> None:
    """A tool message whose assistant call was dropped (truncated args)
    must not reach the LLM — the API rejects an undeclared tool_call_id."""
    mem = ConversationMemory()
    await mem.add_message({"role": "user", "content": [{"type": "text", "text": "q"}]})
    await mem.add_message(
        assistant_message("partial text", [_tool_call('{"content": "x')])
    )
    await mem.add_message(
        {"role": "tool", "tool_call_id": "call_1", "content": "[Error] JSONDecodeError"}
    )
    fwd = mem.get_forward_messages()
    assert [m["role"] for m in fwd] == ["user", "assistant"]
    assert fwd[-1]["tool_calls"] is None
    assert fwd[-1]["content"] == "partial text"


@pytest.mark.asyncio
async def test_forward_messages_keeps_valid_pair() -> None:
    """A healthy assistant tool_call + tool response pair passes through."""
    mem = ConversationMemory()
    await mem.add_message({"role": "user", "content": [{"type": "text", "text": "q"}]})
    await mem.add_message(assistant_message("", [_tool_call('{"content": "ok"}')]))
    await mem.add_message(
        {"role": "tool", "tool_call_id": "call_1", "content": '{"status": "ok"}'}
    )
    fwd = mem.get_forward_messages()
    assert [m["role"] for m in fwd] == ["user", "assistant", "tool"]
    assert fwd[-1]["content"] == '{"status": "ok"}'


@pytest.mark.asyncio
async def test_forward_messages_strips_unanswered_tool_call() -> None:
    """An assistant tool_call that was never answered (run interrupted
    mid-tool, client disconnect) must not reach the LLM — providers reject
    an unanswered call (InferHub 2013 "tool call result does not follow
    tool call"). The call is stripped from the assistant message, its
    content (if any) is kept (mh-incubator #48)."""
    mem = ConversationMemory()
    await mem.add_message({"role": "user", "content": [{"type": "text", "text": "q"}]})
    await mem.add_message(
        assistant_message("will check", [_tool_call('{"content": "ok"}')])
    )
    await mem.add_message(
        {"role": "user", "content": [{"type": "text", "text": "go on"}]}
    )
    fwd = mem.get_forward_messages()
    assert [m["role"] for m in fwd] == ["user", "assistant", "user"]
    assert fwd[1]["tool_calls"] is None
    assert fwd[1]["content"] == "will check"


@pytest.mark.asyncio
async def test_forward_messages_strips_partially_answered_calls() -> None:
    """Of two calls declared by one assistant message, the unanswered one
    is stripped while the answered pair survives."""
    mem = ConversationMemory()
    await mem.add_message({"role": "user", "content": [{"type": "text", "text": "q"}]})
    await mem.add_message(
        assistant_message(
            "", [_tool_call('{"content": "ok"}'), {**_tool_call("{}"), "id": "call_2"}]
        )
    )
    await mem.add_message(
        {"role": "tool", "tool_call_id": "call_1", "content": '{"status": "ok"}'}
    )
    await mem.add_message(
        {"role": "user", "content": [{"type": "text", "text": "next"}]}
    )
    fwd = mem.get_forward_messages()
    assert [m["role"] for m in fwd] == ["user", "assistant", "tool", "user"]
    calls = fwd[1]["tool_calls"]
    assert [tc["id"] for tc in calls] == ["call_1"]


@pytest.mark.asyncio
async def test_forward_messages_strips_trailing_unanswered_call() -> None:
    """Buffer ends with an unanswered assistant tool_call — the last
    message of the session — the call is stripped, content kept."""
    mem = ConversationMemory()
    await mem.add_message({"role": "user", "content": [{"type": "text", "text": "q"}]})
    await mem.add_message(
        assistant_message("final words", [_tool_call('{"content": "ok"}')])
    )
    fwd = mem.get_forward_messages()
    assert [m["role"] for m in fwd] == ["user", "assistant"]
    assert fwd[-1]["tool_calls"] is None
    assert fwd[-1]["content"] == "final words"


@pytest.mark.asyncio
async def test_forward_messages_drops_contentless_unanswered_call() -> None:
    """Assistant with only an unanswered call (no content) is removed
    entirely from the payload."""
    mem = ConversationMemory()
    await mem.add_message({"role": "user", "content": [{"type": "text", "text": "q"}]})
    await mem.add_message(assistant_message("", [_tool_call('{"content": "ok"}')]))
    await mem.add_message(
        {"role": "user", "content": [{"type": "text", "text": "next"}]}
    )
    fwd = mem.get_forward_messages()
    assert [m["role"] for m in fwd] == ["user", "user"]


# ── compaction summarizer payload sanitization ────────────────────


def _summarize_payload(messages: list[dict]) -> list[dict]:
    from minimal_harness.agent._compaction import build_chat_payload

    return build_chat_payload("sys", messages, None, summary_prompt="summarize")


def test_compaction_payload_drops_truncated_call_and_dangling_tool() -> None:
    payload = _summarize_payload(
        [
            assistant_message("", [_tool_call('{"content": "x')]),
            {"role": "tool", "tool_call_id": "call_1", "content": "[Error] boom"},
        ]
    )
    roles = [m["role"] for m in payload]
    assert "tool" not in roles
    assert all(not m.get("tool_calls") for m in payload)


def test_compaction_payload_keeps_valid_pair_and_strips_dangling_call() -> None:
    payload = _summarize_payload(
        [
            assistant_message("", [_tool_call('{"content": "ok"}')]),
            {"role": "tool", "tool_call_id": "call_1", "content": "done"},
            assistant_message("", [_tool_call('{"content": "never ran"}')]),
        ]
    )
    # Valid pair kept; the last assistant's call has no following tool
    # response → stripped → empty assistant removed (pre-existing
    # behaviour). Payload ends with the user summary request.
    assert [m["role"] for m in payload] == ["system", "assistant", "tool", "user"]
    assert payload[1].get("tool_calls") is not None


def test_compaction_payload_strips_partially_answered_calls() -> None:
    """Of two calls declared by one assistant message, the unanswered one
    is stripped from the summarizer payload (mirror of the forward-message
    healing, mh-incubator #48)."""
    payload = _summarize_payload(
        [
            assistant_message(
                "",
                [_tool_call('{"content": "ok"}'), {**_tool_call("{}"), "id": "call_2"}],
            ),
            {"role": "tool", "tool_call_id": "call_1", "content": "done"},
            {"role": "user", "content": [{"type": "text", "text": "next"}]},
        ]
    )
    assistant_msgs = [m for m in payload if m["role"] == "assistant"]
    calls = [tc for m in assistant_msgs for tc in (m.get("tool_calls") or [])]
    assert [tc["id"] for tc in calls] == ["call_1"]


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


# ── _execute_tools skips name-less (truncated) tool calls ─────────


@pytest.mark.asyncio
async def test_execute_tools_skips_tool_call_without_name() -> None:
    # 回归（issue #62）：名称 chunk 未到达的截断调用不执行、不产生
    # ToolStart/ToolEnd（否则会以 "unknown" 名记进 metrics）。
    agent = BaseAgent(llm_provider=None)  # type: ignore[arg-type]
    mem = ConversationMemory()
    tool = create_streaming_tool(name="echo", fn=_echo_fn)
    nameless: ToolCall = {
        "id": "call_x",
        "type": "function",
        "function": {"name": "", "arguments": "{}"},
    }
    events = [
        e
        async for e in agent._execute_tools(
            [nameless, _tool_call("{}")],
            stop_event=None,
            tools=[tool],
            memory=mem,
        )
    ]
    tool_ends = [e for e in events if isinstance(e, ToolEnd)]
    assert len(tool_ends) == 1
    assert tool_ends[0].tool_call["function"]["name"] == "echo"
    # 只有真实调用留下 tool 消息，空名调用不落库。
    tool_msgs = [m for m in mem.get_all_messages() if m["role"] == "tool"]
    assert len(tool_msgs) == 1
