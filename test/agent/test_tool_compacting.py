"""Tests for ToolCompactionAgent and Memory.compress_tool_messages()."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Sequence

import pytest
from minimal_harness.agent.middleware import Middleware
from minimal_harness.agent.tool_compacting import ToolCompactionAgent
from minimal_harness.llm.llm import LLMResponse, Stream
from minimal_harness.memory import ConversationMemory, Message
from minimal_harness.types import (
    AgentEnd,
    CompactionEnd,
    CompactionStart,
    LLMChunkDelta,
    ToolCall,
)

# ---------------------------------------------------------------------------
# Helpers (mirrored from test_compacting.py)
# ---------------------------------------------------------------------------


def _user(text: str) -> Message:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _assistant(text: str, tool_calls: list[ToolCall] | None = None) -> Message:
    return {"role": "assistant", "content": text, "tool_calls": tool_calls}


def _tool(tool_call_id: str, content: str) -> Message:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


class ScriptedLLMProvider:
    """An LLMProvider that returns a scripted sequence of responses."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[list[Message]] = []

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[Any] = (),
        stop_event: asyncio.Event | None = None,
        **kwargs: Any,
    ) -> Stream[LLMChunkDelta]:
        self.calls.append(list(messages))
        if not self._responses:
            raise RuntimeError("ScriptedLLMProvider: out of scripted responses")
        resp = self._responses.pop(0)

        async def _gen() -> AsyncIterator[Any]:
            if resp.content:
                yield LLMChunkDelta(content=resp.content)
            yield resp

        return Stream(_gen())


async def _streaming_summarizer(
    messages: list[Message], existing_summary: str | None
) -> AsyncIterator[str]:
    """Summarizer that emits chunks for testing."""
    count = len(messages)
    for word in ["[", f"{count}", " msgs", "]"]:
        yield word


# ---------------------------------------------------------------------------
# Memory.compress_tool_messages() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compress_tool_no_tool_messages() -> None:
    """No tool messages → no compression events."""
    mem = ConversationMemory()
    await mem.add_message(_user("hi"))
    await mem.add_message(_assistant("hello"))
    events = [
        e
        async for e in mem.compress_tool_messages(
            _streaming_summarizer, tool_token_threshold=100
        )
    ]
    assert events == []


@pytest.mark.asyncio
async def test_compress_tool_under_threshold() -> None:
    """Small tool messages → under threshold → no compression."""
    mem = ConversationMemory()
    await mem.add_message(_user("q"))
    await mem.add_message(
        _assistant(
            "thinking",
            [
                ToolCall(
                    id="c1",
                    type="function",
                    function={"name": "get_weather", "arguments": "{}"},
                )
            ],
        )
    )
    await mem.add_message(_tool("c1", "sunny, 25°C"))  # ~15 chars → ~7 tokens
    events = [
        e
        async for e in mem.compress_tool_messages(
            _streaming_summarizer,
            tool_token_threshold=100,  # threshold high
        )
    ]
    assert events == [], "should not compress when under threshold"


@pytest.mark.asyncio
async def test_compress_tool_exceeds_threshold() -> None:
    """Large tool messages → exceeded threshold → compression happens."""
    mem = ConversationMemory()
    await mem.add_message(_user("q"))
    await mem.add_message(
        _assistant(
            "thinking",
            [
                ToolCall(
                    id="c1",
                    type="function",
                    function={"name": "get_weather", "arguments": "{}"},
                ),
                ToolCall(
                    id="c2",
                    type="function",
                    function={"name": "get_weather", "arguments": "{}"},
                ),
            ],
        )
    )
    # Each tool message has 300 chars → ~150 tokens each → 300 total
    await mem.add_message(_tool("c1", "x" * 300))
    await mem.add_message(_tool("c2", "y" * 300))

    events: list[Any] = []
    async for evt in mem.compress_tool_messages(
        _streaming_summarizer,
        tool_token_threshold=50,  # low threshold
    ):
        events.append(evt)

    # Should have Start + Chunks + End
    assert len(events) >= 2
    assert isinstance(events[0], CompactionStart)
    assert isinstance(events[-1], CompactionEnd)
    assert events[0].dropped_message_count == 2
    assert events[-1].error is None
    assert events[-1].dropped_message_count == 2
    assert "2 msgs" in events[-1].summary

    # Verify buffer: tool messages preserved individually with compressed content
    tool_msgs = [m for m in mem.get_all_messages() if m.get("role") == "tool"]
    assert len(tool_msgs) == 2, f"expected 2 tool msgs preserved, got {len(tool_msgs)}"
    for m in tool_msgs:
        assert "compressed" in m.get("meta", {}), (
            "Each tool msg should have compressed=True"
        )
        assert m["meta"]["compressed"] is True
    # Combined content across all preserved tool messages should contain the summary
    combined = "".join(str(m.get("content", "")) for m in tool_msgs)
    assert "[2 msgs]" in combined, (
        "Summary text should appear across combined tool msgs"
    )

    # Verify replay history retains original tool messages
    replay = mem.get_replay_messages()
    replay_tool_msgs = [m for m in replay if m.get("role") == "tool"]
    # 2 compressed (current) + 2 originals (pre_compression) = 4
    assert len(replay_tool_msgs) == 4, (
        f"expected 4 replay tool msgs, got {len(replay_tool_msgs)}"
    )


@pytest.mark.asyncio
async def test_compress_tool_failure_soft() -> None:
    """Summarizer failure → buffer unchanged, error in CompactionEnd."""

    async def failing_summarizer(
        messages: list[Message], existing: str | None
    ) -> AsyncIterator[str]:
        yield "partial-"
        raise RuntimeError("summarizer died")

    mem = ConversationMemory()
    await mem.add_message(_user("q"))
    await mem.add_message(
        _assistant(
            "thinking",
            [
                ToolCall(
                    id="c1",
                    type="function",
                    function={"name": "tool", "arguments": "{}"},
                )
            ],
        )
    )
    await mem.add_message(_tool("c1", "x" * 500))  # big tool result

    events: list[Any] = []
    async for evt in mem.compress_tool_messages(
        failing_summarizer, tool_token_threshold=10
    ):
        events.append(evt)

    assert isinstance(events[-1], CompactionEnd)
    assert events[-1].error is not None
    assert "RuntimeError" in events[-1].error
    assert events[-1].dropped_message_count == 0  # no messages dropped

    # Buffer unchanged
    tool_msgs = [m for m in mem.get_all_messages() if m.get("role") == "tool"]
    assert len(tool_msgs) == 1  # still the original


@pytest.mark.asyncio
async def test_compress_tool_with_threshold_zero() -> None:
    """threshold=0 → always compress if there are any tool messages."""
    mem = ConversationMemory()
    await mem.add_message(_user("q"))
    await mem.add_message(
        _assistant(
            "thinking",
            [
                ToolCall(
                    id="c1",
                    type="function",
                    function={"name": "tool", "arguments": "{}"},
                )
            ],
        )
    )
    await mem.add_message(_tool("c1", "tiny"))  # only 5 chars → ~2 tokens

    events: list[Any] = []
    async for evt in mem.compress_tool_messages(
        _streaming_summarizer, tool_token_threshold=0
    ):
        events.append(evt)

    assert len(events) >= 1
    assert isinstance(events[0], CompactionStart)
    assert isinstance(events[-1], CompactionEnd)
    assert events[-1].error is None
    assert events[-1].dropped_message_count == 1


# ---------------------------------------------------------------------------
# ToolCompactionAgent integration tests
# ---------------------------------------------------------------------------


async def _collect(
    agent: ToolCompactionAgent,
    user_input: list[dict],
    memory: ConversationMemory,
    tools: list[Any],
) -> list[Any]:
    events: list[Any] = []
    async for evt in agent.run(
        user_input=user_input,
        memory=memory,
        tools=tools,
    ):
        events.append(evt)
    return events


@pytest.mark.asyncio
async def test_agent_no_tools_no_compression() -> None:
    """No tool calls → no compression events at all."""
    provider = ScriptedLLMProvider(
        [
            LLMResponse(
                content="hello back",
                reasoning_content=None,
                tool_calls=[],
                finish_reason="stop",
                usage={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            ),
        ]
    )
    agent = ToolCompactionAgent(
        llm_provider=provider,
        summarizer=_streaming_summarizer,
        round_compress=True,
    )
    mem = ConversationMemory()
    events = await _collect(agent, [{"type": "text", "text": "hi"}], mem, [])
    # No CompactionStart/End in events
    compaction_events = [
        e for e in events if isinstance(e, (CompactionStart, CompactionEnd))
    ]
    assert compaction_events == []


@pytest.mark.asyncio
async def test_agent_within_round_compression() -> None:
    """Tool results exceed threshold → compress before next LLM call.

    The agent runs one round:
      LLM → tool_calls → execute tools → _post_tool_execution → compress
    """
    provider = ScriptedLLMProvider(
        [
            # First LLM call: returns tool calls
            LLMResponse(
                content="let me check",
                reasoning_content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        type="function",
                        function={"name": "echo", "arguments": '{"x":"a"}'},
                    ),
                    ToolCall(
                        id="c2",
                        type="function",
                        function={"name": "echo", "arguments": '{"x":"b"}'},
                    ),
                ],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            ),
            # Second LLM call: no tools (final answer)
            LLMResponse(
                content="done",
                reasoning_content=None,
                tool_calls=[],
                finish_reason="stop",
                usage={"prompt_tokens": 50, "completion_tokens": 3, "total_tokens": 53},
            ),
        ]
    )

    class EchoTool:
        name = "echo"

        def to_schema(self) -> dict:
            return {
                "name": "echo",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                },
            }

        async def execute(
            self, args: dict, tc: Any, stop_event: Any = None
        ) -> AsyncIterator[Any]:
            from minimal_harness.types import ToolEnd, ToolProgress, ToolResult

            yield ToolProgress(chunk=f"echoing {args['x']}", tool_call=tc)
            yield ToolEnd(
                result=ToolResult(content="x" * 400), tool_call=tc
            )  # 400 chars → ~200 tokens

    # Low threshold so compression triggers
    agent = ToolCompactionAgent(
        llm_provider=provider,
        summarizer=_streaming_summarizer,
        round_compress=True,
    )
    mem = ConversationMemory()

    events = await _collect(
        agent, [{"type": "text", "text": "do it"}], mem, [EchoTool()]
    )

    # Verify compression happened
    starts = [e for e in events if isinstance(e, CompactionStart)]
    ends = [e for e in events if isinstance(e, CompactionEnd)]
    assert len(starts) >= 1, "expected at least one CompactionStart"
    assert len(ends) >= 1, "expected at least one CompactionEnd"
    assert ends[-1].error is None, f"compaction failed: {ends[-1].error}"

    # Verify final state: tool messages compressed in place (each preserved)
    tool_msgs = [m for m in mem.get_all_messages() if m.get("role") == "tool"]
    assert len(tool_msgs) == 2, (
        f"expected 2 compressed tool msgs (one per call), got {len(tool_msgs)}"
    )
    for m in tool_msgs:
        assert m.get("meta", {}).get("compressed"), "Each tool msg should be compressed"

    # Verify AgentEnd has the correct response
    agent_ends = [e for e in events if isinstance(e, AgentEnd)]
    assert len(agent_ends) == 1
    assert agent_ends[0].response == "done"


@pytest.mark.asyncio
async def test_agent_within_round_no_compression_needed() -> None:
    """Small tool results → no compression (under threshold)."""
    provider = ScriptedLLMProvider(
        [
            LLMResponse(
                content="let me check",
                reasoning_content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        type="function",
                        function={"name": "echo", "arguments": '{"x":"a"}'},
                    ),
                ],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            ),
            LLMResponse(
                content="done",
                reasoning_content=None,
                tool_calls=[],
                finish_reason="stop",
                usage={"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
            ),
        ]
    )

    class EchoTool:
        name = "echo"

        def to_schema(self) -> dict:
            return {
                "name": "echo",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                },
            }

        async def execute(
            self, args: dict, tc: Any, stop_event: Any = None
        ) -> AsyncIterator[Any]:
            from minimal_harness.types import ToolEnd, ToolProgress, ToolResult

            yield ToolProgress(chunk="small", tool_call=tc)
            yield ToolEnd(
                result=ToolResult(content="tiny"), tool_call=tc
            )  # tiny → under threshold

    agent = ToolCompactionAgent(
        llm_provider=provider,
        summarizer=_streaming_summarizer,
        round_compress=False,
    )
    mem = ConversationMemory()
    events = await _collect(
        agent, [{"type": "text", "text": "do it"}], mem, [EchoTool()]
    )

    starts = [e for e in events if isinstance(e, CompactionStart)]
    assert starts == [], "should not compress when under threshold"

    # Verify tool message is intact
    tool_msgs = [m for m in mem.get_all_messages() if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"] == "tiny"


@pytest.mark.asyncio
async def test_agent_round_compress_at_end() -> None:
    """round_compress=True → post_llm_response compresses remaining tool msgs.

    Even if tool results are small (under within-round threshold),
    at the end of the round they should still be compressed into one summary.
    """
    provider = ScriptedLLMProvider(
        [
            # First LLM call: returns tool calls
            LLMResponse(
                content="let me check",
                reasoning_content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        type="function",
                        function={"name": "echo", "arguments": '{"x":"a"}'},
                    ),
                ],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            ),
            # Second LLM call: no tools (final answer)
            LLMResponse(
                content="done",
                reasoning_content=None,
                tool_calls=[],
                finish_reason="stop",
                usage={"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
            ),
        ]
    )

    class EchoTool:
        name = "echo"

        def to_schema(self) -> dict:
            return {
                "name": "echo",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                },
            }

        async def execute(
            self, args: dict, tc: Any, stop_event: Any = None
        ) -> AsyncIterator[Any]:
            from minimal_harness.types import ToolEnd, ToolProgress, ToolResult

            yield ToolProgress(chunk="small", tool_call=tc)
            yield ToolEnd(result=ToolResult(content="small result"), tool_call=tc)

    # High within-round threshold (no compression during tool exec)
    # But round_compress=True → compression at end of round (threshold=0)
    agent = ToolCompactionAgent(
        llm_provider=provider,
        summarizer=_streaming_summarizer,
        round_compress=True,  # but still compress at end of round
    )
    mem = ConversationMemory()
    events = await _collect(
        agent, [{"type": "text", "text": "do it"}], mem, [EchoTool()]
    )

    # Should have compression events from the round-end compression
    starts = [e for e in events if isinstance(e, CompactionStart)]
    ends = [e for e in events if isinstance(e, CompactionEnd)]
    assert len(starts) >= 1, "expected round-end compression"
    assert len(ends) >= 1
    assert ends[-1].error is None

    # Tool messages should be compressed in place
    tool_msgs = [m for m in mem.get_all_messages() if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].get("meta", {}).get("compressed"), (
        "Tool msg should be marked compressed"
    )


@pytest.mark.asyncio
async def test_agent_multiple_rounds() -> None:
    """Multiple rounds: each round's tool results are compressed independently."""
    provider = ScriptedLLMProvider(
        [
            # Round 1: tool calls
            LLMResponse(
                content="r1 tools",
                reasoning_content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        type="function",
                        function={"name": "echo", "arguments": '{"x":"a"}'},
                    ),
                ],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            ),
            # Round 2: tool calls again
            LLMResponse(
                content="r2 tools",
                reasoning_content=None,
                tool_calls=[
                    ToolCall(
                        id="c2",
                        type="function",
                        function={"name": "echo", "arguments": '{"x":"b"}'},
                    ),
                ],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35},
            ),
            # Round 3: final answer
            LLMResponse(
                content="r3 final",
                reasoning_content=None,
                tool_calls=[],
                finish_reason="stop",
                usage={"prompt_tokens": 50, "completion_tokens": 3, "total_tokens": 53},
            ),
        ]
    )

    class EchoTool:
        name = "echo"

        def to_schema(self) -> dict:
            return {
                "name": "echo",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                },
            }

        async def execute(
            self, args: dict, tc: Any, stop_event: Any = None
        ) -> AsyncIterator[Any]:
            from minimal_harness.types import ToolEnd, ToolProgress, ToolResult

            yield ToolProgress(chunk=f"echo {args['x']}", tool_call=tc)
            yield ToolEnd(result=ToolResult(content="x" * 300), tool_call=tc)

    agent = ToolCompactionAgent(
        llm_provider=provider,
        summarizer=_streaming_summarizer,
        round_compress=True,
    )
    mem = ConversationMemory()
    events = await _collect(
        agent, [{"type": "text", "text": "do it"}], mem, [EchoTool()]
    )

    # Should have multiple compression events
    starts = [e for e in events if isinstance(e, CompactionStart)]
    ends = [e for e in events if isinstance(e, CompactionEnd)]
    assert len(starts) >= 1
    assert len(ends) >= 1
    assert ends[-1].error is None

    # Final: tool messages should be compressed in place
    all_msgs = mem.get_all_messages()
    tool_msgs = [m for m in all_msgs if m.get("role") == "tool"]
    # Each round's tool message preserved individually with compressed content
    assert len(tool_msgs) == 2, (
        f"expected 2 tool msgs (one per round), got {len(tool_msgs)}"
    )
    for m in tool_msgs:
        assert m.get("meta", {}).get("compressed"), (
            "Each tool msg should be marked compressed"
        )


@pytest.mark.asyncio
async def test_agent_compaction_failure_soft() -> None:
    """Tool compression failure → soft fail, agent continues."""

    async def failing_summarizer(
        messages: list[Message], existing: str | None
    ) -> AsyncIterator[str]:
        yield "partial-"
        raise RuntimeError("summarizer died")

    provider = ScriptedLLMProvider(
        [
            LLMResponse(
                content="let me check",
                reasoning_content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        type="function",
                        function={"name": "echo", "arguments": '{"x":"a"}'},
                    ),
                ],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            ),
            LLMResponse(
                content="final answer",
                reasoning_content=None,
                tool_calls=[],
                finish_reason="stop",
                usage={"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
            ),
        ]
    )

    class EchoTool:
        name = "echo"

        def to_schema(self) -> dict:
            return {
                "name": "echo",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                },
            }

        async def execute(
            self, args: dict, tc: Any, stop_event: Any = None
        ) -> AsyncIterator[Any]:
            from minimal_harness.types import ToolEnd, ToolProgress, ToolResult

            yield ToolProgress(chunk="echo", tool_call=tc)
            yield ToolEnd(result=ToolResult(content="x" * 500), tool_call=tc)

    agent = ToolCompactionAgent(
        llm_provider=provider,
        summarizer=failing_summarizer,
        round_compress=True,
    )
    mem = ConversationMemory()
    events = await _collect(
        agent, [{"type": "text", "text": "do it"}], mem, [EchoTool()]
    )

    # Compression failed but agent still completes
    ends = [e for e in events if isinstance(e, CompactionEnd)]
    assert len(ends) >= 1
    assert ends[-1].error is not None
    assert "RuntimeError" in ends[-1].error

    agent_ends = [e for e in events if isinstance(e, AgentEnd)]
    assert len(agent_ends) == 1
    # Agent should still complete without error (soft fail)
    assert agent_ends[0].error is None
    assert agent_ends[0].response == "final answer"

    # Tool messages should still be in memory (not dropped on failure)
    tool_msgs = [m for m in mem.get_all_messages() if m.get("role") == "tool"]
    assert len(tool_msgs) == 1  # original tool msg preserved


@pytest.mark.asyncio
async def test_agent_forwards_events_to_middleware() -> None:
    """Compaction events are forwarded to middleware hooks."""

    class Recorder(Middleware):
        def __init__(self) -> None:
            self.starts: list[CompactionStart] = []
            self.ends: list[CompactionEnd] = []

        async def on_compaction_start(self, event: CompactionStart) -> None:
            self.starts.append(event)

        async def on_compaction_end(self, event: CompactionEnd) -> None:
            self.ends.append(event)

    provider = ScriptedLLMProvider(
        [
            LLMResponse(
                content="tools",
                reasoning_content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        type="function",
                        function={"name": "echo", "arguments": '{"x":"a"}'},
                    ),
                ],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            ),
            LLMResponse(
                content="done",
                reasoning_content=None,
                tool_calls=[],
                finish_reason="stop",
                usage={"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
            ),
        ]
    )

    class EchoTool:
        name = "echo"

        def to_schema(self) -> dict:
            return {
                "name": "echo",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                },
            }

        async def execute(
            self, args: dict, tc: Any, stop_event: Any = None
        ) -> AsyncIterator[Any]:
            from minimal_harness.types import ToolEnd, ToolProgress, ToolResult

            yield ToolProgress(chunk="echo", tool_call=tc)
            yield ToolEnd(result=ToolResult(content="x" * 400), tool_call=tc)

    recorder = Recorder()
    agent = ToolCompactionAgent(
        llm_provider=provider,
        summarizer=_streaming_summarizer,
        round_compress=True,
        middleware=[recorder],
    )
    mem = ConversationMemory()
    await _collect(agent, [{"type": "text", "text": "do it"}], mem, [EchoTool()])

    assert len(recorder.starts) >= 1
    assert len(recorder.ends) >= 1
    assert recorder.ends[-1].error is None
