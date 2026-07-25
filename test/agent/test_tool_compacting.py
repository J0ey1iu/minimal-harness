"""Tests for ToolCompactionAgent and Memory.discard_tool_messages()."""

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
# Helpers
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
# Memory.discard_tool_messages() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discard_tool_no_tool_messages() -> None:
    """No tool messages → no discard events."""
    mem = ConversationMemory()
    await mem.add_message(_user("hi"))
    await mem.add_message(_assistant("hello"))
    events = [e async for e in mem.discard_tool_messages()]
    assert events == []


@pytest.mark.asyncio
async def test_discard_tool_removes_from_live_buffer() -> None:
    """Tool messages are removed from live buffer but kept in replay."""
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
    await mem.add_message(_tool("c1", "sunny, 25°C"))
    await mem.add_message(_user("follow-up"))

    events: list[Any] = []
    async for evt in mem.discard_tool_messages():
        events.append(evt)

    assert len(events) == 2
    assert isinstance(events[0], CompactionStart)
    assert events[0].dropped_message_count == 1
    assert isinstance(events[-1], CompactionEnd)
    assert events[-1].error is None
    assert events[-1].dropped_message_count == 1

    # Live buffer: tool message removed
    live_msgs = mem.get_all_messages()
    tool_msgs = [m for m in live_msgs if m.get("role") == "tool"]
    assert tool_msgs == [], "tool message should be removed from live buffer"
    roles = [m.get("role") for m in live_msgs]
    assert roles == ["user", "assistant", "user"]

    # Replay history: tool message preserved
    replay = mem.get_replay_messages()
    replay_tool = [m for m in replay if m.get("role") == "tool"]
    assert len(replay_tool) == 1
    assert replay_tool[0]["content"] == "sunny, 25°C"


@pytest.mark.asyncio
async def test_discard_tool_multiple_messages() -> None:
    """Multiple tool messages in one round are all discarded."""
    mem = ConversationMemory()
    await mem.add_message(_user("q"))
    await mem.add_message(
        _assistant(
            "thinking",
            [
                ToolCall(
                    id="c1",
                    type="function",
                    function={"name": "tool1", "arguments": "{}"},
                ),
                ToolCall(
                    id="c2",
                    type="function",
                    function={"name": "tool2", "arguments": "{}"},
                ),
            ],
        )
    )
    await mem.add_message(_tool("c1", "result1"))
    await mem.add_message(_tool("c2", "result2"))

    events: list[Any] = []
    async for evt in mem.discard_tool_messages():
        events.append(evt)

    assert len(events) == 2
    assert events[0].dropped_message_count == 2
    assert events[-1].dropped_message_count == 2

    live_msgs = mem.get_all_messages()
    tool_msgs = [m for m in live_msgs if m.get("role") == "tool"]
    assert tool_msgs == [], "all tool messages removed"

    replay = mem.get_replay_messages()
    replay_tool = [m for m in replay if m.get("role") == "tool"]
    assert len(replay_tool) == 2


@pytest.mark.asyncio
async def test_discard_tool_only_affects_forward_buffer() -> None:
    """Messages before _forward_offset are not touched."""
    mem = ConversationMemory()
    await mem.add_message(_user("earlier"))
    await mem.add_message(_assistant("reply"))
    mem._forward_offset = 2
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
    await mem.add_message(_tool("c1", "data"))

    events: list[Any] = []
    async for evt in mem.discard_tool_messages():
        events.append(evt)

    assert len(events) == 2
    assert events[0].dropped_message_count == 1

    live = mem.get_all_messages()
    assert live[0]["role"] == "user"
    assert live[0]["content"] == [{"type": "text", "text": "earlier"}]
    assert live[1]["role"] == "assistant"


# ---------------------------------------------------------------------------
# ToolCompactionAgent integration tests -- discard behaviour
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
async def test_agent_no_tools_no_discard() -> None:
    """No tool calls -- no discard events at all."""
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
    )
    mem = ConversationMemory()
    events = await _collect(agent, [{"type": "text", "text": "hi"}], mem, [])
    compaction_events = [
        e for e in events if isinstance(e, (CompactionStart, CompactionEnd))
    ]
    assert compaction_events == []


@pytest.mark.asyncio
async def test_agent_discard_tool_after_execution() -> None:
    """Tool results are discarded from forward buffer after execution."""
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

            yield ToolProgress(chunk=f"echoing {args['x']}", tool_call=tc)
            yield ToolEnd(
                result=ToolResult(content="big-result-" * 100), tool_call=tc
            )

    agent = ToolCompactionAgent(
        llm_provider=provider,
        summarizer=_streaming_summarizer,
    )
    mem = ConversationMemory()
    events = await _collect(
        agent, [{"type": "text", "text": "do it"}], mem, [EchoTool()]
    )

    starts = [e for e in events if isinstance(e, CompactionStart)]
    ends = [e for e in events if isinstance(e, CompactionEnd)]
    assert len(starts) >= 1, "expected at least one CompactionStart from discard"
    assert len(ends) >= 1, "expected at least one CompactionEnd from discard"
    assert ends[-1].error is None

    # Tool messages should NOT be visible via ``get_forward_messages()``
    # (LLM context), but SHOULD be in ``get_all_messages()`` so they
    # can be persisted by ``save_memory()`` and displayed after refresh.
    forward_tool = [m for m in mem.get_forward_messages() if m.get("role") == "tool"]
    assert forward_tool == [], "tool messages should be hidden from forward buffer"

    all_tool = [m for m in mem.get_all_messages() if m.get("role") == "tool"]
    assert len(all_tool) == 1, "tool messages should be preserved in all_messages"

    # SHOULD be in replay history too
    replay = mem.get_replay_messages()
    replay_tool = [m for m in replay if m.get("role") == "tool"]
    assert len(replay_tool) == 1, "tool message should be preserved in replay"

    agent_ends = [e for e in events if isinstance(e, AgentEnd)]
    assert len(agent_ends) == 1
    assert agent_ends[0].error is None
    assert agent_ends[0].response == "done"


@pytest.mark.asyncio
async def test_agent_discard_multiple_rounds() -> None:
    """Multiple rounds: each round's tool results are discarded independently."""
    provider = ScriptedLLMProvider(
        [
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
    )
    mem = ConversationMemory()
    events = await _collect(
        agent, [{"type": "text", "text": "do it"}], mem, [EchoTool()]
    )

    ends = [e for e in events if isinstance(e, CompactionEnd)]
    assert len(ends) >= 1
    assert ends[-1].error is None

    # Tool messages should be hidden from forward (LLM context) but
    # preserved in all_messages for persistence.
    forward_tool = [m for m in mem.get_forward_messages() if m.get("role") == "tool"]
    assert forward_tool == [], "tool messages hidden from forward buffer"

    all_tool = [m for m in mem.get_all_messages() if m.get("role") == "tool"]
    assert len(all_tool) == 2, "tool messages preserved in all_messages for persistence"

    agent_ends = [e for e in events if isinstance(e, AgentEnd)]
    assert len(agent_ends) == 1
    assert agent_ends[0].error is None
    assert agent_ends[0].response == "r3 final"


# ---------------------------------------------------------------------------
# ToolCompactionAgent integration tests -- full conversation compaction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_compaction_disabled_by_default() -> None:
    """threshold=0 -- no compaction events."""
    provider = ScriptedLLMProvider(
        [
            LLMResponse(
                content="hello",
                reasoning_content=None,
                tool_calls=[],
                finish_reason="stop",
                usage={"prompt_tokens": 100, "completion_tokens": 3, "total_tokens": 103},
            ),
        ]
    )
    agent = ToolCompactionAgent(
        llm_provider=provider,
        summarizer=_streaming_summarizer,
        prompt_token_threshold=0,
    )
    mem = ConversationMemory()
    events = await _collect(agent, [{"type": "text", "text": "hi"}], mem, [])
    compaction_events = [
        e for e in events if isinstance(e, (CompactionStart, CompactionEnd))
    ]
    assert compaction_events == []


@pytest.mark.asyncio
async def test_agent_compaction_exceeds_threshold() -> None:
    """Token usage exceeds threshold -- compaction events + MessageEvent."""
    provider = ScriptedLLMProvider(
        [
            LLMResponse(
                content="hello back",
                reasoning_content=None,
                tool_calls=[],
                finish_reason="stop",
                usage={"prompt_tokens": 500, "completion_tokens": 3, "total_tokens": 503},
            ),
        ]
    )
    agent = ToolCompactionAgent(
        llm_provider=provider,
        summarizer=_streaming_summarizer,
        prompt_token_threshold=100,
        keep_recent=2,
    )
    mem = ConversationMemory()
    await mem.add_message(_user("earlier1"))
    await mem.add_message(_assistant("reply1"))
    await mem.add_message(_user("earlier2"))
    await mem.add_message(_assistant("reply2"))

    events = await _collect(agent, [{"type": "text", "text": "hi"}], mem, [])

    starts = [e for e in events if isinstance(e, CompactionStart)]
    ends = [e for e in events if isinstance(e, CompactionEnd)]
    assert len(starts) >= 1, "expected compaction start"
    assert len(ends) >= 1, "expected compaction end"
    assert ends[-1].error is None
    assert ends[-1].dropped_message_count > 0

    # Should have MessageEvent for compaction
    from minimal_harness.types import MessageEvent
    comp_event = [
        e for e in events
        if isinstance(e, MessageEvent) and e.message.get("role") == "compaction"
    ]
    assert len(comp_event) >= 1

    # Agent completes successfully
    agent_ends = [e for e in events if isinstance(e, AgentEnd)]
    assert len(agent_ends) == 1
    assert agent_ends[0].error is None


@pytest.mark.asyncio
async def test_agent_forwards_events_to_middleware() -> None:
    """Discard events are forwarded to middleware hooks."""

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
        middleware=[recorder],
    )
    mem = ConversationMemory()
    await _collect(agent, [{"type": "text", "text": "do it"}], mem, [EchoTool()])

    assert len(recorder.starts) >= 1
    assert len(recorder.ends) >= 1
    assert recorder.ends[-1].error is None
