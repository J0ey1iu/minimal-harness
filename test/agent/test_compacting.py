from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Sequence

import pytest

from minimal_harness.agent.compacting import CompactionAgent
from minimal_harness.agent.middleware import Middleware
from minimal_harness.llm.llm import LLMResponse, Stream
from minimal_harness.memory import (
    ConversationMemory,
    Message,
    assistant_message,
    user_message,
)
from minimal_harness.types import (
    AgentEvent,
    CompactionChunk,
    CompactionEnd,
    CompactionStart,
    LLMChunkDelta,
    TokenUsage,
    ToolCall,
)

# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------


def _user(text: str) -> Message:
    return user_message([{"type": "text", "text": text}])


def _assistant(text: str) -> Message:
    return assistant_message(text, None)


async def _stream_response(
    content: str | None = None,
    tool_calls: list[ToolCall] | None = None,
    usage: TokenUsage | None = None,
) -> Stream[LLMChunkDelta]:
    """Return a Stream that yields a single LLMResponse with no chunks."""

    async def _gen() -> AsyncIterator[Any]:
        yield LLMResponse(
            content=content,
            reasoning_content=None,
            tool_calls=tool_calls or [],
            finish_reason="stop",
            usage=usage,
        )

    return Stream(_gen())


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
            # Yield a single delta then the LLMResponse so the Stream captures it.
            if resp.content:
                yield LLMChunkDelta(content=resp.content)
            yield resp

        return Stream(_gen())


async def _noop_summarizer(
    messages: list[Message], existing_summary: str | None
) -> AsyncIterator[str]:
    yield f"[summary of {len(messages)} msgs, prior={existing_summary}]"


async def _streaming_summarizer(
    messages: list[Message], existing_summary: str | None
) -> AsyncIterator[str]:
    for word in ["alpha", " beta", " gamma"]:
        yield word


async def _failing_summarizer(
    messages: list[Message], existing_summary: str | None
) -> AsyncIterator[str]:
    yield "partial-"
    raise RuntimeError("summarizer blew up")


def _populate(memory: ConversationMemory, n_turns: int) -> None:
    """Add n_turns of (user, assistant) pairs to memory."""
    for i in range(n_turns):
        asyncio.get_event_loop().run_until_complete(memory.add_message(_user(f"q{i}")))
        asyncio.get_event_loop().run_until_complete(
            memory.add_message(_assistant(f"a{i}"))
        )


@pytest.fixture
def populated_memory() -> ConversationMemory:
    mem = ConversationMemory()
    for i in range(10):
        asyncio.get_event_loop().run_until_complete(mem.add_message(_user(f"q{i}")))
        asyncio.get_event_loop().run_until_complete(
            mem.add_message(_assistant(f"a{i}"))
        )
    return mem


# ---------------------------------------------------------------------------
# Memory.compact() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compact_yields_no_events_when_nothing_to_fold() -> None:
    mem = ConversationMemory()
    await mem.add_message(_user("only one"))
    events = [e async for e in mem.compact(_noop_summarizer, keep_recent=4)]
    assert events == []
    assert len(mem.get_all_messages()) == 1


@pytest.mark.asyncio
async def test_compact_emits_start_chunk_end_in_order(populated_memory) -> None:
    mem = populated_memory
    events: list[Any] = []
    async for evt in mem.compact(
        _streaming_summarizer, keep_recent=4, prompt_tokens=9000
    ):
        events.append(evt)

    assert len(events) == 5  # 1 Start + 3 Chunks + 1 End
    assert isinstance(events[0], CompactionStart)
    assert all(isinstance(e, CompactionChunk) for e in events[1:-1])
    assert isinstance(events[-1], CompactionEnd)

    start = events[0]
    assert start.dropped_message_count == 16  # 20 total - 4 recent
    assert start.existing_summary is None
    assert start.keep_recent == 4
    assert start.prompt_tokens == 9000

    end = events[-1]
    assert end.error is None
    assert end.dropped_message_count == 16
    assert end.new_offset == 0  # offset stays at 0; summary is the natural start
    assert end.summary == "alpha beta gamma"


@pytest.mark.asyncio
async def test_compact_accumulates_streamed_chunks(populated_memory) -> None:
    mem = populated_memory
    chunks: list[CompactionChunk] = []
    async for evt in mem.compact(_streaming_summarizer, keep_recent=4):
        if isinstance(evt, CompactionChunk):
            chunks.append(evt)
    accumulated = [c.accumulated for c in chunks]
    assert accumulated == ["alpha", "alpha beta", "alpha beta gamma"]


@pytest.mark.asyncio
async def test_compact_replaces_buffer_with_summary_plus_tail(populated_memory) -> None:
    mem = populated_memory
    async for _ in mem.compact(_streaming_summarizer, keep_recent=4):
        pass

    msgs = mem.get_all_messages()
    # Summary is a CompactionMessage at index 0 with a `meta` field.
    assert msgs[0]["role"] == "compaction"
    assert msgs[0]["content"] == "alpha beta gamma"
    assert msgs[0].get("meta") is not None
    assert msgs[0]["meta"]["dropped_count"] == 16
    # Tail of 4 = last 4 messages of the original 20
    assert len(msgs) == 5
    assert msgs[-1] == _assistant("a9")
    assert msgs[-2] == _user("q9")


@pytest.mark.asyncio
async def test_compact_offset_stays_at_zero(populated_memory) -> None:
    """After compaction, offset stays at 0: the summary is the natural
    start of the compacted conversation, no message is skipped."""
    mem = populated_memory
    assert mem._forward_offset == 0
    async for _ in mem.compact(_noop_summarizer, keep_recent=4):
        pass
    assert mem._forward_offset == 0


@pytest.mark.asyncio
async def test_compact_get_forward_messages_includes_summary(populated_memory) -> None:
    mem = populated_memory
    async for _ in mem.compact(_streaming_summarizer, keep_recent=4):
        pass

    # get_all_messages() returns the raw storage — summary is a
    # CompactionMessage (role="compaction") at index 0.
    raw = mem.get_all_messages()
    assert raw[0]["role"] == "compaction"
    assert raw[0]["content"] == "alpha beta gamma"

    # get_forward_messages() is what the LLM sees — the summary is
    # re-projected to role="assistant" so it looks like a normal
    # historical turn.
    forward = mem.get_forward_messages()
    assert forward[0]["role"] == "assistant"
    assert forward[0]["content"] == "alpha beta gamma"
    # Must NOT carry the `meta` field of CompactionMessage — that's
    # internal storage metadata, not LLM-visible.
    assert "meta" not in forward[0]
    # 1 summary + 4 recent = 5
    assert len(forward) == 5


@pytest.mark.asyncio
async def test_compact_second_call_passes_existing_summary() -> None:
    mem = ConversationMemory()
    received_existing: list[str | None] = []

    async def tracking_summarizer(
        messages: list[Message], existing_summary: str | None
    ) -> AsyncIterator[str]:
        received_existing.append(existing_summary)
        yield f"v1: {len(messages)}"

    # First compaction: 20 msgs, keep_recent=4 → drop msgs[0:16]
    for i in range(10):
        await mem.add_message(_user(f"q{i}"))
        await mem.add_message(_assistant(f"a{i}"))
    async for _ in mem.compact(tracking_summarizer, keep_recent=4):
        pass
    # After 1st comp: msgs = [compaction_summary, q8, a8, q9, a9], offset=0

    # Add more messages so the second compaction has something to fold
    for i in range(10, 20):
        await mem.add_message(_user(f"q{i}"))
        await mem.add_message(_assistant(f"a{i}"))
    # Now msgs = [compaction_summary, q8, a8, q9, a9, q10, ..., q19, a19] = 21 msgs

    async for _ in mem.compact(tracking_summarizer, keep_recent=4):
        pass

    assert received_existing[0] is None  # first call: no prior summary
    assert received_existing[1] is not None  # second call: prior summary present
    assert received_existing[1].startswith("v1:")


@pytest.mark.asyncio
async def test_compact_failure_keeps_buffer_intact() -> None:
    mem = ConversationMemory()
    for i in range(10):
        await mem.add_message(_user(f"q{i}"))
        await mem.add_message(_assistant(f"a{i}"))
    before = [dict(m) for m in mem.get_all_messages()]

    end_evt: CompactionEnd | None = None
    async for evt in mem.compact(_failing_summarizer, keep_recent=4):
        if isinstance(evt, CompactionEnd):
            end_evt = evt

    assert end_evt is not None
    assert end_evt.error is not None
    assert "RuntimeError" in end_evt.error
    assert end_evt.dropped_message_count == 0
    assert end_evt.new_offset == 0  # offset unchanged
    after = [dict(m) for m in mem.get_all_messages()]
    assert before == after


@pytest.mark.asyncio
async def test_compact_survives_dump_load_cycle() -> None:
    mem = ConversationMemory()
    for i in range(8):
        await mem.add_message(_user(f"q{i}"))
        await mem.add_message(_assistant(f"a{i}"))
    async for _ in mem.compact(_streaming_summarizer, keep_recent=4):
        pass

    raw = mem.get_all_messages()
    assert raw[0]["role"] == "compaction"
    assert raw[0]["content"] == "alpha beta gamma"
    assert raw[0].get("meta") is not None

    dumped = mem.dump_memory()
    new_mem = ConversationMemory()
    new_mem.load_memory(dumped)

    # Round-trip: raw storage keeps the CompactionMessage; offset is 0.
    raw_after = new_mem.get_all_messages()
    assert raw_after[0]["role"] == "compaction"
    assert raw_after[0]["content"] == "alpha beta gamma"
    assert new_mem._forward_offset == 0
    # LLM view re-projects the compaction to role="assistant".
    forward = new_mem.get_forward_messages()
    assert forward[0]["role"] == "assistant"
    assert forward[0]["content"] == "alpha beta gamma"
    assert len(forward) == 5  # summary + 4 recent


# ---------------------------------------------------------------------------
# CompactionAgent tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_does_not_compact_under_threshold() -> None:
    summary_called = 0

    async def summarizer(
        messages: list[Message], existing: str | None
    ) -> AsyncIterator[str]:
        nonlocal summary_called
        summary_called += 1
        yield "x"

    provider = ScriptedLLMProvider(
        [
            LLMResponse(
                content="hi",
                reasoning_content=None,
                tool_calls=[],
                finish_reason="stop",
                usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 5,
                    "total_tokens": 105,
                },
            ),
        ]
    )
    agent = CompactionAgent(
        llm_provider=provider,
        summarizer=summarizer,
        prompt_token_threshold=8000,
        max_iterations=1,
    )
    memory = ConversationMemory()
    events: list[AgentEvent] = []
    async for evt in agent.run(
        user_input=[{"type": "text", "text": "hi"}], memory=memory, tools=[]
    ):
        events.append(evt)

    assert summary_called == 0
    assert not any(
        isinstance(e, (CompactionStart, CompactionChunk, CompactionEnd)) for e in events
    )


@pytest.mark.asyncio
async def test_agent_triggers_compaction_over_threshold() -> None:
    summary_called = 0

    async def summarizer(
        messages: list[Message], existing: str | None
    ) -> AsyncIterator[str]:
        nonlocal summary_called
        summary_called += 1
        yield "compacted!"

    provider = ScriptedLLMProvider(
        [
            LLMResponse(
                content="done",
                reasoning_content=None,
                tool_calls=[],
                finish_reason="stop",
                usage={
                    "prompt_tokens": 9000,
                    "completion_tokens": 5,
                    "total_tokens": 9005,
                },
            ),
        ]
    )
    agent = CompactionAgent(
        llm_provider=provider,
        summarizer=summarizer,
        prompt_token_threshold=8000,
        keep_recent=2,
        max_iterations=1,
    )
    memory = ConversationMemory()
    # Pre-populate with enough messages to compact
    for i in range(6):
        await memory.add_message(_user(f"q{i}"))
        await memory.add_message(_assistant(f"a{i}"))
    memory.set_message_usage(
        {"prompt_tokens": 9000, "completion_tokens": 0, "total_tokens": 9000}
    )

    events: list[AgentEvent] = []
    async for evt in agent.run(
        user_input=[{"type": "text", "text": "go"}], memory=memory, tools=[]
    ):
        events.append(evt)

    assert summary_called == 1
    starts = [e for e in events if isinstance(e, CompactionStart)]
    ends = [e for e in events if isinstance(e, CompactionEnd)]
    assert len(starts) == 1
    assert len(ends) == 1
    assert starts[0].prompt_tokens == 9000
    assert ends[0].summary == "compacted!"

    # The summary must also be surfaced as a MessageEvent so the
    # TUI and other frontends can persist/replay it. The role is
    # "compaction" — the frontend can re-project it to assistant for
    # the LLM but must preserve the original form for storage.
    from minimal_harness.types import MessageEvent

    summary_events = [
        e
        for e in events
        if isinstance(e, MessageEvent) and e.message.get("role") == "compaction"
    ]
    assert len(summary_events) == 1
    assert summary_events[0].message["content"] == "compacted!"
    assert summary_events[0].message.get("meta") is not None


@pytest.mark.asyncio
async def test_assistant_message_can_be_folded_by_same_turn_compaction() -> None:
    """When the buffer is so large that even the just-added assistant
    turn cannot fit in the ``keep_recent`` tail, compaction folds it
    into the new summary. The frontend still saw the raw assistant
    text via ``MessageEvent``; only what the NEXT LLM call sees is
    compacted. The buffer's invariant is: after the turn (if compact
    succeeded), it's below threshold, with at most ``keep_recent``
    recent messages plus one CompactionMessage at msgs[0].
    """

    summary_called = 0
    summarized_messages: list[Message] = []

    async def summarizer(
        messages: list[Message], existing: str | None
    ) -> AsyncIterator[str]:
        nonlocal summary_called
        summary_called += 1
        summarized_messages.extend(messages)
        yield "compacted!"

    provider = ScriptedLLMProvider(
        [
            LLMResponse(
                content="the-just-added-reply",
                reasoning_content=None,
                tool_calls=[],
                finish_reason="stop",
                usage={
                    "prompt_tokens": 9000,
                    "completion_tokens": 5,
                    "total_tokens": 9005,
                },
            ),
        ]
    )
    agent = CompactionAgent(
        llm_provider=provider,
        summarizer=summarizer,
        prompt_token_threshold=8000,
        keep_recent=0,  # aggressive: fold EVERYTHING, including the new assistant
        max_iterations=1,
    )
    memory = ConversationMemory()
    for i in range(6):
        await memory.add_message(_user(f"q{i}"))
        await memory.add_message(_assistant(f"a{i}"))
    memory.set_message_usage(
        {"prompt_tokens": 9000, "completion_tokens": 0, "total_tokens": 9000}
    )

    from minimal_harness.types import MessageEvent

    events: list = []
    async for evt in agent.run(
        user_input=[{"type": "text", "text": "go"}], memory=memory, tools=[]
    ):
        events.append(evt)

    # The summarizer was invoked. The user_input message AND the
    # just-added assistant message were both fed to the summarizer
    # (they were in msgs[0:end-0] = msgs[0:end] = everything).
    assert summary_called == 1
    folded_contents = [m.get("content") for m in summarized_messages]
    assert any("go" in str(c) for c in folded_contents if c)
    # The assistant reply itself was also in the fold region.
    assert any("the-just-added-reply" in str(c) for c in folded_contents if c)

    # Event order: msg.assistant first (primary content), then
    # compact events, then msg.compaction, then agent.end.
    assistant_idx = next(
        i
        for i, e in enumerate(events)
        if isinstance(e, MessageEvent) and e.message.get("role") == "assistant"
    )
    compact_start_idx = next(
        i for i, e in enumerate(events) if isinstance(e, CompactionStart)
    )
    assert assistant_idx < compact_start_idx

    # Frontend must have seen the raw assistant text via MessageEvent.
    assistant_events = [
        e
        for e in events
        if isinstance(e, MessageEvent) and e.message.get("role") == "assistant"
    ]
    assert len(assistant_events) == 1
    assert assistant_events[0].message.get("content") == "the-just-added-reply"

    # Buffer invariant: with keep_recent=0 the buffer is just the
    # CompactionMessage (the assistant has been folded in).
    msgs = [dict(m) for m in memory.get_all_messages()]
    compaction_msgs = [m for m in msgs if m.get("role") == "compaction"]
    assert len(compaction_msgs) == 1
    assert compaction_msgs[0].get("content") == "compacted!"
    non_compaction = [m for m in msgs if m.get("role") != "compaction"]
    assert non_compaction == []

    # What the NEXT LLM call sees is ONLY the compacted summary
    # (the just-added assistant has been folded in and is gone
    # from the buffer; the user sees it only through the
    # ``MessageEvent`` stream / session replay).
    forwarded = memory.get_forward_messages()
    assert len(forwarded) == 1
    assert forwarded[0].get("role") == "assistant"
    assert forwarded[0].get("content") == "compacted!"


@pytest.mark.asyncio
async def test_agent_does_not_emit_summary_message_event_on_failure() -> None:
    """Failed compaction must NOT emit a MessageEvent for the (non-existent)
    summary, otherwise frontends would render a phantom message."""

    async def failing_summarizer(
        messages: list[Message], existing: str | None
    ) -> AsyncIterator[str]:
        yield "partial-"
        raise RuntimeError("summarizer died")

    provider = ScriptedLLMProvider(
        [
            LLMResponse(
                content="ok",
                reasoning_content=None,
                tool_calls=[],
                finish_reason="stop",
                usage={
                    "prompt_tokens": 9000,
                    "completion_tokens": 1,
                    "total_tokens": 9001,
                },
            ),
        ]
    )
    agent = CompactionAgent(
        llm_provider=provider,
        summarizer=failing_summarizer,
        prompt_token_threshold=8000,
        keep_recent=2,
        max_iterations=1,
    )
    memory = ConversationMemory()
    for i in range(6):
        await memory.add_message(_user(f"q{i}"))
        await memory.add_message(_assistant(f"a{i}"))

    events: list[AgentEvent] = []
    async for evt in agent.run(
        user_input=[{"type": "text", "text": "go"}], memory=memory, tools=[]
    ):
        events.append(evt)

    from minimal_harness.types import MessageEvent

    summary_events = [
        e
        for e in events
        if isinstance(e, MessageEvent) and e.message.get("role") == "compaction"
    ]
    assert summary_events == [], "failed compaction must not emit a system MessageEvent"


@pytest.mark.asyncio
async def test_agent_forwards_compaction_events_to_middleware() -> None:
    class Recorder(Middleware):
        def __init__(self) -> None:
            self.starts: list[CompactionStart] = []
            self.ends: list[CompactionEnd] = []

        async def on_compaction_start(self, event: CompactionStart) -> None:
            self.starts.append(event)

        async def on_compaction_end(self, event: CompactionEnd) -> None:
            self.ends.append(event)

    async def summarizer(
        messages: list[Message], existing: str | None
    ) -> AsyncIterator[str]:
        yield "abc"

    provider = ScriptedLLMProvider(
        [
            LLMResponse(
                content="ok",
                reasoning_content=None,
                tool_calls=[],
                finish_reason="stop",
                usage={
                    "prompt_tokens": 9000,
                    "completion_tokens": 1,
                    "total_tokens": 9001,
                },
            ),
        ]
    )
    recorder = Recorder()
    agent = CompactionAgent(
        llm_provider=provider,
        summarizer=summarizer,
        prompt_token_threshold=8000,
        keep_recent=2,
        max_iterations=1,
        middleware=[recorder],
    )
    memory = ConversationMemory()
    for i in range(6):
        await memory.add_message(_user(f"q{i}"))
        await memory.add_message(_assistant(f"a{i}"))
    memory.set_message_usage(
        {"prompt_tokens": 9000, "completion_tokens": 0, "total_tokens": 9000}
    )

    async for _ in agent.run(
        user_input=[{"type": "text", "text": "go"}], memory=memory, tools=[]
    ):
        pass

    assert len(recorder.starts) == 1
    assert len(recorder.ends) == 1
    assert recorder.ends[0].summary == "abc"


@pytest.mark.asyncio
async def test_agent_raises_on_compaction_failure() -> None:
    async def failing_summarizer(
        messages: list[Message], existing: str | None
    ) -> AsyncIterator[str]:
        yield "partial-"
        raise RuntimeError("summarizer died")

    provider = ScriptedLLMProvider(
        [
            LLMResponse(
                content="ok",
                reasoning_content=None,
                tool_calls=[],
                finish_reason="stop",
                usage={
                    "prompt_tokens": 9000,
                    "completion_tokens": 1,
                    "total_tokens": 9001,
                },
            ),
        ]
    )
    agent = CompactionAgent(
        llm_provider=provider,
        summarizer=failing_summarizer,
        prompt_token_threshold=8000,
        keep_recent=2,
        max_iterations=1,
    )
    memory = ConversationMemory()
    for i in range(6):
        await memory.add_message(_user(f"q{i}"))
        await memory.add_message(_assistant(f"a{i}"))
    memory.set_message_usage(
        {"prompt_tokens": 9000, "completion_tokens": 0, "total_tokens": 9000}
    )

    from minimal_harness.types import AgentEnd

    end_evt: AgentEnd | None = None
    async for evt in agent.run(
        user_input=[{"type": "text", "text": "go"}], memory=memory, tools=[]
    ):
        if isinstance(evt, AgentEnd):
            end_evt = evt

    assert end_evt is not None
    assert end_evt.error is not None
    assert "Compaction failed" in end_evt.error
    assert "summarizer died" in end_evt.error


@pytest.mark.asyncio
async def test_compaction_failure_preserves_assistant_message() -> None:
    """When the summarizer raises, the LLM has already produced a
    response. The agent MUST still record that assistant turn in
    memory and emit its MessageEvent before terminating with
    AgentEnd.error. Otherwise the reply is silently dropped and the
    next turn has no context for what the LLM just said.
    """

    async def failing_summarizer(
        messages: list[Message], existing: str | None
    ) -> AsyncIterator[str]:
        yield "partial-"
        raise RuntimeError("summarizer died")

    provider = ScriptedLLMProvider(
        [
            LLMResponse(
                content="the-real-reply",
                reasoning_content=None,
                tool_calls=[],
                finish_reason="stop",
                usage={
                    "prompt_tokens": 9000,
                    "completion_tokens": 5,
                    "total_tokens": 9005,
                },
            ),
        ]
    )
    agent = CompactionAgent(
        llm_provider=provider,
        summarizer=failing_summarizer,
        prompt_token_threshold=8000,
        keep_recent=2,
        max_iterations=1,
    )
    memory = ConversationMemory()
    for i in range(6):
        await memory.add_message(_user(f"q{i}"))
        await memory.add_message(_assistant(f"a{i}"))
    memory.set_message_usage(
        {"prompt_tokens": 9000, "completion_tokens": 0, "total_tokens": 9000}
    )

    from minimal_harness.types import (
        AgentEnd,
        CompactionEnd,
        MessageEvent,
    )

    events: list = []
    async for evt in agent.run(
        user_input=[{"type": "text", "text": "go"}], memory=memory, tools=[]
    ):
        events.append(evt)

    # 1. Agent ends with the wrapped compaction error
    ends = [e for e in events if isinstance(e, AgentEnd)]
    assert len(ends) == 1
    assert ends[0].error is not None
    assert "Compaction failed" in ends[0].error

    # 2. CompactionEnd reports the error and (critically) an empty
    #    ``summary`` — a partial streaming text from a failed
    #    summarizer is not a valid fold and must not be propagated.
    compaction_ends = [e for e in events if isinstance(e, CompactionEnd)]
    assert len(compaction_ends) == 1
    assert compaction_ends[0].error is not None
    assert compaction_ends[0].summary == ""

    # 3. No MessageEvent(role="compaction") is emitted (the fold did
    #    not actually happen, so the frontend should not render a
    #    "Folded summary" block).
    compaction_msg_events = [
        e
        for e in events
        if isinstance(e, MessageEvent) and e.message.get("role") == "compaction"
    ]
    assert compaction_msg_events == []

    # 4. The LLM's assistant reply IS still emitted as a MessageEvent
    #    so the frontend can show it.
    assistant_msg_events = [
        e
        for e in events
        if isinstance(e, MessageEvent) and e.message.get("role") == "assistant"
    ]
    assert len(assistant_msg_events) == 1
    assert assistant_msg_events[0].message.get("content") == "the-real-reply"

    # 5. The assistant turn IS recorded in memory so the next turn
    #    has full conversation context.
    msgs = [dict(m) for m in memory.get_all_messages()]
    assert any(
        m.get("role") == "assistant" and m.get("content") == "the-real-reply"
        for m in msgs
    ), "assistant reply must be in memory even when compact failed"
    # 6. No fake CompactionMessage was inserted on failure.
    assert not any(m.get("role") == "compaction" for m in msgs)

    # 7. Event order: assistant MessageEvent (primary content) ->
    #    compact events (housekeeping) -> agent.end (terminal error).
    #    The LLM's reply is the first thing the user sees; the
    #    compaction error block follows.
    assistant_idx = next(
        i
        for i, e in enumerate(events)
        if isinstance(e, MessageEvent) and e.message.get("role") == "assistant"
    )
    compaction_end_idx = next(
        i for i, e in enumerate(events) if isinstance(e, CompactionEnd)
    )
    agent_end_idx = next(i for i, e in enumerate(events) if isinstance(e, AgentEnd))
    assert assistant_idx < compaction_end_idx < agent_end_idx


@pytest.mark.asyncio
async def test_agent_ignores_compaction_when_usage_missing() -> None:
    summary_called = 0

    async def summarizer(
        messages: list[Message], existing: str | None
    ) -> AsyncIterator[str]:
        nonlocal summary_called
        summary_called += 1
        yield "x"

    provider = ScriptedLLMProvider(
        [
            LLMResponse(
                content="ok",
                reasoning_content=None,
                tool_calls=[],
                finish_reason="stop",
                usage=None,  # No usage reported
            ),
        ]
    )
    agent = CompactionAgent(
        llm_provider=provider,
        summarizer=summarizer,
        prompt_token_threshold=8000,
        max_iterations=1,
    )
    memory = ConversationMemory()
    async for _ in agent.run(
        user_input=[{"type": "text", "text": "hi"}], memory=memory, tools=[]
    ):
        pass

    assert summary_called == 0
