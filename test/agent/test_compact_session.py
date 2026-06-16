"""Tests for ``AgentRuntime.compact_session`` — the manual-compact path
that backs the ``/compact`` slash command.

These tests verify that:

1. The runtime surfaces the same ``CompactionStart / CompactionChunk /
   CompactionEnd`` event stream the agent loop produces.
2. The summarizer and LLM provider come from the runtime's configured
   ``compaction_summarizer_factory`` and ``llm_provider_resolver``.
3. The threshold/keep_recent fall back to the runtime's defaults when
   the session's agent has no ``CompactionSettings``.
4. Per-agent ``CompactionSettings`` override the runtime defaults.
5. The method refuses to do anything if no summarizer factory is
   configured.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Sequence
from unittest.mock import AsyncMock, MagicMock

import pytest
from minimal_harness.agent.runtime import AgentRuntime
from minimal_harness.llm.llm import LLMChunkDelta, LLMResponse, Stream
from minimal_harness.memory import (
    ConversationMemory,
    Memory,
    Message,
    assistant_message,
    user_message,
)
from minimal_harness.types import (
    AgentMetadata,
    CompactionChunk,
    CompactionEnd,
    CompactionStart,
)

# ── helpers ────────────────────────────────────────────────────────


class _StubLLMProvider:
    """A no-op LLMProvider — the summarizer is what we actually test."""

    def __init__(self) -> None:
        self.chat_calls: list[list[Message]] = []

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[Any] = (),
        stop_event: asyncio.Event | None = None,
        **kwargs: Any,
    ) -> Stream[LLMChunkDelta]:
        self.chat_calls.append(list(messages))

        async def _gen() -> AsyncIterator[Any]:
            yield LLMResponse(
                content="",
                reasoning_content=None,
                tool_calls=[],
                finish_reason="stop",
                usage=None,
            )

        return Stream(_gen())


async def _streaming_summarizer(
    messages: list[Message], existing: str | None
) -> AsyncIterator[str]:
    for w in ["alpha", " beta", " gamma"]:
        yield w


def _make_runtime(
    summarizer_factory: Any | None = None,
    default_settings: Any | None = None,
    agent_metadata: AgentMetadata | None = None,
    memory: Memory | None = None,
) -> AgentRuntime:
    """Build an AgentRuntime wired to a stub session store + agent registry."""

    # Agent registry
    if agent_metadata is None:
        agent_metadata = AgentMetadata(name="compacting_assistant")
    reg = MagicMock()
    reg.get = AsyncMock(return_value=agent_metadata)

    # Session store
    inner_memory = memory or ConversationMemory()

    class _StubSession:
        """A session-shaped object that delegates Memory operations to
        a real ConversationMemory instance."""

        def __init__(self, inner: Memory, agent_name: str) -> None:
            self.memory = inner
            self.agent_name = agent_name

        def get_message_usage(self) -> Any:
            return self.memory.get_message_usage()

        def compact(
            self,
            summarizer: Any,
            keep_recent: int,
            total_tokens: int = 0,
        ) -> Any:
            return self.memory.compact(
                summarizer, keep_recent, total_tokens=total_tokens
            )

        def reset_message_usage(self) -> None:
            self.memory.reset_message_usage()

    session = _StubSession(inner_memory, agent_metadata.name)
    store = MagicMock()
    store.get_session = AsyncMock(return_value=session)

    # Tool registry (unused by compact_session)
    tool_reg = MagicMock()

    runtime = AgentRuntime(
        agent_registry=reg,
        session_store=store,
        tool_registry=tool_reg,
        llm_provider_resolver=lambda _meta: _StubLLMProvider(),
        compaction_summarizer_factory=summarizer_factory,
        default_compaction_settings=default_settings,
    )
    return runtime, inner_memory


# ── tests ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compact_session_yields_canonical_event_stream() -> None:
    """compact_session must yield the exact same event sequence the
    CompactionAgent loop yields, in order: 1× Start, N× Chunk, 1× End.
    """
    inner = ConversationMemory()
    for i in range(8):
        await inner.add_message(user_message([{"type": "text", "text": f"q{i}"}]))
        await inner.add_message(assistant_message(f"a{i}"))

    runtime, _ = _make_runtime(
        summarizer_factory=lambda _llm: _streaming_summarizer,
        default_settings={
            "prompt_token_threshold": 100,
            "keep_recent": 2,
        },
        memory=inner,
    )

    events: list[Any] = []
    async for evt in runtime.compact_session("session-1"):
        events.append(evt)

    assert len(events) == 5  # 1 Start + 3 Chunks + 1 End
    assert isinstance(events[0], CompactionStart)
    assert all(isinstance(e, CompactionChunk) for e in events[1:-1])
    assert isinstance(events[-1], CompactionEnd)
    assert events[-1].summary == "alpha beta gamma"
    assert events[-1].error is None


@pytest.mark.asyncio
async def test_compact_session_uses_runtime_defaults_when_agent_has_no_settings() -> (
    None
):
    """When the agent's ``CompactionSettings`` is None, the runtime
    must use its ``default_compaction_settings`` (with its own
    fallbacks) so the fold still works for plain ``agent_type=simple``
    agents."""
    inner = ConversationMemory()
    for i in range(6):
        await inner.add_message(user_message([{"type": "text", "text": f"q{i}"}]))
        await inner.add_message(assistant_message(f"a{i}"))

    agent = AgentMetadata(name="plain", agent_type="simple", compaction=None)
    runtime, _ = _make_runtime(
        summarizer_factory=lambda _llm: _streaming_summarizer,
        default_settings={
            "prompt_token_threshold": 100,
            "keep_recent": 1,
        },
        agent_metadata=agent,
        memory=inner,
    )

    events: list[Any] = []
    async for evt in runtime.compact_session("s"):
        events.append(evt)

    end = next(e for e in events if isinstance(e, CompactionEnd))
    assert end.dropped_message_count == 11  # 12 - 1 recent
    assert end.summary == "alpha beta gamma"


@pytest.mark.asyncio
async def test_compact_session_per_agent_settings_override_defaults() -> None:
    """The agent's CompactionSettings must take precedence over the
    runtime defaults. We test that by giving the agent ``keep_recent=4``
    while the runtime default is ``keep_recent=1`` — the fold should
    keep 4 messages, not 1.
    """
    inner = ConversationMemory()
    for i in range(8):
        await inner.add_message(user_message([{"type": "text", "text": f"q{i}"}]))
        await inner.add_message(assistant_message(f"a{i}"))

    agent = AgentMetadata(
        name="compacting",
        agent_type="compacting",
        compaction={"prompt_token_threshold": 100, "keep_recent": 4},
    )
    runtime, _ = _make_runtime(
        summarizer_factory=lambda _llm: _streaming_summarizer,
        default_settings={"prompt_token_threshold": 100, "keep_recent": 1},
        agent_metadata=agent,
        memory=inner,
    )

    events: list[Any] = []
    async for evt in runtime.compact_session("s"):
        events.append(evt)

    end = next(e for e in events if isinstance(e, CompactionEnd))
    assert end.dropped_message_count == 12  # 16 - 4 recent
    # Full history preserved.
    assert len(inner.get_replay_messages()) == 17  # 16 + 1 summary


@pytest.mark.asyncio
async def test_compact_session_raises_when_no_summarizer_factory() -> None:
    """Without a ``compaction_summarizer_factory``, the runtime must
    refuse to compact — silent fallbacks would mask configuration
    errors and produce empty summaries.
    """
    runtime, _ = _make_runtime(
        summarizer_factory=None,
        default_settings={"keep_recent": 2},
    )

    with pytest.raises(RuntimeError, match="compaction_summarizer_factory"):
        async for _ in runtime.compact_session("s"):
            pass


@pytest.mark.asyncio
async def test_compact_session_propagates_summarizer_error() -> None:
    """If the summarizer raises, ``compact_session`` must surface
    ``CompactionEnd(error=...)`` (same contract as the agent loop).
    The buffer is untouched.
    """

    async def _failing(
        messages: list[Message], existing: str | None
    ) -> AsyncIterator[str]:
        yield "partial-"
        raise RuntimeError("summarizer down")

    inner = ConversationMemory()
    for i in range(6):
        await inner.add_message(user_message([{"type": "text", "text": f"q{i}"}]))
        await inner.add_message(assistant_message(f"a{i}"))
    before = [dict(m) for m in inner.get_replay_messages()]

    runtime, _ = _make_runtime(
        summarizer_factory=lambda _llm: _failing,
        default_settings={"keep_recent": 1},
        memory=inner,
    )

    events: list[Any] = []
    async for evt in runtime.compact_session("s"):
        events.append(evt)

    ends = [e for e in events if isinstance(e, CompactionEnd)]
    assert len(ends) == 1
    assert ends[0].error is not None
    assert "summarizer down" in ends[0].error
    assert ends[0].summary == ""
    # Buffer is untouched on failure.
    assert [dict(m) for m in inner.get_replay_messages()] == before
