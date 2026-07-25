"""Tests for ``AgentRuntime.compact_session`` — the manual-compact path
that backs the ``/compact`` slash command.

These tests verify that:

1. The runtime surfaces the same ``CompactionStart / CompactionChunk /
   CompactionEnd`` event stream the agent loop produces.
2. The summarizer uses the LLM provider resolved through the runtime's
   ``llm_provider_resolver``, and its streamed chunks become
   ``CompactionChunk`` events.
3. The chat payload preserves the agent's ``system_prompt``, the
   conversation history (with prior summaries re-projected to
   ``assistant`` turns), and ends with a user-side summary request.
4. The threshold/keep_recent fall back to the runtime's defaults when
   the session's agent has no ``CompactionSettings``.
5. Per-agent ``CompactionSettings`` override the runtime defaults.
6. If the summarizer raises mid-stream, ``compact_session`` surfaces
   ``CompactionEnd(error=...)`` (same contract as the agent loop) and
   the buffer is untouched.
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
    """A configurable no-op LLMProvider used to drive the summarizer.

    Tests populate ``chunks`` to control what the summarizer yields, or
    set ``raise_after`` to make the underlying ``chat`` stream fail
    mid-stream.
    """

    def __init__(
        self,
        chunks: Sequence[str] = (),
        raise_after: int | None = None,
    ) -> None:
        self.chat_calls: list[list[Message]] = []
        self._chunks = list(chunks)
        self._raise_after = raise_after

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[Any] = (),
        stop_event: asyncio.Event | None = None,
        **kwargs: Any,
    ) -> Stream[LLMChunkDelta]:
        self.chat_calls.append(list(messages))

        chunks = self._chunks
        raise_after = self._raise_after

        async def _gen() -> AsyncIterator[Any]:
            for i, text in enumerate(chunks):
                yield LLMChunkDelta(
                    content=text,
                    reasoning=None,
                    tool_calls=[],
                )
                if raise_after is not None and i == raise_after:
                    raise RuntimeError("summarizer down")
            yield LLMResponse(
                content="",
                reasoning_content=None,
                tool_calls=[],
                finish_reason="stop",
                usage=None,
            )

        return Stream(_gen())


def _make_runtime(
    llm_provider: _StubLLMProvider | None = None,
    default_settings: Any | None = None,
    agent_metadata: AgentMetadata | None = None,
    memory: Memory | None = None,
) -> AgentRuntime:
    """Build an AgentRuntime wired to a stub session store + agent registry."""
    if agent_metadata is None:
        agent_metadata = AgentMetadata(name="compacting_assistant")
    reg = MagicMock()
    reg.get = AsyncMock(return_value=agent_metadata)

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

    tool_reg = MagicMock()

    provider = llm_provider or _StubLLMProvider(chunks=["alpha", " beta", " gamma"])
    runtime = AgentRuntime(
        agent_registry=reg,
        session_store=store,
        tool_registry=tool_reg,
        llm_provider_resolver=lambda _meta: provider,
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
async def test_compact_session_chat_payload_preserves_system_prompt_and_history() -> (
    None
):
    """The chat payload must keep the agent's ``system_prompt``
    unchanged, feed the conversation history verbatim (with prior
    ``role="compaction"`` messages re-projected to ``assistant``),
    and end with the user-side summary request.
    """
    from minimal_harness.agent._compaction import DEFAULT_SUMMARY_REQUEST as SUMMARY_REQUEST

    inner = ConversationMemory()
    # Place a prior compaction summary at offset 0 — the canonical
    # location after a fold. ``memory.compact`` extracts it as
    # ``existing_summary`` and the summarizer prepends it as an
    # assistant turn before the conversation slice.
    await inner.add_message(
        {
            "role": "compaction",
            "content": "prior summary text",
            "meta": {"dropped_count": 4, "keep_recent": 6},
        }
    )
    await inner.add_message(user_message([{"type": "text", "text": "q0"}]))
    await inner.add_message(assistant_message("a0"))
    await inner.add_message(user_message([{"type": "text", "text": "q1"}]))
    await inner.add_message(assistant_message("a1"))
    await inner.add_message(user_message([{"type": "text", "text": "q2"}]))
    await inner.add_message(assistant_message("a2"))

    provider = _StubLLMProvider(chunks=["x"])
    agent = AgentMetadata(
        name="compacting",
        agent_type="compacting",
        system_prompt="you are the agent",
        compaction={"prompt_token_threshold": 100, "keep_recent": 2},
    )
    runtime, _ = _make_runtime(
        llm_provider=provider,
        default_settings={"prompt_token_threshold": 100, "keep_recent": 1},
        agent_metadata=agent,
        memory=inner,
    )

    events: list[Any] = []
    async for evt in runtime.compact_session("s"):
        events.append(evt)

    assert isinstance(events[-1], CompactionEnd)
    # Exactly one LLM call was made.
    assert len(provider.chat_calls) == 1
    payload = provider.chat_calls[0]

    # 1) system prompt comes first, unchanged.
    assert payload[0] == {"role": "system", "content": "you are the agent"}

    # 2) prior compaction summary is re-projected to an assistant turn
    #    BEFORE the conversation slice (matches get_forward_messages).
    assert payload[1] == {"role": "assistant", "content": "prior summary text"}

    # 3) The slice is appended verbatim (no JSON-dump). We expect the
    #    user/assistant chain from after the prior summary up to
    #    keep_recent=2.
    history_roles = [m["role"] for m in payload[2:-1]]
    assert history_roles == ["user", "assistant", "user", "assistant"]

    # 4) The last message is the user-side summary request, and no
    #    role="compaction" leaks into the slice (they are all
    #    re-projected to assistant).
    assert payload[-1]["role"] == "user"
    assert payload[-1]["content"] == SUMMARY_REQUEST
    assert all(m["role"] != "compaction" for m in payload[1:-1])


@pytest.mark.asyncio
async def test_compact_session_skips_messages_before_prior_compaction() -> None:
    """After a fold, the prior compaction summary is the only
    representation of pre-compaction history visible to the LLM.
    Messages that lived before the prior summary must not be sent —
    only the prior summary text + the messages added since then.
    """
    from minimal_harness.agent._compaction import DEFAULT_SUMMARY_REQUEST as SUMMARY_REQUEST

    inner = ConversationMemory()
    # A prior compaction summary at offset 0.
    await inner.add_message(
        {
            "role": "compaction",
            "content": "prior summary text",
            "meta": {"dropped_count": 4, "keep_recent": 2},
        }
    )
    # Four messages added since the prior compaction.
    await inner.add_message(user_message([{"type": "text", "text": "q0"}]))
    await inner.add_message(assistant_message("a0"))
    await inner.add_message(user_message([{"type": "text", "text": "q1"}]))
    await inner.add_message(assistant_message("a1"))

    # Inject messages BEFORE the prior compaction directly into the
    # internal buffer. These must be invisible to the LLM.
    pre_compaction_payload = [
        {"role": "user", "content": "should-never-appear-q"},
        {"role": "assistant", "content": "should-never-appear-a"},
    ]
    inner._messages[0:0] = pre_compaction_payload  # type: ignore[index]
    # Advance _forward_offset past the pre-compaction noise so that
    # ``memory.compact`` correctly locates the prior compaction at
    # the new offset.
    inner._forward_offset += len(pre_compaction_payload)

    provider = _StubLLMProvider(chunks=["x"])
    agent = AgentMetadata(
        name="compacting",
        agent_type="compacting",
        system_prompt="you are the agent",
        compaction={"prompt_token_threshold": 100, "keep_recent": 2},
    )
    runtime, _ = _make_runtime(
        llm_provider=provider,
        default_settings={"prompt_token_threshold": 100, "keep_recent": 1},
        agent_metadata=agent,
        memory=inner,
    )

    events: list[Any] = []
    async for evt in runtime.compact_session("s"):
        events.append(evt)
    assert isinstance(events[-1], CompactionEnd)
    assert len(provider.chat_calls) == 1
    payload = provider.chat_calls[0]

    # Flatten the chat payload to a single string for substring checks
    # — the pre-compaction messages are distinctive enough that any
    # leak will show up.
    serialized = str(payload)
    assert "should-never-appear-q" not in serialized
    assert "should-never-appear-a" not in serialized

    # The payload still ends with the user-side summary request, and
    # the prior summary is still prepended.
    assert payload[-1]["role"] == "user"
    assert payload[-1]["content"] == SUMMARY_REQUEST
    assert {"role": "assistant", "content": "prior summary text"} in payload


@pytest.mark.asyncio
async def test_compact_session_skips_system_message_when_prompt_empty() -> None:
    """If the agent has no ``system_prompt``, the payload must start
    with the conversation history (not a synthetic empty system turn).
    """
    from minimal_harness.agent._compaction import DEFAULT_SUMMARY_REQUEST as SUMMARY_REQUEST

    inner = ConversationMemory()
    await inner.add_message(user_message([{"type": "text", "text": "q0"}]))
    await inner.add_message(assistant_message("a0"))

    provider = _StubLLMProvider(chunks=["x"])
    agent = AgentMetadata(
        name="compacting",
        agent_type="compacting",
        system_prompt="",  # empty
        compaction={"prompt_token_threshold": 100, "keep_recent": 1},
    )
    runtime, _ = _make_runtime(
        llm_provider=provider,
        default_settings={"prompt_token_threshold": 100, "keep_recent": 1},
        agent_metadata=agent,
        memory=inner,
    )

    events: list[Any] = []
    async for evt in runtime.compact_session("s"):
        events.append(evt)
    assert isinstance(events[-1], CompactionEnd)
    assert len(provider.chat_calls) == 1
    payload = provider.chat_calls[0]
    # No system message when system_prompt is empty.
    assert payload[0]["role"] != "system"
    # The trailing message is still the user-side summary request.
    assert payload[-1]["role"] == "user"
    assert payload[-1]["content"] == SUMMARY_REQUEST


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
async def test_compact_session_propagates_summarizer_error() -> None:
    """If the LLM stream raises mid-flight, ``compact_session`` must
    surface ``CompactionEnd(error=...)`` (same contract as the agent
    loop). The buffer is untouched.
    """
    inner = ConversationMemory()
    for i in range(6):
        await inner.add_message(user_message([{"type": "text", "text": f"q{i}"}]))
        await inner.add_message(assistant_message(f"a{i}"))
    before = [dict(m) for m in inner.get_replay_messages()]

    runtime, _ = _make_runtime(
        llm_provider=_StubLLMProvider(chunks=["partial-"], raise_after=0),
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


@pytest.mark.asyncio
async def test_compact_session_uses_custom_compaction_prompt() -> None:
    """When ``compaction_prompt`` is set in the agent's CompactionSettings,
    the chat payload must end with that custom prompt instead of the
    built-in ``DEFAULT_SUMMARY_REQUEST``.
    """
    inner = ConversationMemory()
    await inner.add_message(user_message([{"type": "text", "text": "q0"}]))
    await inner.add_message(assistant_message("a0"))

    custom_prompt = "Please translate the above conversation into French."
    provider = _StubLLMProvider(chunks=["traduction"])
    agent = AgentMetadata(
        name="compacting",
        agent_type="compacting",
        system_prompt="you are the agent",
        compaction={
            "prompt_token_threshold": 100,
            "keep_recent": 1,
            "compaction_prompt": custom_prompt,
        },
    )
    runtime, _ = _make_runtime(
        llm_provider=provider,
        default_settings={"prompt_token_threshold": 100, "keep_recent": 1},
        agent_metadata=agent,
        memory=inner,
    )

    events: list[Any] = []
    async for evt in runtime.compact_session("s"):
        events.append(evt)

    assert isinstance(events[-1], CompactionEnd)
    assert len(provider.chat_calls) == 1
    payload = provider.chat_calls[0]
    # The last message must be the custom prompt.
    assert payload[-1]["role"] == "user"
    assert payload[-1]["content"] == custom_prompt


@pytest.mark.asyncio
async def test_compact_session_uses_default_prompt_when_custom_empty() -> None:
    """When ``compaction_prompt`` is set to an empty string, the
    built-in ``DEFAULT_SUMMARY_REQUEST`` must be used.
    """
    from minimal_harness.agent._compaction import DEFAULT_SUMMARY_REQUEST

    inner = ConversationMemory()
    await inner.add_message(user_message([{"type": "text", "text": "q0"}]))
    await inner.add_message(assistant_message("a0"))

    provider = _StubLLMProvider(chunks=["x"])
    agent = AgentMetadata(
        name="compacting",
        agent_type="compacting",
        system_prompt="you are the agent",
        compaction={
            "prompt_token_threshold": 100,
            "keep_recent": 1,
            "compaction_prompt": "",  # empty string
        },
    )
    runtime, _ = _make_runtime(
        llm_provider=provider,
        default_settings={"prompt_token_threshold": 100, "keep_recent": 1},
        agent_metadata=agent,
        memory=inner,
    )

    events: list[Any] = []
    async for evt in runtime.compact_session("s"):
        events.append(evt)

    assert isinstance(events[-1], CompactionEnd)
    assert len(provider.chat_calls) == 1
    payload = provider.chat_calls[0]
    # The last message must be the built-in default.
    assert payload[-1]["role"] == "user"
    assert payload[-1]["content"] == DEFAULT_SUMMARY_REQUEST
