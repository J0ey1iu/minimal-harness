"""Canonical message ids: stamped once at add time, stable across reloads.

Covers the invariant behind the MessageEvent/AgentEnd message-id fix:

* ``Memory.add_message`` stamps ``msg-{seq}`` (per-session counter) so a
  freshly streamed message and its reloaded row share the same id.
* The id survives ``dump_memory``/``load_memory`` and never collides with
  the read-side fallback for legacy rows.
* Compaction summaries and tool-message discards never renumber existing
  ids (identity is independent of buffer order).
* The LLM wire payload (``get_forward_messages``) never carries ``id``.
* ``MessageEvent`` / ``AgentEnd`` surface the same id.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Sequence

import pytest
from minimal_harness.agent.base import BaseAgent
from minimal_harness.llm.llm import LLMResponse, Stream
from minimal_harness.memory import (
    ConversationMemory,
    Message,
    assistant_message,
    tool_message,
    user_message,
)
from minimal_harness.types import AgentEnd, AgentEvent, LLMChunkDelta, MessageEvent


# Read-side convention shared by SessionRepository adapters:
# prefer the stored id, fall back to the positional one for legacy rows.
def read_side_id(msg: Message, position: int) -> str:
    return msg.get("id") or f"msg-{position}"


def _user(text: str) -> Message:
    return user_message([{"type": "text", "text": text}])


class ScriptedLLMProvider:
    """Returns a single plain text response (no tool calls)."""

    def __init__(self, content: str = "ok") -> None:
        self.content = content

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[Any] = (),
        stop_event: Any = None,
        **kwargs: Any,
    ) -> Stream[LLMChunkDelta]:
        async def _gen() -> AsyncIterator[Any]:
            yield LLMResponse(
                content=self.content,
                reasoning_content=None,
                tool_calls=[],
                finish_reason="stop",
                usage=None,
            )

        return Stream(_gen())


@pytest.mark.asyncio
async def test_add_message_stamps_sequential_ids() -> None:
    mem = ConversationMemory()
    await mem.add_message(_user("q"))
    await mem.add_message(assistant_message("a", None))
    await mem.add_message(tool_message("tc1", "result"))
    assert mem.get_all_messages()[0].get("id") == "msg-0"
    assert mem.get_all_messages()[1].get("id") == "msg-1"
    assert mem.get_all_messages()[2].get("id") == "msg-2"


@pytest.mark.asyncio
async def test_ids_survive_dump_load_and_sequence_continues() -> None:
    mem = ConversationMemory()
    await mem.add_message(_user("q"))
    await mem.add_message(assistant_message("a", None))
    data = mem.dump_memory()

    restored = ConversationMemory()
    restored.load_memory(data)
    assert restored.get_all_messages()[0].get("id") == "msg-0"
    assert restored.get_all_messages()[1].get("id") == "msg-1"

    nxt = assistant_message("b", None)
    await restored.add_message(nxt)
    assert nxt.get("id") == "msg-2"


@pytest.mark.asyncio
async def test_legacy_dump_without_ids_starts_sequence_at_len() -> None:
    """Pre-migration dumps have no ids and no counter. New stamps must start
    after the existing rows so they can never collide with the read-side
    ``msg-{i}`` fallback."""
    mem = ConversationMemory()
    await mem.add_message(_user("q1"))
    await mem.add_message(_user("q2"))
    data = mem.dump_memory()
    data.pop("next_message_seq")
    data["messages"][0].pop("id")
    data["messages"][1].pop("id")

    restored = ConversationMemory()
    restored.load_memory(data)
    nxt = assistant_message("a", None)
    await restored.add_message(nxt)
    assert nxt.get("id") == "msg-2"

    # read-side ids are unique across legacy fallback + new stamps
    ids = {read_side_id(m, i) for i, m in enumerate(restored.get_all_messages())}
    assert ids == {"msg-0", "msg-1", "msg-2"}


@pytest.mark.asyncio
async def test_compact_summary_gets_a_stamped_id() -> None:
    mem = ConversationMemory()
    for i in range(4):
        await mem.add_message(_user(f"q{i}"))
        await mem.add_message(assistant_message(f"a{i}", None))

    async def _summarizer(
        msgs: list[Message], existing: str | None
    ) -> AsyncIterator[str]:
        yield "summary"

    async for _ in mem.compact(_summarizer, keep_recent=2):
        pass

    summary = mem.get_all_messages()[mem.get_forward_offset()]
    assert summary["role"] == "compaction"
    assert summary.get("id") == "msg-8"

    # existing messages keep their original ids after the mid-buffer insert
    assert mem.get_all_messages()[0].get("id") == "msg-0"
    assert mem.get_all_messages()[1].get("id") == "msg-1"


@pytest.mark.asyncio
async def test_discard_tool_messages_never_reuses_ids() -> None:
    mem = ConversationMemory()
    await mem.add_message(_user("q"))
    tc1 = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "f", "arguments": "{}"},
    }
    await mem.add_message(assistant_message("thinking", [tc1]))
    await mem.add_message(tool_message("call_1", "result"))
    await mem.add_message(assistant_message("final", None))

    async for _ in mem.discard_tool_messages():
        pass

    # the popped tool row's id must never be handed out again
    nxt = assistant_message("after", None)
    await mem.add_message(nxt)
    assert nxt.get("id") == "msg-4"
    ids = [m.get("id") for m in mem.get_all_messages()]
    assert len(ids) == len(set(ids))


def test_get_forward_messages_strips_id_from_llm_payload() -> None:
    mem = ConversationMemory()
    mem.load_memory(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "q"}],
                    "id": "msg-0",
                },
                {
                    "role": "assistant",
                    "content": "a",
                    "tool_calls": None,
                    "id": "msg-1",
                },
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "extra": {},
            "next_message_seq": 2,
        }
    )
    forward = mem.get_forward_messages()
    assert forward[0]["role"] == "user"
    assert forward[1]["role"] == "assistant"
    assert all("id" not in m for m in forward)


@pytest.mark.asyncio
async def test_agent_events_and_agent_end_carry_canonical_id() -> None:
    mem = ConversationMemory()
    agent = BaseAgent(llm_provider=ScriptedLLMProvider(), max_iterations=1)
    events: list[AgentEvent] = []
    async for evt in agent.run(
        user_input=[{"type": "text", "text": "hello"}],
        stop_event=None,
        memory=mem,
        tools=[],
        system_prompt="",
    ):
        events.append(evt)

    msg_events = [e for e in events if isinstance(e, MessageEvent)]
    assistant_msg_events = [
        e for e in msg_events if e.message.get("role") == "assistant"
    ]
    assert assistant_msg_events, "expected an assistant MessageEvent"
    assert assistant_msg_events[-1].message.get("id") == "msg-1"

    # AgentEnd exposes the same id the frontend should commit the stream with
    agent_end = events[-1]
    assert isinstance(agent_end, AgentEnd)
    assert agent_end.message_id == "msg-1"

    # streaming id == read-side id after reload
    persisted = mem.get_all_messages()
    read_ids = {read_side_id(m, i) for i, m in enumerate(persisted)}
    assert agent_end.message_id in read_ids


@pytest.mark.asyncio
async def test_agent_end_message_id_picks_last_assistant_of_the_round() -> None:
    """One AgentStart~AgentEnd region can hold several messages (assistant
    with tool_calls, tool result, final assistant). The frontend groups the
    whole region into one bubble and targets the LAST assistant message —
    AgentEnd.message_id must match that one, not an intermediate one."""
    tc = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "nope", "arguments": "{}"},
    }

    class _ToolThenAnswer:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(
            self,
            messages: Sequence[Message],
            tools: Sequence[Any] = (),
            stop_event: Any = None,
            **kwargs: Any,
        ) -> Stream[LLMChunkDelta]:
            self.calls += 1
            if self.calls == 1:

                async def _gen1() -> AsyncIterator[Any]:
                    yield LLMResponse(
                        content=None,
                        reasoning_content="thinking",
                        tool_calls=[tc],
                        finish_reason="tool_calls",
                        usage=None,
                    )

                return Stream(_gen1())

            async def _gen2() -> AsyncIterator[Any]:
                yield LLMResponse(
                    content="final answer",
                    reasoning_content=None,
                    tool_calls=[],
                    finish_reason="stop",
                    usage=None,
                )

            return Stream(_gen2())

    mem = ConversationMemory()
    agent = BaseAgent(llm_provider=_ToolThenAnswer(), max_iterations=3)
    events: list[AgentEvent] = []
    async for evt in agent.run(
        user_input=[{"type": "text", "text": "go"}],
        stop_event=None,
        memory=mem,
        tools=[],
        system_prompt="",
    ):
        events.append(evt)

    msgs = mem.get_all_messages()
    assistant_msgs = [m for m in msgs if m.get("role") == "assistant"]
    assert len(assistant_msgs) >= 2, "expected multiple assistant messages in the round"
    last_assistant_id = assistant_msgs[-1].get("id")

    agent_end = events[-1]
    assert isinstance(agent_end, AgentEnd)
    assert agent_end.message_id == last_assistant_id
    assert agent_end.message_id != assistant_msgs[0].get("id")

    # the read-side rule ("last assistant of the region") agrees
    read_ids = [
        m.get("id") or f"msg-{i}"
        for i, m in enumerate(msgs)
        if m.get("role") == "assistant"
    ]
    assert read_ids[-1] == agent_end.message_id


@pytest.mark.asyncio
async def test_agent_end_message_id_none_when_no_assistant_turn() -> None:
    """Error path: no assistant message was produced — message_id stays None
    and the frontend falls back to minting."""

    class _BoomProvider(ScriptedLLMProvider):
        async def chat(
            self,
            messages: Sequence[Message],
            tools: Sequence[Any] = (),
            stop_event: Any = None,
            **kwargs: Any,
        ) -> Stream[LLMChunkDelta]:
            raise RuntimeError("provider down")

    mem = ConversationMemory()
    agent = BaseAgent(llm_provider=_BoomProvider(), max_iterations=1)
    events: list[AgentEvent] = []
    async for evt in agent.run(
        user_input=[{"type": "text", "text": "hello"}],
        stop_event=None,
        memory=mem,
        tools=[],
        system_prompt="",
    ):
        events.append(evt)

    agent_end = events[-1]
    assert isinstance(agent_end, AgentEnd)
    assert agent_end.error is not None
    assert agent_end.message_id is None
