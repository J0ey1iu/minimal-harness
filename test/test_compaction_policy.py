"""RFC #57 compaction policy: leading-edge trigger, soft-limit, keep-recent anchoring."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Sequence

from minimal_harness.agent._compaction import (
    COMPACTION_SUMMARY_PROMPT,
    build_chat_payload,
    compute_effective_keep_recent,
    estimate_prompt_tokens,
    resolve_compaction_prompt,
    should_fold,
    soft_limit_threshold,
)
from minimal_harness.agent.compacting import CompactionAgent
from minimal_harness.llm.llm import LLMResponse, Stream
from minimal_harness.memory import (
    ConversationMemory,
    Message,
    assistant_message,
    user_message,
)
from minimal_harness.types import (
    CompactionStart,
    LLMChunkDelta,
    ToolCall,
)


class _FakeMemory:
    """Minimal Memory-shaped object for unit-level policy checks."""

    def __init__(self, messages: list[Message], usage: dict[str, int]) -> None:
        self._messages = messages
        self._usage = usage

    def get_forward_messages(self) -> list[Message]:
        return list(self._messages)

    def get_all_messages(self) -> list[Message]:
        return list(self._messages)

    def get_message_usage(self) -> dict[str, int]:
        return self._usage


def _user(text: str) -> Message:
    return user_message([{"type": "text", "text": text}])


def _assistant(text: str, tool_calls: list[ToolCall] | None = None) -> Message:
    return assistant_message(text, tool_calls)


def _tool_result(tool_call_id: str, text: str) -> Message:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": text}


_TW: ToolCall = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "t", "arguments": "{}"},
}


# ── estimate_prompt_tokens ────────────────────────────────────────────


def test_estimate_prompt_tokens_ascii_and_cjk():
    mem = _FakeMemory([_user("hello world"), _assistant("你好世界")], {})
    # 11 + 12 bytes ≈ 23 bytes; //4 ≈ 5 tokens. Bound is a rough estimate:
    # assert a sane range, not exactness.
    est = estimate_prompt_tokens(mem)
    assert 4 <= est <= 8


def test_estimate_prompt_tokens_multimodal_parts():
    mem = _FakeMemory(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "abc"},
                    {"type": "text", "text": "xyz"},
                ],
            }
        ],
        {},
    )
    assert 1 <= estimate_prompt_tokens(mem) <= 2


# ── compute_effective_keep_recent ─────────────────────────────────────


def test_anchor_tail_preserves_static_count():
    msgs = [_user("hi"), _assistant("yo")]
    assert compute_effective_keep_recent(_FakeMemory(msgs, {}), 6, "tail") == 6


def test_anchor_last_user_floors_at_user_turn():
    msgs = [_user("a"), _assistant("1"), _user("b"), _assistant("2")]
    # last user at index 2 → keep at least 4-2 = 2
    assert compute_effective_keep_recent(_FakeMemory(msgs, {}), 0, "last_user") == 2
    assert compute_effective_keep_recent(_FakeMemory(msgs, {}), 6, "last_user") == 6


def test_anchor_last_tool_round_walks_past_user():
    msgs = [
        _user("go"),
        _assistant("", tool_calls=[_TW]),
        _tool_result("call_1", "42"),
        _assistant("done"),
    ]
    # tool round starts at index 1 → keep at least 4-1 = 3, never fold it
    eff = compute_effective_keep_recent(_FakeMemory(msgs, {}), 0, "last_tool_round")
    assert eff == 3


def test_anchor_last_tool_round_plain_chat_falls_back_to_user():
    msgs = [_user("go"), _assistant("plain reply")]
    eff = compute_effective_keep_recent(_FakeMemory(msgs, {}), 0, "last_tool_round")
    assert eff == 2  # last user at 0 → keep 2


def test_anchor_empty_memory_keeps_static():
    assert compute_effective_keep_recent(_FakeMemory([], {}), 4, "last_tool_round") == 4


# ── soft limit / trigger policy ───────────────────────────────────────


def test_soft_limit_threshold_disabled_by_default():
    assert soft_limit_threshold(0.0, 64000) == 0
    assert soft_limit_threshold(0.55, 0) == 0


def test_soft_limit_threshold_enabled():
    assert soft_limit_threshold(0.55, 64000) == 35200


def test_should_fold_posthoc_threshold_only():
    mem = _FakeMemory([_user("hi")], {"total_tokens": 9000})
    assert should_fold(mem, 8000) == (True, 9000)
    mem2 = _FakeMemory([_user("hi")], {"total_tokens": 100})
    assert should_fold(mem2, 8000) == (False, 100)


def test_should_fold_leading_edge_fires_before_provider_usage():
    # No post-hoc usage yet, but the buffered prompt is already huge.
    big = _user("x" * 200_000)  # ~50_000 est tokens
    mem = _FakeMemory([big], {"total_tokens": 0})
    fold, cumulative = should_fold(
        mem, 8000, soft_limit_ratio=0.55, max_context_tokens=64000
    )
    assert fold is True
    assert cumulative == 0  # post-hoc count stays honest


def test_should_fold_or_semantics_with_small_buffer():
    small = _FakeMemory([_user("hi")], {"total_tokens": 0})
    assert should_fold(
        small, 8000, soft_limit_ratio=0.55, max_context_tokens=64000
    ) == (False, 0)


def test_should_fold_estimate_leading_edge_flag_off():
    big = _user("x" * 200_000)
    mem = _FakeMemory([big], {"total_tokens": 0})
    fold, _ = should_fold(
        mem,
        8000,
        soft_limit_ratio=0.55,
        max_context_tokens=64000,
        estimate_leading_edge=False,
    )
    assert fold is False  # only the provider-reported usage counts


# ── summary prompt preset ─────────────────────────────────────────────


def test_restated_goal_preset_resolves():
    assert resolve_compaction_prompt("restated-goal") == COMPACTION_SUMMARY_PROMPT
    assert resolve_compaction_prompt("custom instruction") == "custom instruction"
    assert resolve_compaction_prompt(None) is None


def test_build_chat_payload_uses_preset():
    chat = build_chat_payload(
        "sys", [_user("hi")], None, summary_prompt="restated-goal"
    )
    assert chat[-1]["role"] == "user"
    assert chat[-1]["content"] == COMPACTION_SUMMARY_PROMPT


def test_restated_goal_preset_is_four_section_dense_format():
    p = COMPACTION_SUMMARY_PROMPT
    for heading in [
        "Goals",
        "Decisions & Outcomes",
        "Open Questions / Pending",
        "Entities & IDs",
    ]:
        assert heading in p, heading
    assert "Restated goal:" in p
    assert "ORIGINAL goal" in p


# ── end-to-end: leading-edge pre-flight folds before the first LLM call ──


class _CountingSummarizer:
    def __init__(self) -> None:
        self.calls = 0
        self.summarized: list[Message] = []

    def __call__(
        self, messages: list[Message], existing: str | None
    ) -> AsyncIterator[str]:
        self.calls += 1
        self.summarized = list(messages)

        async def _gen():
            yield "SUMMARY:" + (existing or "")

        return _gen()


class _OneShotProvider:
    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[Any] = (),
        stop_event: asyncio.Event | None = None,
        **kwargs: Any,
    ) -> Stream[LLMChunkDelta]:
        async def _gen() -> AsyncIterator[Any]:
            yield LLMResponse("final answer", None, [], "stop", None)

        return Stream(_gen())


async def test_pre_llm_hook_folds_oversized_buffer_before_first_call():
    memory = ConversationMemory()
    # History: an assistant turn with a huge tool result already in buffer.
    await memory.add_message(_user("run the big task"))
    await memory.add_message(_assistant("", tool_calls=[_TW]))
    await memory.add_message(_tool_result("call_1", "y" * 300_000))

    summarizer = _CountingSummarizer()
    agent = CompactionAgent(
        llm_provider=_OneShotProvider(),
        summarizer=summarizer,
        prompt_token_threshold=8000,
        keep_recent=1,
        soft_limit_ratio=0.5,
        max_context_tokens=64000,  # soft threshold 32000 est tokens
        estimate_leading_edge=True,
    )
    events = [ev async for ev in agent.run([], memory=memory, tools=[])]
    assert summarizer.calls == 1, "pre-flight must have folded before the LLM call"
    kinds = [type(e).__name__ for e in events]
    assert "CompactionStart" in kinds and "CompactionEnd" in kinds
    assert "AgentEnd" in kinds


async def test_no_fold_when_soft_limit_disabled():
    memory = ConversationMemory()
    await memory.add_message(_user("run the big task"))
    await memory.add_message(_assistant("", tool_calls=[_TW]))
    await memory.add_message(_tool_result("call_1", "y" * 300_000))

    summarizer = _CountingSummarizer()
    agent = CompactionAgent(
        llm_provider=_OneShotProvider(),
        summarizer=summarizer,
        prompt_token_threshold=8000,
        keep_recent=1,
        # defaults: soft_limit_ratio=0.0 → RFC #57 leading edge disabled
    )
    events = [ev async for ev in agent.run([], memory=memory, tools=[])]
    assert summarizer.calls == 0
    assert not any(isinstance(e, CompactionStart) for e in events)


async def test_keep_recent_anchor_flows_into_compact():
    from minimal_harness.agent._compaction import compute_effective_keep_recent

    memory = ConversationMemory()
    await memory.add_message(_user("go"))
    await memory.add_message(_assistant("", tool_calls=[_TW]))
    await memory.add_message(_tool_result("call_1", "42"))

    # simulate the agent computing the floor the way _run_compaction does
    anchor_eff = compute_effective_keep_recent(memory, 0, "last_tool_round")
    # keep_start=1 (the assistant-with-tool_calls round) → keep tail [1:] = 2
    assert anchor_eff == 2
