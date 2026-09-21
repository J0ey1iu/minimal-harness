from __future__ import annotations

import re
from typing import Any, AsyncIterator, Sequence, cast

import pytest

from minimal_harness.agent.compacting import CompactionAgent
from minimal_harness.llm.llm import LLMResponse, Stream
from minimal_harness.memory import (
    ConversationMemory,
    Message,
    system_message,
)
from minimal_harness.tool.base import create_streaming_tool
from minimal_harness.types import (
    AgentEvent,
    CompactionStart,
    LLMChunkDelta,
    ToolCall,
    ToolResult,
)

GOAL = 100


def _extract_counter(messages: Sequence[Message]) -> int:
    """Like a real LLM, the provider must read its progress from the
    messages it actually sees — never from hidden state."""
    best = 0
    for m in messages:
        text = ""
        c = m.get("content")
        if isinstance(c, str):
            text = c
        elif isinstance(c, list):
            text = " ".join(str(p.get("text") or "") for p in c if isinstance(p, dict))
        for hit in re.findall(r"counter=(\d+)", text):
            best = max(best, int(hit))
    return best


class StatefulLongTaskProvider:
    """LLM driving a 100-step counter task, reading progress ONLY from
    the visible messages. Also enforces the Anthropic wire contract
    (every request must contain a user-role message, first non-system
    message must be user) — exactly the rule issue #62 violated.
    """

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []
        self.requests_with_user: int = 0
        self.requests_with_system: int = 0
        self.shapes: dict[tuple, int] = {}  # (role, brief) tuple -> count

    @staticmethod
    def _enforce_anthropic_contract(messages: Sequence[Message]) -> None:
        non_system = [m for m in messages if m.get("role") != "system"]
        if not non_system or non_system[0].get("role") != "user":
            raise RuntimeError(
                "AnthropicReject: first message must use the 'user' role"
            )
        if not any(m.get("role") == "user" for m in messages):
            raise RuntimeError("AnthropicReject: no user-role message")

    @staticmethod
    def _brief(m: Message) -> str:
        role = m.get("role")
        if role == "user":
            c = m.get("content")
            if isinstance(c, list):
                return str(c[0].get("text") if isinstance(c[0], dict) else c[0])[:60]
            return str(c)[:60]
        if role == "assistant":
            if m.get("tool_calls"):
                return f"tool_calls x{len(m.get('tool_calls') or [])}"
            return str(m.get("content"))[:60]
        if role == "tool":
            return str(m.get("content"))[:30]
        return str(m.get("content"))[:40]

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[Any] = (),
        stop_event: Any = None,
        **kwargs: Any,
    ) -> Stream[LLMChunkDelta]:
        msgs = list(messages)
        self.calls.append(msgs)
        self.shapes[tuple((m.get("role"), self._brief(m)) for m in msgs)] = (
            self.shapes.get(tuple((m.get("role"), self._brief(m)) for m in msgs), 0) + 1
        )
        self._enforce_anthropic_contract(msgs)
        self.requests_with_user += any(m.get("role") == "user" for m in msgs)
        self.requests_with_system += any(m.get("role") == "system" for m in msgs)

        counter = _extract_counter(msgs)
        if counter >= GOAL:
            content = f"task complete: counter={counter}"
            tool_calls = []
        else:
            content = None
            tool_calls: list[ToolCall] = cast(
                list[ToolCall],
                [
                    {
                        "id": f"call_{counter}",
                        "type": "function",
                        "function": {"name": "increment", "arguments": "{}"},
                    }
                ],
            )
        resp = LLMResponse(
            content=content,
            reasoning_content=None,
            tool_calls=tool_calls,
            finish_reason="stop",
            # Big per-call usage → `should_fold` fires after EVERY LLM
            # response → ~GOAL consecutive compactions over the run.
            usage={"prompt_tokens": 5000, "completion_tokens": 2, "total_tokens": 5002},
        )

        async def _gen() -> AsyncIterator[Any]:
            if resp.content:
                yield LLMChunkDelta(content=resp.content)
            yield resp

        return Stream(_gen())


async def _honest_summarizer(
    messages: list[Message], existing_summary: str | None
) -> AsyncIterator[str]:
    """Summary keeps the two facts the next turn needs: the task goal and
    the latest confirmed progress (what a real summarizer would keep)."""
    yield f"[progress summary: goal={GOAL}, last confirmed counter={_extract_counter(messages)}]"


def _assert_expected_llm_view(calls: list[list[Message]], n_calls: int) -> None:
    """Every request the LLM actually saw must have the canonical post-compaction
    shape, down to message-level roles and order:

      call 0:    [system, user(real input)]                     — before any fold
      call 1..N: [system, user(Continue), assistant(summary),
                  assistant(tool_calls), tool]                  — after every fold

    i.e. system survives every fold; exactly ONE user message exists and is the
    first non-system message (Anthropic contract); the compaction summary is
    projected to assistant right after it; the live tool round stays paired.
    """
    assert len(calls) == n_calls
    for i, msgs in enumerate(calls):
        roles = [m.get("role") for m in msgs]
        # Exactly one system, and it must be at index 0.
        assert roles.count("system") == 1, roles
        assert roles[0] == "system", roles
        if i == 0:
            # Pre-fold: nothing compacted yet, just system + real user input.
            assert roles == ["system", "user"], roles
            assert "start the counter task" in str(msgs[1].get("content"))
            continue
        # Post-fold canonical shape: user, summary(assistant), tool round.
        assert roles == ["system", "user", "assistant", "assistant", "tool"], roles
        # Exactly one user turn, synthetic "Continue.", first after system.
        assert msgs[1].get("role") == "user"
        assert "Continue." in str(msgs[1].get("content"))
        # Compaction summary projected to assistant, right after the user.
        assert msgs[2].get("role") == "assistant"
        assert not msgs[2].get("tool_calls")
        assert "progress summary" in str(msgs[2].get("content")), msgs[2]
        # The live tool round: assistant declares, tool answers, ids match.
        tool_calls = msgs[3].get("tool_calls") or []
        assert tool_calls, msgs[3]
        declared = tool_calls[0].get("id")
        assert msgs[4].get("role") == "tool"
        assert msgs[4].get("tool_call_id") == declared
        # The tool result carries the current progress the LLM acts on.
        assert "counter=" in str(msgs[4].get("content"))


@pytest.mark.asyncio
async def test_long_task_survives_many_compactions() -> None:
    """A very long tool task that triggers compaction after EVERY LLM
    response must still run to completion. Regression for issue #62:
    each fold leaves the LLM-visible buffer with a user-role message
    (synthetic, non-persisted) and the system message intact."""
    state = {"value": 0}

    async def increment() -> AsyncIterator[Any]:
        state["value"] += 1
        yield ToolResult(
            content={"status": "ok", "content": f"counter={state['value']}"}
        )

    tool = create_streaming_tool(name="increment", fn=increment)
    provider = StatefulLongTaskProvider()
    agent = CompactionAgent(
        llm_provider=provider,
        summarizer=_honest_summarizer,
        prompt_token_threshold=1000,
        keep_recent=0,
        max_iterations=1000,
    )

    memory = ConversationMemory()
    await memory.add_message(
        system_message("Counter task: increment from 0 until the counter reaches 100")
    )

    events: list[AgentEvent] = []
    async for evt in agent.run(
        user_input=[{"type": "text", "text": "start the counter task"}],
        memory=memory,
        tools=[tool],
    ):
        events.append(evt)

    # Task completed in full.
    assert state["value"] == GOAL
    assert any("task complete" in str(getattr(e, "content", "")) for e in events), (
        "agent never produced the final answer"
    )

    # Really multiple compactions happened over the run.
    folds = [e for e in events if isinstance(e, CompactionStart)]
    assert len(folds) >= 50, f"expected heavy compaction, got {len(folds)} folds"

    # Every single LLM request met the Anthropic contract — a fold may
    # never leave the buffer without a user-role message.
    assert provider.requests_with_user == len(provider.calls)
    # The system prompt survived every fold.
    assert provider.requests_with_system == len(provider.calls)

    # Each LLM request had exactly the expected post-compaction shape:
    # [system, user(Continue), assistant(summary), assistant(tool_calls), tool].
    _assert_expected_llm_view(provider.calls, len(provider.calls))

    # The synthetic "Continue" user turns are projection-only: nothing
    # carrying them may have reached the persistent buffer.
    persisted_text = str([m.get("content") for m in memory.get_all_messages()])
    assert "Continue." not in persisted_text

    # Every request's first non-system message was user (already enforced
    # inside chat, but assert idempotently for the report).
    for msgs in provider.calls:
        non_system = [m for m in msgs if m.get("role") != "system"]
        assert non_system[0].get("role") == "user"
