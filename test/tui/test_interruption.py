from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator
from unittest.mock import MagicMock

import pytest

from minimal_harness.agent.registry import AgentRegistry
from minimal_harness.agent.runtime import AgentRuntime
from minimal_harness.client.built_in.buffer import StreamBuffer
from minimal_harness.client.built_in.context import AppContext
from minimal_harness.client.built_in.session_controller import SessionController
from minimal_harness.types import LLMChunkDelta


class _SlowAgent:
    """Agent that respects stop_event between events."""

    def __init__(self, events: list[Any], delay: float = 0.05) -> None:
        self.events = events
        self._delay = delay

    @property
    def memory(self) -> Any:
        return MagicMock()

    async def run(
        self,
        user_input: Any,
        stop_event: asyncio.Event | None = None,
        memory: Any = None,
        tools: Any = None,
    ) -> AsyncIterator[Any]:
        for event in self.events:
            if stop_event is not None and stop_event.is_set():
                break
            try:
                await asyncio.wait_for(asyncio.Event().wait(), timeout=self._delay)
            except asyncio.TimeoutError:
                pass
            if stop_event is not None and stop_event.is_set():
                break
            yield event


class _MockRegistry:
    def register(self, agent, *, name=None, description=None, tools=None): ...
    def unregister(self, name):
        return True

    def get(self, name):
        return None

    def get_all(self):
        return []

    def names(self):
        return []

    def clear(self): ...
    def add_listener(self, listener): ...
    def remove_listener(self, listener): ...


@pytest.fixture
def runtime():
    return AgentRuntime(_MockRegistry())


@pytest.mark.asyncio
async def test_stop_event_halts_agent_early(runtime):
    agent = _SlowAgent([{"n": 1}, {"n": 2}, {"n": 3}], delay=0.08)
    _, stop_event, event_queue = runtime.run(
        agent=agent,
        memory=MagicMock(),
        tools=[],
        user_input=[],
    )

    event1 = await event_queue.get()
    assert event1 == {"n": 1}

    stop_event.set()

    sentinel = await event_queue.get()
    assert sentinel is None


@pytest.mark.asyncio
async def test_stop_before_next_iteration_prevents_more_events(runtime):
    agent = _SlowAgent([{"n": 1}, {"n": 2}], delay=0.15)
    task, stop_event, event_queue = runtime.run(
        agent=agent,
        memory=MagicMock(),
        tools=[],
        user_input=[],
    )

    await event_queue.get()

    stop_event.set()

    sentinel = await event_queue.get()
    assert sentinel is None


@pytest.mark.asyncio
async def test_independent_task_interruption(runtime):
    agent_a = _SlowAgent([{"src": "a", "i": 1}, {"src": "a", "i": 2}], delay=0.1)
    agent_b = _SlowAgent([{"src": "b", "i": 1}], delay=0.05)

    _, stop_a, queue_a = runtime.run(
        agent=agent_a,
        memory=MagicMock(),
        tools=[],
        user_input=[],
    )
    _, stop_b, queue_b = runtime.run(
        agent=agent_b,
        memory=MagicMock(),
        tools=[],
        user_input=[],
    )

    a1 = await queue_a.get()
    assert a1 == {"src": "a", "i": 1}

    stop_a.set()

    b1 = await queue_b.get()
    assert b1 == {"src": "b", "i": 1}

    sentinel_b = await queue_b.get()
    assert sentinel_b is None

    sentinel_a = await queue_a.get()
    assert sentinel_a is None


# -- SessionController interruption ------------------------------------------


class TestSessionControllerInterruption:
    def test_interrupt_calls_stop_on_current_session(self):
        ctx = MagicMock(spec=AppContext)
        ctrl = SessionController(MagicMock(spec=AgentRuntime), AgentRegistry(), ctx)
        session = MagicMock()
        ctrl._sessions["s1"] = session
        ctrl._current_session_id = "s1"

        ctrl.interrupt()
        session.interrupt.assert_called_once()

    def test_interrupt_sets_stop_event_for_active_run(self):
        ctx = MagicMock(spec=AppContext)
        ctrl = SessionController(MagicMock(spec=AgentRuntime), AgentRegistry(), ctx)
        ctrl._sessions["s1"] = MagicMock()
        ctrl._current_session_id = "s1"

        stop_event = asyncio.Event()
        ctrl._active_runs["s1"] = (MagicMock(), stop_event, MagicMock())

        ctrl.interrupt()
        assert stop_event.is_set()


# -- Buffer state after stop (no textual app needed) ---------------------------


def test_buffer_clear_after_streaming():
    buf = StreamBuffer()
    buf.add_chunk(LLMChunkDelta(content="partial"))
    assert buf.content == "partial"
    buf.clear()
    assert buf.content == ""
    assert buf._flushed is False


@pytest.mark.asyncio
async def test_consecutive_runs_are_independent(runtime):
    agent_a = _SlowAgent([{"src": "a"}], delay=0.01)
    agent_b = _SlowAgent([{"src": "b"}], delay=0.01)

    _, _, queue_a = runtime.run(
        agent=agent_a,
        memory=MagicMock(),
        tools=[],
        user_input=[],
    )
    _, _, queue_b = runtime.run(
        agent=agent_b,
        memory=MagicMock(),
        tools=[],
        user_input=[],
    )

    assert await queue_a.get() == {"src": "a"}
    assert await queue_a.get() is None

    assert await queue_b.get() == {"src": "b"}
    assert await queue_b.get() is None
