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
from minimal_harness.types import AgentMetadata, LLMChunkDelta


class _SlowAgent:
    """Agent that respects stop_event between events."""

    def __init__(self, events: list[Any], delay: float = 0.05) -> None:
        self.events = events
        self._delay = delay

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


class _MockAgentRegistry:
    def register(self, metadata: AgentMetadata):
        return metadata

    def unregister(self, name):
        return True

    def get(self, name):
        return AgentMetadata(
            name=name,
            description="",
            system_prompt="",
            agent_type="simple",
            tool_names=[],
            metadata_id=name,
        )

    def get_all(self):
        return []

    def names(self):
        return []

    def clear(self): ...

    def add_listener(self, listener): ...

    def remove_listener(self, listener): ...


def _make_mock_memory_store():
    store = MagicMock()
    store.get_memory.return_value = MagicMock(memory_id="mem1")
    store.create_memory.return_value = MagicMock(memory_id="mem1")
    return store


def _make_mock_tool_registry():
    reg = MagicMock()
    reg.get.return_value = None
    reg.get_all.return_value = []
    return reg


@pytest.fixture
def runtime():
    reg = _MockAgentRegistry()
    mem_store = _make_mock_memory_store()
    tool_reg = _make_mock_tool_registry()
    reg.register(AgentMetadata(name="test_agent", metadata_id="test_agent"))
    rt = AgentRuntime(
        agent_registry=reg,
        memory_store=mem_store,
        tool_registry=tool_reg,
    )
    rt._create_agent = MagicMock()
    return rt


@pytest.mark.asyncio
async def test_stop_event_halts_agent_early(runtime):
    agent = _SlowAgent([{"n": 1}, {"n": 2}, {"n": 3}], delay=0.08)
    runtime._create_agent = lambda agent_type: agent
    _, stop_event, event_queue = runtime.run(
        user_input=[],
        agent_metadata_id="test_agent",
        memory_id="mem1",
    )

    event1 = await event_queue.get()
    assert event1 == {"n": 1}

    stop_event.set()

    sentinel = await event_queue.get()
    assert sentinel is None


@pytest.mark.asyncio
async def test_stop_before_next_iteration_prevents_more_events(runtime):
    agent = _SlowAgent([{"n": 1}, {"n": 2}], delay=0.15)
    runtime._create_agent = lambda agent_type: agent
    task, stop_event, event_queue = runtime.run(
        user_input=[],
        agent_metadata_id="test_agent",
        memory_id="mem1",
    )

    await event_queue.get()

    stop_event.set()

    sentinel = await event_queue.get()
    assert sentinel is None


@pytest.mark.asyncio
async def test_independent_task_interruption(runtime):
    agent_a = _SlowAgent([{"src": "a", "i": 1}, {"src": "a", "i": 2}], delay=0.1)
    agent_b = _SlowAgent([{"src": "b", "i": 1}], delay=0.05)

    runtime._agent_registry.register(name="agent_a", metadata_id="agent_a")
    runtime._agent_registry.register(name="agent_b", metadata_id="agent_b")

    create_calls: list[str] = []

    def _create_agent(agent_type: str):
        create_calls.append(agent_type)
        return [agent_a, agent_b][len(create_calls) - 1]

    runtime._create_agent = _create_agent

    _, stop_a, queue_a = runtime.run(
        user_input=[],
        agent_metadata_id="agent_a",
        memory_id="mem1",
    )
    _, stop_b, queue_b = runtime.run(
        user_input=[],
        agent_metadata_id="agent_b",
        memory_id="mem1",
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
        ctrl = SessionController(MagicMock(), AgentRegistry(), ctx)
        session = MagicMock()
        ctrl._sessions["s1"] = session
        ctrl._current_session_id = "s1"

        ctrl.interrupt()
        session.interrupt.assert_called_once()

    def test_interrupt_sets_stop_event_for_active_run(self):
        ctx = MagicMock(spec=AppContext)
        ctrl = SessionController(MagicMock(), AgentRegistry(), ctx)
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

    runtime._agent_registry.register(name="agent_a", metadata_id="agent_a")
    runtime._agent_registry.register(name="agent_b", metadata_id="agent_b")

    create_calls: list[str] = []

    def _create_agent(agent_type: str):
        create_calls.append(agent_type)
        return [agent_a, agent_b][len(create_calls) - 1]

    runtime._create_agent = _create_agent

    _, _, queue_a = runtime.run(
        user_input=[],
        agent_metadata_id="agent_a",
        memory_id="mem1",
    )
    _, _, queue_b = runtime.run(
        user_input=[],
        agent_metadata_id="agent_b",
        memory_id="mem1",
    )

    assert await queue_a.get() == {"src": "a"}
    assert await queue_a.get() is None

    assert await queue_b.get() == {"src": "b"}
    assert await queue_b.get() is None
