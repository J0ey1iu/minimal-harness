from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, AsyncIterator
from unittest.mock import MagicMock

import pytest

from minimal_harness.agent import AgentRuntime
from minimal_harness.agent.runtime import AgentRuntimeProtocol
from minimal_harness.memory import ExtendedInputContentPart
from minimal_harness.tool.base import Tool

if TYPE_CHECKING:
    pass


class _MockRegistry:
    """Minimal AgentRegistryProtocol stub for testing."""

    def register(
        self, agent: Any, *, name: str | None = None, description: str | None = None
    ) -> None: ...

    def unregister(self, name: str) -> bool: ...

    def get(self, name: str) -> Any | None: ...

    def get_all(self) -> list[Any]: ...

    def names(self) -> list[str]: ...

    def clear(self) -> None: ...

    def add_listener(self, listener: Any) -> None: ...

    def remove_listener(self, listener: Any) -> None: ...


class _TestAgent:
    """Minimal Agent that records run args and yields a preset event list."""

    def __init__(self, events: list[Any] | None = None) -> None:
        self.events: list[Any] = events or []
        self.run_args: tuple | None = None

    async def run(
        self,
        user_input: Any,
        stop_event: asyncio.Event | None = None,
        memory: Any = None,
        tools: Any = None,
    ) -> AsyncIterator[Any]:
        self.run_args = (user_input, stop_event, memory, tools)
        for event in self.events:
            if stop_event is not None and stop_event.is_set():
                break
            yield event


class _SlowAgent:
    """Agent that yields events with a delay, used for stop-event testing."""

    def __init__(self, events: list[Any]) -> None:
        self.events = events
        self._sleep = 0.05

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
            await asyncio.sleep(self._sleep)
            yield event


def _input(text: str = "hi") -> list[ExtendedInputContentPart]:
    return [{"type": "text", "text": text}]


@pytest.fixture
def runtime() -> AgentRuntime:
    return AgentRuntime(_MockRegistry())


# -- Return type -------------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_task_stop_event_and_queue(runtime: AgentRuntime) -> None:
    agent = _TestAgent()
    task, stop_event, event_queue = runtime.run(
        agent=agent,
        memory=MagicMock(),
        tools=[],
        user_input=_input(),
    )
    assert isinstance(task, asyncio.Task)
    assert isinstance(stop_event, asyncio.Event)
    assert isinstance(event_queue, asyncio.Queue)


# -- Argument forwarding -----------------------------------------------


@pytest.mark.asyncio
async def test_run_forwards_args_to_agent(runtime: AgentRuntime) -> None:
    agent = _TestAgent()
    memory = MagicMock()
    tools: list[Tool] = [MagicMock()]
    user_input = _input("hi")

    task, stop_event, event_queue = runtime.run(
        agent=agent,
        memory=memory,
        tools=tools,
        user_input=user_input,
    )

    await event_queue.get()  # None sentinel

    assert agent.run_args is not None
    forwarded_input, forwarded_stop, forwarded_memory, forwarded_tools = agent.run_args
    assert forwarded_input == user_input
    assert forwarded_memory is memory
    assert tools[0] in forwarded_tools  # original tool preserved
    assert any(t.name == "handoff" for t in forwarded_tools)
    assert any(t.name == "discover_agents" for t in forwarded_tools)
    assert forwarded_stop is stop_event  # should be the same object


# -- Event streaming ---------------------------------------------------


@pytest.mark.asyncio
async def test_run_streams_events_through_queue(runtime: AgentRuntime) -> None:
    events_in: list[Any] = [
        MagicMock(spec=["__getitem__"]),
        MagicMock(spec=["__getitem__"]),
    ]
    agent = _TestAgent(events_in)

    task, stop_event, event_queue = runtime.run(
        agent=agent,
        memory=MagicMock(),
        tools=[],
        user_input=[],
    )

    events_out: list[Any] = []
    while True:
        event = await event_queue.get()
        if event is None:
            break
        events_out.append(event)

    assert events_out == events_in


# -- Sentinel ----------------------------------------------------------


@pytest.mark.asyncio
async def test_run_sends_none_sentinel_when_done(runtime: AgentRuntime) -> None:
    agent = _TestAgent()

    task, stop_event, event_queue = runtime.run(
        agent=agent,
        memory=MagicMock(),
        tools=[],
        user_input=[],
    )

    sentinel = await event_queue.get()
    assert sentinel is None


# -- Stop event --------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_event_halts_agent(runtime: AgentRuntime) -> None:
    agent = _SlowAgent([{"type": "chunk"}])

    task, stop_event, event_queue = runtime.run(
        agent=agent,
        memory=MagicMock(),
        tools=[],
        user_input=[],
    )

    first = await event_queue.get()
    assert first == {"type": "chunk"}

    stop_event.set()

    sentinel = await event_queue.get()
    assert sentinel is None  # agent stopped early


# -- Statelessness -----------------------------------------------------


@pytest.mark.asyncio
async def test_consecutive_runs_are_independent(runtime: AgentRuntime) -> None:
    agent_a = _TestAgent(["from-a"])
    agent_b = _TestAgent(["from-b"])

    task_a, stop_a, queue_a = runtime.run(
        agent=agent_a,
        memory=MagicMock(),
        tools=[],
        user_input=[],
    )
    task_b, stop_b, queue_b = runtime.run(
        agent=agent_b,
        memory=MagicMock(),
        tools=[],
        user_input=[],
    )

    result_a = await queue_a.get()
    assert result_a == "from-a"
    assert await queue_a.get() is None

    result_b = await queue_b.get()
    assert result_b == "from-b"
    assert await queue_b.get() is None


# -- Protocol conformance ----------------------------------------------


def test_agent_runtime_conforms_to_protocol() -> None:
    assert isinstance(AgentRuntime(_MockRegistry()), AgentRuntimeProtocol)

    class CustomRuntime:
        def run(
            self,
            agent: Any,
            memory: Any,
            tools: Any,
            user_input: Any,
        ) -> tuple[asyncio.Task, asyncio.Event, asyncio.Queue]:
            return (
                asyncio.create_task(asyncio.sleep(0)),
                asyncio.Event(),
                asyncio.Queue(),
            )

    assert isinstance(CustomRuntime(), AgentRuntimeProtocol)


def test_agent_runtime_protocol_requires_run() -> None:
    class BadRuntime:
        pass

    assert not isinstance(BadRuntime(), AgentRuntimeProtocol)
