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


class _MockToolRegistry:
    """Minimal ToolRegistry stub for testing."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools = {t.name: t for t in (tools or [])}

    def get(self, name: str) -> Any | None:
        return self._tools.get(name)

    def get_all(self) -> list[Tool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def register(self, tool: Any) -> None:
        self._tools[tool.name] = tool

    def clear(self) -> None:
        self._tools.clear()


class _MockMemoryStore:
    """Minimal MemoryStore stub for testing."""

    def __init__(self) -> None:
        self._memories: dict[str, Any] = {}

    def create_memory(self, memory_id=None, agent_name=""):
        from uuid import uuid4

        mid = memory_id or uuid4().hex
        mem = MagicMock()
        self._memories[mid] = mem
        return MagicMock(memory_id=mid)

    def get_memory(self, memory_id: str):
        return self._memories.get(memory_id)

    def save_memory(self, memory, memory_id, extra=None):
        self._memories[memory_id] = memory


class _MockAgentRegistry:
    """Minimal AgentRegistryProtocol stub for testing."""

    def __init__(self, metadata_list: list[Any] | None = None) -> None:
        self._data: dict[str, Any] = {}
        for m in metadata_list or []:
            self._data[m.metadata_id] = m

    def register(
        self,
        *,
        name="",
        description="",
        system_prompt="",
        agent_type="simple",
        tool_names=None,
        metadata_id=None,
    ):
        mid = metadata_id or name
        from minimal_harness.agent.registry import AgentMetadata

        meta = AgentMetadata(
            name=name,
            description=description,
            system_prompt=system_prompt,
            agent_type=agent_type,
            tool_names=tool_names or [],
            metadata_id=mid,
        )
        self._data[mid] = meta
        return meta

    def unregister(self, name: str) -> bool:
        return self._data.pop(name, None) is not None

    def get(self, name: str) -> Any | None:
        return self._data.get(name)

    def get_all(self) -> list[Any]:
        return list(self._data.values())

    def names(self) -> list[str]:
        return list(self._data.keys())

    def clear(self) -> None:
        self._data.clear()

    def add_listener(self, listener: Any) -> None:
        pass

    def remove_listener(self, listener: Any) -> None:
        pass


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
        system_prompt: str = "",
    ) -> AsyncIterator[Any]:
        self.run_args = (user_input, stop_event, memory, tools, system_prompt)
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
        system_prompt: str = "",
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
    reg = _MockAgentRegistry()
    mem_store = _MockMemoryStore()
    tool_reg = _MockToolRegistry()
    rt = AgentRuntime(
        agent_registry=reg,
        memory_store=mem_store,
        tool_registry=tool_reg,
    )
    rt._create_agent = lambda agent_type: _TestAgent()
    return rt


@pytest.fixture
def runtime_with_agent() -> AgentRuntime:
    reg = _MockAgentRegistry()
    mem_store = _MockMemoryStore()
    tool_reg = _MockToolRegistry()
    agent = _TestAgent()

    reg.register(name="test_agent", metadata_id="test_agent")
    mem_store.create_memory(memory_id="mem1")
    rt = AgentRuntime(
        agent_registry=reg,
        memory_store=mem_store,
        tool_registry=tool_reg,
    )
    rt._create_agent = lambda agent_type: agent
    return rt


# -- Return type -------------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_task_stop_event_and_queue(
    runtime_with_agent: AgentRuntime,
) -> None:
    task, stop_event, event_queue = runtime_with_agent.run(
        user_input=_input(),
        agent_metadata_id="test_agent",
        memory_id="mem1",
    )
    assert isinstance(task, asyncio.Task)
    assert isinstance(stop_event, asyncio.Event)
    assert isinstance(event_queue, asyncio.Queue)


# -- Argument forwarding -----------------------------------------------


@pytest.mark.asyncio
async def test_run_forwards_args_to_agent(runtime: AgentRuntime) -> None:
    reg = runtime._agent_registry
    mem_store = runtime._memory_store
    tool_reg = runtime._tool_registry

    mock_tool = MagicMock(spec=Tool)
    mock_tool.name = "mock_tool"
    tool_reg.register(mock_tool)

    agent = _TestAgent()
    reg.register(
        name="test_agent",
        metadata_id="test_agent",
        tool_names=["mock_tool"],
    )
    mem_store.create_memory(memory_id="mem1")

    runtime._create_agent = lambda agent_type: agent

    user_input = _input("hi")

    task, stop_event, event_queue = runtime.run(
        user_input=user_input,
        agent_metadata_id="test_agent",
        memory_id="mem1",
    )

    await event_queue.get()  # None sentinel

    assert agent.run_args is not None
    (
        forwarded_input,
        forwarded_stop,
        forwarded_memory,
        forwarded_tools,
        forwarded_system_prompt,
    ) = agent.run_args
    assert forwarded_input == user_input
    assert mock_tool in forwarded_tools  # tool resolved from registry
    assert any(t.name == "handoff" for t in forwarded_tools if hasattr(t, "name"))
    assert any(
        t.name == "discover_agents" for t in forwarded_tools if hasattr(t, "name")
    )
    assert forwarded_stop is stop_event


# -- Event streaming ---------------------------------------------------


@pytest.mark.asyncio
async def test_run_streams_events_through_queue(
    runtime_with_agent: AgentRuntime,
) -> None:
    task, stop_event, event_queue = runtime_with_agent.run(
        user_input=[],
        agent_metadata_id="test_agent",
        memory_id="mem1",
    )

    event = await event_queue.get()
    assert event is None  # _TestAgent yields no events by default


# -- Sentinel ----------------------------------------------------------


@pytest.mark.asyncio
async def test_run_sends_none_sentinel_when_done(
    runtime_with_agent: AgentRuntime,
) -> None:
    task, stop_event, event_queue = runtime_with_agent.run(
        user_input=[],
        agent_metadata_id="test_agent",
        memory_id="mem1",
    )

    sentinel = await event_queue.get()
    assert sentinel is None


# -- Stop event --------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_event_halts_agent(runtime: AgentRuntime) -> None:
    reg = runtime._agent_registry
    mem_store = runtime._memory_store

    agent = _SlowAgent([{"type": "chunk"}])
    reg.register(name="test_agent", metadata_id="test_agent")
    mem_store.create_memory(memory_id="mem1")

    runtime._create_agent = lambda agent_type: agent

    task, stop_event, event_queue = runtime.run(
        user_input=[],
        agent_metadata_id="test_agent",
        memory_id="mem1",
    )

    first = await event_queue.get()
    assert first == {"type": "chunk"}

    stop_event.set()

    sentinel = await event_queue.get()
    assert sentinel is None  # agent stopped early


# -- Statelessness -----------------------------------------------------


@pytest.mark.asyncio
async def test_consecutive_runs_are_independent(runtime: AgentRuntime) -> None:
    reg = runtime._agent_registry
    mem_store = runtime._memory_store

    agent_a = _TestAgent(["from-a"])
    agent_b = _TestAgent(["from-b"])

    reg.register(name="agent_a", metadata_id="agent_a")
    reg.register(name="agent_b", metadata_id="agent_b")
    mem_store.create_memory(memory_id="mem_a")
    mem_store.create_memory(memory_id="mem_b")

    create_calls: list[str] = []

    def _create_agent(agent_type: str) -> _TestAgent:
        create_calls.append(agent_type)
        return [agent_a, agent_b][len(create_calls) - 1]

    runtime._create_agent = _create_agent

    task_a, stop_a, queue_a = runtime.run(
        user_input=[],
        agent_metadata_id="agent_a",
        memory_id="mem_a",
    )
    task_b, stop_b, queue_b = runtime.run(
        user_input=[],
        agent_metadata_id="agent_b",
        memory_id="mem_b",
    )

    result_a = await queue_a.get()
    assert result_a == "from-a"
    assert await queue_a.get() is None

    result_b = await queue_b.get()
    assert result_b == "from-b"
    assert await queue_b.get() is None


# -- Protocol conformance ----------------------------------------------


def test_agent_runtime_conforms_to_protocol() -> None:
    reg = _MockAgentRegistry()
    mem_store = _MockMemoryStore()
    tool_reg = _MockToolRegistry()
    rt = AgentRuntime(
        agent_registry=reg,
        memory_store=mem_store,
        tool_registry=tool_reg,
    )
    rt._create_agent = lambda agent_type: _TestAgent()
    assert isinstance(rt, AgentRuntimeProtocol)

    class CustomRuntime:
        _agent_registry: Any = None
        _memory_store: Any = None
        _tool_registry: Any = None

        def run(
            self,
            user_input: Any,
            agent_metadata_id: str,
            memory_id: str,
            agent_type: Any = None,
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
