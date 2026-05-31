from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator
from unittest.mock import MagicMock

import pytest

from minimal_harness.agent.runtime import AgentRuntime, AgentRuntimeProtocol
from minimal_harness.memory import ExtendedInputContentPart
from minimal_harness.tool.built_in.runtime_tools import register_runtime_tools
from minimal_harness.types import AgentMetadata, LocalToolBinding, ToolMetadata


async def _dummy_fn(**kwargs: Any) -> AsyncIterator[Any]:
    if False:
        yield None


class _MockToolRegistry:
    """Minimal ToolRegistry stub for testing."""

    def __init__(self, tools: list[ToolMetadata] | None = None) -> None:
        self._tools: dict[str, ToolMetadata] = {t.name: t for t in (tools or [])}

    async def get(self, name: str) -> ToolMetadata | None:
        return self._tools.get(name)

    async def get_all(self) -> list[ToolMetadata]:
        return list(self._tools.values())

    async def names(self) -> list[str]:
        return list(self._tools.keys())

    async def register(self, metadata: ToolMetadata) -> None:
        self._tools[metadata.name] = metadata

    async def register_from_binding(
        self,
        name: str,
        description: str,
        parameters: dict,
        binding: Any,
        display_name: str | None = None,
        display_name_locale: dict[str, str] | None = None,
        description_locale: dict[str, str] | None = None,
    ) -> None:
        pass

    async def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    async def clear(self) -> None:
        self._tools.clear()


class _MockSessionStore:
    """Minimal SessionStore stub for testing."""

    def __init__(self) -> None:
        self._sessions: dict[str, Any] = {}

    async def create_session(
        self,
        session_id: str | None = None,
        agent_name: str = "",
        user_id: str = "",
        scenario_id: str | None = None,
    ):
        from uuid import uuid4

        mid = session_id or uuid4().hex
        ses = MagicMock()
        ses.memory_id = mid
        ses.session_id = mid
        ses.title = None
        ses.created_at = ""
        ses.agent_name = agent_name
        ses.user_id = user_id
        ses.scenario_id = scenario_id
        self._sessions[mid] = ses
        return ses

    async def get_session(self, session_id: str):
        return self._sessions.get(session_id)

    async def save_memory(self, memory, session_id, extra=None):
        self._sessions[session_id] = memory

    async def delete_session(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    async def list_sessions(self) -> list[Any]:
        return []


class _MockAgentRegistry:
    """Minimal AgentRegistryProtocol stub for testing."""

    def __init__(self, metadata_list: list[Any] | None = None) -> None:
        self._data: dict[str, Any] = {}
        self._name_to_id: dict[str, str] = {}
        for m in metadata_list or []:
            self._data[m.metadata_id] = m
            self._name_to_id[m.name] = m.metadata_id

    async def register(self, metadata: Any) -> Any:
        self._data[metadata.metadata_id] = metadata
        self._name_to_id[metadata.name] = metadata.metadata_id
        return metadata

    async def unregister(self, name: str) -> bool:
        mid = self._name_to_id.get(name, name)
        self._name_to_id.pop(name, None)
        return self._data.pop(mid, None) is not None

    async def get(self, name: str) -> Any | None:
        mid = self._name_to_id.get(name, name)
        return self._data.get(mid)

    async def get_all(self, exclude: str | None = None) -> list[Any]:
        if exclude is None:
            return list(self._data.values())
        exclude_key = self._name_to_id.get(exclude, exclude)
        return [v for k, v in self._data.items() if k != exclude_key]

    async def names(self) -> list[str]:
        return list(self._name_to_id.keys())

    async def clear(self) -> None:
        self._data.clear()
        self._name_to_id.clear()

    async def add_listener(self, listener: Any) -> None:
        pass

    async def remove_listener(self, listener: Any) -> None:
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
        context: Any = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
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
        context: Any = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        for event in self.events:
            if stop_event is not None and stop_event.is_set():
                break
            await asyncio.sleep(self._sleep)
            yield event


def _input(text: str = "hi") -> list[ExtendedInputContentPart]:
    return [{"type": "text", "text": text}]


@pytest.fixture
async def runtime() -> AgentRuntime:
    reg = _MockAgentRegistry()
    ses_store = _MockSessionStore()
    tool_reg = _MockToolRegistry()
    rt = AgentRuntime(
        agent_registry=reg,
        session_store=ses_store,
        tool_registry=tool_reg,
        llm_provider_resolver=lambda _: (
            MagicMock()
        ),  # never called; _create_agent overridden
    )
    rt._create_agent = lambda metadata, middleware=None: _TestAgent()
    return rt


@pytest.fixture
async def runtime_with_agent() -> AgentRuntime:
    from minimal_harness.agent.registry import AgentMetadata

    reg = _MockAgentRegistry()
    mem_store = _MockSessionStore()
    tool_reg = _MockToolRegistry()
    agent = _TestAgent()

    await reg.register(AgentMetadata(name="test_agent", metadata_id="test_agent"))
    await mem_store.create_session(session_id="mem1")
    rt = AgentRuntime(
        agent_registry=reg,
        session_store=mem_store,
        tool_registry=tool_reg,
        llm_provider_resolver=lambda _: (
            MagicMock()
        ),  # never called; _create_agent overridden
    )
    rt._create_agent = lambda metadata, middleware=None: agent
    return rt


# -- Return type -------------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_task_stop_event_and_queue(
    runtime_with_agent: AgentRuntime,
) -> None:
    task, stop_event, event_queue = await runtime_with_agent.run(
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
    reg = runtime.agent_registry
    ses_store = runtime.session_store
    tool_reg = runtime.tool_registry

    mock_tool_meta = ToolMetadata(
        name="mock_tool",
        description="Mock tool",
        parameters={"type": "object", "properties": {}},
        binding=LocalToolBinding(fn=_dummy_fn),
    )
    await tool_reg.register(mock_tool_meta)

    await register_runtime_tools(
        agent_registry=reg,
        session_store=ses_store,
        tool_registry=tool_reg,
        run_fn=runtime.run,
    )

    agent = _TestAgent()
    await reg.register(
        AgentMetadata(
            name="test_agent",
            metadata_id="test_agent",
            tool_names=["mock_tool"],
        )
    )
    await ses_store.create_session(session_id="mem1")

    runtime._create_agent = lambda metadata, middleware=None: agent

    user_input = _input("hi")

    task, stop_event, event_queue = await runtime.run(
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
    assert any(t.name == "mock_tool" for t in forwarded_tools)
    assert forwarded_stop is stop_event


# -- Event streaming ---------------------------------------------------


@pytest.mark.asyncio
async def test_run_streams_events_through_queue(
    runtime_with_agent: AgentRuntime,
) -> None:
    task, stop_event, event_queue = await runtime_with_agent.run(
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
    task, stop_event, event_queue = await runtime_with_agent.run(
        user_input=[],
        agent_metadata_id="test_agent",
        memory_id="mem1",
    )

    sentinel = await event_queue.get()
    assert sentinel is None


# -- Stop event --------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_event_halts_agent(runtime: AgentRuntime) -> None:
    reg = runtime.agent_registry
    ses_store = runtime.session_store

    agent = _SlowAgent([{"type": "chunk"}])
    await reg.register(AgentMetadata(name="test_agent", metadata_id="test_agent"))
    await ses_store.create_session(session_id="mem1")

    runtime._create_agent = lambda metadata, middleware=None: agent

    task, stop_event, event_queue = await runtime.run(
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
    reg = runtime.agent_registry
    ses_store = runtime.session_store

    agent_a = _TestAgent(["from-a"])
    agent_b = _TestAgent(["from-b"])

    await reg.register(AgentMetadata(name="agent_a", metadata_id="agent_a"))
    await reg.register(AgentMetadata(name="agent_b", metadata_id="agent_b"))
    await ses_store.create_session(session_id="mem_a")
    await ses_store.create_session(session_id="mem_b")

    create_calls: list[str] = []

    def _create_agent_func(
        metadata: AgentMetadata, middleware: Any = None
    ) -> _TestAgent:
        create_calls.append(metadata.agent_type)
        return [agent_a, agent_b][len(create_calls) - 1]

    runtime._create_agent = _create_agent_func

    task_a, stop_a, queue_a = await runtime.run(
        user_input=[],
        agent_metadata_id="agent_a",
        memory_id="mem_a",
    )
    task_b, stop_b, queue_b = await runtime.run(
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


@pytest.mark.asyncio
async def test_agent_runtime_conforms_to_protocol() -> None:
    reg = _MockAgentRegistry()
    ses_store = _MockSessionStore()
    tool_reg = _MockToolRegistry()
    rt = AgentRuntime(
        agent_registry=reg,
        session_store=ses_store,
        tool_registry=tool_reg,
        llm_provider_resolver=lambda _: (
            MagicMock()
        ),  # never called; _create_agent overridden
    )
    rt._create_agent = lambda metadata, middleware=None: _TestAgent()
    assert isinstance(rt, AgentRuntimeProtocol)

    class CustomRuntime:
        agent_registry: Any = None
        session_store: Any = None
        tool_registry: Any = None

        async def run(
            self,
            user_input: Any,
            agent_metadata_id: str,
            memory_id: str,
            agent_type: Any = None,
            tool_names: Any = None,
            context: Any = None,
            llm_kwargs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> tuple[asyncio.Task, asyncio.Event, asyncio.Queue]:
            return (
                asyncio.create_task(asyncio.sleep(0)),
                asyncio.Event(),
                asyncio.Queue(),
            )

    assert isinstance(CustomRuntime(), AgentRuntimeProtocol)


@pytest.mark.asyncio
async def test_agent_runtime_protocol_requires_run() -> None:
    class BadRuntime:
        pass

    assert not isinstance(BadRuntime(), AgentRuntimeProtocol)
