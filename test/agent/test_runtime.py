from __future__ import annotations

import asyncio
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from minimal_harness.agent.registry import AgentRegistry, Session
from minimal_harness.agent.runtime import AgentRuntime
from minimal_harness.types import ToolCall, ToolEnd


@pytest.fixture
def runtime() -> AgentRuntime:
    return AgentRuntime(AgentRegistry())


def _fake_sid(suffix: str = "abc") -> str:
    return f"test-session-{suffix}"


class TestCreateSession:
    def test_creates_session_with_unique_ids(self, runtime: AgentRuntime) -> None:
        s1 = _make_session(runtime, sid="id-1")
        s2 = _make_session(runtime, sid="id-2")

        assert s1.session_id == "id-1"
        assert s2.session_id == "id-2"
        assert s1.session_id != s2.session_id

    def test_stores_sessions_in_registry(self, runtime: AgentRuntime) -> None:
        s1 = _make_session(runtime, sid="a")
        s2 = _make_session(runtime, sid="b")

        assert runtime.get_session("a") is s1
        assert runtime.get_session("b") is s2
        assert runtime.get_session("nonexistent") is None

    def test_list_sessions(self, runtime: AgentRuntime) -> None:
        s1 = _make_session(runtime, sid="a")
        s2 = _make_session(runtime, sid="b")

        sessions = runtime.list_sessions()
        assert len(sessions) == 2
        assert s1 in sessions
        assert s2 in sessions

    def test_no_implicit_current_session(self, runtime: AgentRuntime) -> None:
        s1 = _make_session(runtime, sid="a")
        _make_session(runtime, sid="b")

        assert runtime.get_session("a") is s1
        assert runtime.get_session("b") is not s1
        assert not hasattr(runtime, "_current")


class TestGetSession:
    def test_returns_none_for_missing_id(self, runtime: AgentRuntime) -> None:
        assert runtime.get_session("missing") is None

    def test_returns_correct_session(self, runtime: AgentRuntime) -> None:
        s1 = _make_session(runtime, sid="x")
        s2 = _make_session(runtime, sid="y")

        assert runtime.get_session("x") is s1
        assert runtime.get_session("y") is s2
        assert runtime.get_session("x") is s1  # still s1


class TestRunSessionIdRequired:
    @pytest.mark.asyncio
    async def test_requires_session_id(self, runtime: AgentRuntime) -> None:
        _make_session(runtime, sid="s1")
        events = []
        async for _ in runtime.run(
            user_input=[{"type": "text", "text": "hello"}],
            session_id=None,
        ):
            events.append(_)
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_requires_valid_session_id(self, runtime: AgentRuntime) -> None:
        _make_session(runtime, sid="s1")
        events = []
        async for _ in runtime.run(
            user_input=[{"type": "text", "text": "hello"}],
            session_id="nonexistent",
        ):
            events.append(_)
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_runs_specified_session_agent(self, runtime: AgentRuntime) -> None:
        s1 = _make_session(runtime, sid="s1")
        s2 = _make_session(runtime, sid="s2")

        events = []
        async for _ in runtime.run(
            user_input=[{"type": "text", "text": "hi"}],
            session_id="s1",
        ):
            events.append(_)

        s1_mock = s1.agent
        s2_mock = s2.agent
        s1_mock.run.assert_called_once()  # type: ignore[reportAttributeAccessIssue]
        s2_mock.run.assert_not_called()  # type: ignore[reportAttributeAccessIssue]


class TestRunningTasks:
    def test_no_running_tasks_initially(self, runtime: AgentRuntime) -> None:
        assert runtime.get_running_session_ids() == []

    def test_is_session_running(self, runtime: AgentRuntime) -> None:
        _make_session(runtime, sid="s1")
        assert not runtime.is_session_running("s1")
        assert not runtime.is_session_running("nonexistent")


class TestLoadSession:
    @patch("minimal_harness.client.built_in.memory.PersistentMemory")
    def test_load_session_adds_to_sessions(
        self, mock_mem_cls: MagicMock, runtime: AgentRuntime
    ) -> None:
        mock_memory = MagicMock()
        mock_memory._session_id = "loaded-id"
        mock_memory.title = "Loaded"
        mock_mem_cls.from_session.return_value = mock_memory

        config = {"provider": "openai", "model": "gpt-4", "api_key": "test"}
        tools: list = []

        session = runtime.load_session(
            session_id="loaded-id",
            config=config,
            tools=tools,
            agent_factory=_fake_agent_factory,
        )

        assert session.session_id == "loaded-id"
        assert runtime.get_session("loaded-id") is session

    @patch("minimal_harness.client.built_in.memory.PersistentMemory")
    def test_multiple_loads_have_different_sessions(
        self, mock_mem_cls: MagicMock, runtime: AgentRuntime
    ) -> None:
        mock_mem_cls.from_session.side_effect = [
            MagicMock(_session_id="s1", title="First"),
            MagicMock(_session_id="s2", title="Second"),
        ]

        config = {"provider": "openai", "model": "gpt-4", "api_key": "test"}
        tools: list = []

        s1 = runtime.load_session("s1", config, tools, _fake_agent_factory)
        s2 = runtime.load_session("s2", config, tools, _fake_agent_factory)

        assert s1.session_id == "s1"
        assert s2.session_id == "s2"
        assert s1.session_id != s2.session_id

    @patch("minimal_harness.client.built_in.memory.PersistentMemory")
    def test_loaded_session_runnable(
        self, mock_mem_cls: MagicMock, runtime: AgentRuntime
    ) -> None:
        mock_memory = MagicMock()
        mock_memory._session_id = "lid"
        mock_memory.title = "Loaded"
        mock_mem_cls.from_session.return_value = mock_memory

        config = {"provider": "openai", "model": "gpt-4", "api_key": "test"}
        tools: list = []

        session = runtime.load_session("lid", config, tools, _fake_agent_factory)

        session.agent.run.assert_not_called()  # type: ignore[reportAttributeAccessIssue]
        assert runtime.get_session("lid") is session


class TestHandoffTool:
    def test_create_handoff_tool(self, runtime: AgentRuntime) -> None:
        tool = runtime.create_handoff_tool()

        assert tool.name == "handoff"
        assert "target_session_id" in tool.parameters["properties"]
        assert "context_summary" in tool.parameters["properties"]
        assert "task_description" in tool.parameters["properties"]
        assert tool.parameters["required"] == [
            "target_session_id",
            "context_summary",
            "task_description",
        ]

    def test_handoff_tool_schema(self, runtime: AgentRuntime) -> None:
        tool = runtime.create_handoff_tool()

        schema = tool.to_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "handoff"
        assert "target_session_id" in schema["function"]["parameters"]["properties"]
        assert "context_summary" in schema["function"]["parameters"]["properties"]
        assert "task_description" in schema["function"]["parameters"]["properties"]

    def test_handoff_tool_anthropic_schema(self, runtime: AgentRuntime) -> None:
        tool = runtime.create_handoff_tool()

        schema = tool.to_anthropic_schema()
        assert schema["name"] == "handoff"
        assert "input_schema" in schema

    @pytest.mark.asyncio
    async def test_handoff_execute_combines_args_and_starts_background_task(
        self, runtime: AgentRuntime
    ) -> None:
        session_id = "test-session-abc"
        session = _make_session(runtime, sid=session_id)
        session.agent.run.return_value.__aiter__.return_value = iter([])  # type: ignore[reportAttributeAccessIssue]
        tool = runtime.create_handoff_tool()

        tool_call = cast(ToolCall, {"id": "call_123", "name": "handoff", "input": {}})
        args = {
            "target_session_id": session_id,
            "context_summary": "Current file is app.py",
            "task_description": "Refactor the login function",
        }

        events = []
        async for event in tool.execute(args, tool_call, stop_event=None):
            events.append(event)

        task = runtime._running_tasks.get(session_id)
        if task:
            await task

        assert events[1].chunk["status"] == "handoff"
        assert runtime.is_session_running(session_id) is False

    @pytest.mark.asyncio
    async def test_handoff_execute_yields_handoff_status(
        self, runtime: AgentRuntime
    ) -> None:
        async def empty_gen():
            return
            yield

        session_id = "test-session-end"
        _make_session(runtime, sid=session_id)
        tool = runtime.create_handoff_tool()

        with patch.object(runtime, "run") as mock_run:
            mock_run.return_value = empty_gen()

            tool_call = cast(
                ToolCall, {"id": "call_456", "name": "handoff", "input": {}}
            )
            args = {
                "target_session_id": session_id,
                "context_summary": "ctx",
                "task_description": "task",
            }

            events = []
            async for event in tool.execute(args, tool_call, stop_event=None):
                events.append(event)

            assert len(events) == 3
            assert events[0].tool_call == tool_call
            assert events[1].chunk["status"] == "handoff"
            assert isinstance(events[2], ToolEnd)

    @pytest.mark.asyncio
    async def test_handoff_execute_accumulates_events_in_session_queue(
        self, runtime: AgentRuntime
    ) -> None:
        inner_event = {"type": "agent_start", "agent": "test"}

        session_id = "test-session-fwd"
        session = _make_session(runtime, sid=session_id)
        session.agent.run.return_value.__aiter__.return_value = iter([inner_event])  # type: ignore[reportAttributeAccessIssue]
        tool = runtime.create_handoff_tool()

        tool_call = cast(ToolCall, {"id": "call_789", "name": "handoff", "input": {}})
        args = {
            "target_session_id": session_id,
            "context_summary": "ctx",
            "task_description": "task",
        }

        async for _ in tool.execute(args, tool_call, stop_event=None):
            pass

        task = runtime._running_tasks.get(session_id)
        if task:
            await task

        assert session.has_events() is True
        queued = await session.drain_events()
        assert inner_event in queued

    @pytest.mark.asyncio
    async def test_handoff_execute_invalid_session(self, runtime: AgentRuntime) -> None:
        session_id = "nonexistent"
        tool = runtime.create_handoff_tool()

        tool_call = cast(ToolCall, {"id": "call_err", "name": "handoff", "input": {}})
        args = {
            "target_session_id": session_id,
            "context_summary": "ctx",
            "task_description": "task",
        }

        events = []
        async for event in tool.execute(args, tool_call, stop_event=None):
            events.append(event)

        assert events[1].chunk["status"] == "error"
        assert "not found" in events[1].chunk["message"]


def _make_session(
    runtime: AgentRuntime,
    sid: str,
) -> Session:
    memory = MagicMock()
    memory._session_id = sid
    memory.title = f"Session {sid}"
    return runtime.create_session(
        config={"provider": "openai", "model": "gpt-4", "api_key": "test"},
        tools=[],
        memory=memory,
        agent_factory=_fake_agent_factory,
    )


def _fake_agent_factory(**kwargs):
    agent = MagicMock()
    agent.run.return_value.__aiter__.return_value = iter([])
    return agent


class TestDiscoverAgentsTool:
    def test_create_discover_agents_tool(self, runtime: AgentRuntime) -> None:
        tool = runtime.create_discover_agents_tool()

        assert tool.name == "discover_agents"
        assert "Discover available agents" in tool.description

    def test_discover_agents_schema(self, runtime: AgentRuntime) -> None:
        tool = runtime.create_discover_agents_tool()

        schema = tool.to_schema()
        assert schema["function"]["name"] == "discover_agents"

    def test_discover_agents_anthropic_schema(self, runtime: AgentRuntime) -> None:
        tool = runtime.create_discover_agents_tool()

        schema = tool.to_anthropic_schema()
        assert schema["name"] == "discover_agents"

    @pytest.mark.asyncio
    async def test_discover_agents_returns_registered_agents(
        self, runtime: AgentRuntime
    ) -> None:
        from minimal_harness.agent.registry import AgentRegistry

        registry = AgentRegistry()
        registry.register(MagicMock(), name="writer", description="Writes content")
        registry.register(MagicMock(), name="coder", description="Writes code")
        runtime = AgentRuntime(registry)

        tool = runtime.create_discover_agents_tool()
        tool_call = cast(
            ToolCall,
            {
                "id": "disc_1",
                "type": "function",
                "function": {"name": "discover_agents", "arguments": "{}"},
            },
        )

        events = []
        async for event in tool.execute({}, tool_call, stop_event=None):
            events.append(event)

        result = events[1].chunk
        assert result["status"] == "ok"
        agents = [a for a in result["agents"] if a["type"] == "registered"]
        assert len(agents) == 2
        names = {a["name"] for a in agents}
        assert "writer" in names
        assert "coder" in names

    @pytest.mark.asyncio
    async def test_discover_agents_returns_sessions(
        self, runtime: AgentRuntime
    ) -> None:
        _make_session(runtime, sid="s1")
        _make_session(runtime, sid="s2")

        tool = runtime.create_discover_agents_tool()
        tool_call = cast(
            ToolCall,
            {
                "id": "disc_2",
                "type": "function",
                "function": {"name": "discover_agents", "arguments": "{}"},
            },
        )

        events = []
        async for event in tool.execute({}, tool_call, stop_event=None):
            events.append(event)

        result = events[1].chunk
        sessions = [s for s in result["agents"] if s["type"] == "session"]
        assert len(sessions) == 2
        sids = {s["session_id"] for s in sessions}
        assert "s1" in sids
        assert "s2" in sids

    @pytest.mark.asyncio
    async def test_discover_agents_shows_running_status(
        self, runtime: AgentRuntime
    ) -> None:
        _make_session(runtime, sid="s1")
        runtime._running_tasks["s1"] = asyncio.create_task(asyncio.sleep(0))

        tool = runtime.create_discover_agents_tool()
        tool_call = cast(
            ToolCall,
            {
                "id": "disc_3",
                "type": "function",
                "function": {"name": "discover_agents", "arguments": "{}"},
            },
        )
        events = []
        async for event in tool.execute({}, tool_call, stop_event=None):
            events.append(event)

        result = events[1].chunk
        s1_info = next(s for s in result["agents"] if s["session_id"] == "s1")
        assert s1_info["running"] is True

        runtime._running_tasks.pop("s1", None)


class TestRuntimeToolInjection:
    def test_inject_runtime_tools_adds_handoff_and_discover(
        self, runtime: AgentRuntime
    ) -> None:
        session = _make_session(runtime, sid="inj-1")
        session.tools = []

        runtime.inject_runtime_tools(session)

        tool_names = {t.name for t in session.tools}
        assert "handoff" in tool_names
        assert "discover_agents" in tool_names

    def test_inject_runtime_tools_skips_existing(self, runtime: AgentRuntime) -> None:
        existing = runtime.create_handoff_tool()
        session = _make_session(runtime, sid="inj-2")
        session.tools = [existing]

        runtime.inject_runtime_tools(session)

        handoff_tools = [t for t in session.tools if t.name == "handoff"]
        assert len(handoff_tools) == 1
        assert handoff_tools[0] is existing

    @pytest.mark.asyncio
    async def test_run_injects_runtime_tools(self, runtime: AgentRuntime) -> None:
        session = _make_session(runtime, sid="inj-run")
        session.tools = []

        mock_event = {"type": "agent_start", "agent": "test"}
        session.agent.run.return_value.__aiter__.return_value = iter([mock_event])  # type: ignore[reportAttributeAccessIssue]

        async for _ in runtime.run(
            user_input=[{"type": "text", "text": "hello"}],
            session_id="inj-run",
        ):
            pass

        tool_names = {t.name for t in session.tools}
        assert "handoff" in tool_names
        assert "discover_agents" in tool_names

    def test_inject_runtime_tools_with_custom_names(
        self, runtime: AgentRuntime
    ) -> None:
        session = _make_session(runtime, sid="inj-custom")
        session.tools = []

        runtime.inject_runtime_tools(session, tool_names=("discover_agents",))

        tool_names = {t.name for t in session.tools}
        assert "handoff" not in tool_names
        assert "discover_agents" in tool_names


class TestSessionEventQueue:
    def test_session_has_event_queue_by_default(self) -> None:
        from minimal_harness.client.built_in.memory import PersistentMemory

        memory = PersistentMemory(session_id=_fake_sid("queue1"))
        session = Session(
            session_id=memory._session_id,
            name="Test",
            agent=MagicMock(),
            memory=memory,
            tools=[],
        )
        assert session.event_queue is not None
        assert session.has_events() is False

    def test_session_drain_events_returns_queued_events(self) -> None:
        from minimal_harness.client.built_in.memory import PersistentMemory

        memory = PersistentMemory(session_id=_fake_sid("queue2"))
        session = Session(
            session_id=memory._session_id,
            name="Test",
            agent=MagicMock(),
            memory=memory,
            tools=[],
        )
        mock_event = {"type": "agent_start", "agent": "test"}
        session.event_queue.put_nowait(mock_event)  # type: ignore[union-attr]
        assert session.has_events() is True
        events = session.event_queue.get_nowait()  # type: ignore[union-attr]
        assert events == mock_event

    @pytest.mark.asyncio
    async def test_session_drain_events_returns_list(self) -> None:
        from minimal_harness.client.built_in.memory import PersistentMemory

        memory = PersistentMemory(session_id=_fake_sid("queue3"))
        session = Session(
            session_id=memory._session_id,
            name="Test",
            agent=MagicMock(),
            memory=memory,
            tools=[],
        )
        mock_event_1 = {"type": "agent_start", "agent": "test1"}
        mock_event_2 = {"type": "agent_end", "agent": "test2"}
        session.event_queue.put_nowait(mock_event_1)  # type: ignore[union-attr]
        session.event_queue.put_nowait(mock_event_2)  # type: ignore[union-attr]
        events = await session.drain_events()
        assert len(events) == 2
        assert events[0] == mock_event_1
        assert events[1] == mock_event_2
        assert session.has_events() is False


class TestRuntimeRunEnqueuesEvents:
    @pytest.mark.asyncio
    async def test_run_enqueues_events_to_session(self, runtime: AgentRuntime) -> None:
        from minimal_harness.client.built_in.memory import PersistentMemory

        memory = PersistentMemory(session_id=_fake_sid("enqueue1"))
        session = runtime.create_session(
            config={"provider": "openai", "model": "gpt-4", "api_key": "test"},
            tools=[],
            memory=memory,
            agent_factory=_fake_agent_factory,
        )
        mock_event = {"type": "agent_start", "agent": "test"}
        session.agent.run.return_value.__aiter__.return_value = iter([mock_event])  # type: ignore[reportAttributeAccessIssue]

        events = []
        async for event in runtime.run(
            user_input=[{"type": "text", "text": "hello"}],
            session_id=session.session_id,
        ):
            events.append(event)

        assert len(events) == 1
        assert events[0] == mock_event
        assert session.has_events() is True

    @pytest.mark.asyncio
    async def test_run_with_no_session_id_does_not_enqueue(
        self, runtime: AgentRuntime
    ) -> None:
        from minimal_harness.client.built_in.memory import PersistentMemory

        memory = PersistentMemory(session_id=_fake_sid("enqueue2"))
        session = runtime.create_session(
            config={"provider": "openai", "model": "gpt-4", "api_key": "test"},
            tools=[],
            memory=memory,
            agent_factory=_fake_agent_factory,
        )
        events = []
        async for event in runtime.run(
            user_input=[{"type": "text", "text": "hello"}],
            session_id=None,
        ):
            events.append(event)

        assert events == []
        assert session.has_events() is False


class TestRegisterAgent:
    def test_register_agent_creates_session_and_registers(
        self, runtime: AgentRuntime
    ) -> None:
        sid = runtime.register_agent(
            name="test-agent",
            description="A test agent",
            system_prompt="You are a test agent.",
            llm_provider=MagicMock(),
            tools=[],
            agent_factory=_fake_agent_factory,
        )

        assert runtime.get_session(sid) is not None
        meta = runtime.registry.get("test-agent")
        assert meta is not None
        assert meta.name == "test-agent"
        assert meta.description == "A test agent"

    def test_register_agent_returns_valid_session_id(
        self, runtime: AgentRuntime
    ) -> None:
        sid = runtime.register_agent(
            name="agent-b",
            description="",
            system_prompt="prompt",
            llm_provider=MagicMock(),
            tools=[],
            agent_factory=_fake_agent_factory,
        )

        assert isinstance(sid, str)
        assert len(sid) > 0

    @pytest.mark.asyncio
    async def test_register_agent_appears_in_discover(
        self, runtime: AgentRuntime
    ) -> None:
        runtime.register_agent(
            name="discover-me",
            description="Findable agent",
            system_prompt="prompt",
            llm_provider=MagicMock(),
            tools=[],
            agent_factory=_fake_agent_factory,
        )
        tool = runtime.create_discover_agents_tool()
        tool_call = cast(
            ToolCall,
            {
                "id": "disc_reg",
                "type": "function",
                "function": {"name": "discover_agents", "arguments": "{}"},
            },
        )
        events = []
        async for event in tool.execute({}, tool_call, stop_event=None):
            events.append(event)

        result = events[1].chunk
        assert result["status"] == "ok"
        agents = [a for a in result["agents"] if a.get("type") == "registered"]
        names = {a["name"] for a in agents}
        assert "discover-me" in names

    def test_register_agent_uses_provided_system_prompt(
        self, runtime: AgentRuntime
    ) -> None:
        llm = MagicMock()
        runtime.register_agent(
            name="prompt-check",
            description="Check system prompt",
            system_prompt="Custom system prompt for testing",
            llm_provider=llm,
            tools=[],
            agent_factory=_fake_agent_factory,
        )

        meta = runtime.registry.get("prompt-check")
        assert meta is not None
        assert meta.name == "prompt-check"
