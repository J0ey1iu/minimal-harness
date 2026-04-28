from __future__ import annotations

import asyncio
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from minimal_harness.agent.registry import AgentRegistry, HandoffTarget
from minimal_harness.agent.runtime import AgentRuntime
from minimal_harness.types import ToolCall, ToolEnd


@pytest.fixture
def runtime() -> AgentRuntime:
    return AgentRuntime(AgentRegistry())


def _fake_sid(suffix: str = "abc") -> str:
    return f"test-session-{suffix}"


def _make_handoff_target(
    runtime: AgentRuntime,
    sid: str,
    name: str | None = None,
    tools: list | None = None,
) -> HandoffTarget:
    memory = MagicMock()
    memory._session_id = sid
    memory.title = f"Session {sid}"
    agent = _fake_agent_factory()
    target = HandoffTarget(
        session_id=sid,
        name=name or f"agent-{sid}",
        agent=agent,
        memory=memory,
        tools=tools or [],
    )
    runtime._handoff_targets[sid] = target
    return target


def _fake_agent_factory(**kwargs):
    agent = MagicMock()
    agent.run.return_value.__aiter__.return_value = iter([])
    return agent


class TestHandoffTool:
    def test_create_handoff_tool(self, runtime: AgentRuntime) -> None:
        tool = runtime.create_handoff_tool()

        assert tool.name == "handoff"
        assert "target_agent_name" in tool.parameters["properties"]
        assert "context_summary" in tool.parameters["properties"]
        assert "task_description" in tool.parameters["properties"]
        assert tool.parameters["required"] == [
            "target_agent_name",
            "context_summary",
            "task_description",
        ]

    def test_handoff_tool_schema(self, runtime: AgentRuntime) -> None:
        tool = runtime.create_handoff_tool()

        schema = tool.to_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "handoff"
        assert "target_agent_name" in schema["function"]["parameters"]["properties"]
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
        _make_handoff_target(runtime, sid=session_id, name="agent-abc")
        tool = runtime.create_handoff_tool()

        tool_call = cast(ToolCall, {"id": "call_123", "name": "handoff", "input": {}})
        args = {
            "target_agent_name": "agent-abc",
            "context_summary": "Current file is app.py",
            "task_description": "Refactor the login function",
        }

        events = []
        async for event in tool.execute(args, tool_call, stop_event=None):
            events.append(event)

        task = runtime._background_tasks.get(session_id)
        if task:
            await task

        assert events[1].chunk["status"] == "handoff"
        assert runtime.is_background_task_running(session_id) is False

    @pytest.mark.asyncio
    async def test_handoff_execute_yields_handoff_status(
        self, runtime: AgentRuntime
    ) -> None:
        async def empty_gen():
            return
            yield

        session_id = "test-session-end"
        _make_handoff_target(runtime, sid=session_id, name="agent-end")
        tool = runtime.create_handoff_tool()

        with patch.object(runtime, "run_background") as mock_run:
            mock_run.return_value = None

            tool_call = cast(
                ToolCall, {"id": "call_456", "name": "handoff", "input": {}}
            )
            args = {
                "target_agent_name": "agent-end",
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
    async def test_handoff_execute_accumulates_events_in_target_queue(
        self, runtime: AgentRuntime
    ) -> None:
        inner_event = {"type": "agent_start", "agent": "test"}

        session_id = "test-session-fwd"
        target = _make_handoff_target(runtime, sid=session_id, name="agent-fwd")
        target.agent.run.return_value.__aiter__.return_value = iter([inner_event])  # type: ignore[reportAttributeAccessIssue]
        tool = runtime.create_handoff_tool()

        tool_call = cast(ToolCall, {"id": "call_789", "name": "handoff", "input": {}})
        args = {
            "target_agent_name": "agent-fwd",
            "context_summary": "ctx",
            "task_description": "task",
        }

        async for _ in tool.execute(args, tool_call, stop_event=None):
            pass

        task = runtime._background_tasks.get(session_id)
        if task:
            await task

        assert not target.event_queue.empty()
        queued = target.event_queue.get_nowait()
        assert queued == inner_event

    @pytest.mark.asyncio
    async def test_handoff_execute_invalid_session(self, runtime: AgentRuntime) -> None:
        tool = runtime.create_handoff_tool()

        tool_call = cast(ToolCall, {"id": "call_err", "name": "handoff", "input": {}})
        args = {
            "target_agent_name": "nonexistent-agent",
            "context_summary": "ctx",
            "task_description": "task",
        }

        events = []
        async for event in tool.execute(args, tool_call, stop_event=None):
            events.append(event)

        assert events[1].chunk["status"] == "error"
        assert "not found" in events[1].chunk["message"]


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
        registry = AgentRegistry()
        registry.register(MagicMock(), name="writer", description="Writes content")
        registry.register(MagicMock(), name="coder", description="Writes code")
        runtime = AgentRuntime(registry)

        runtime._handoff_targets["s1"] = HandoffTarget(
            session_id="s1",
            name="writer",
            agent=MagicMock(),
            memory=MagicMock(),
            tools=[],
        )
        runtime._handoff_targets["s2"] = HandoffTarget(
            session_id="s2",
            name="coder",
            agent=MagicMock(),
            memory=MagicMock(),
            tools=[],
        )

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
        agents = result["agents"]
        assert len(agents) == 2
        names = {a["name"] for a in agents}
        assert "writer" in names
        assert "coder" in names
        assert (
            next(a for a in agents if a["name"] == "writer")["description"]
            == "Writes content"
        )

    @pytest.mark.asyncio
    async def test_discover_agents_returns_handoff_targets(
        self, runtime: AgentRuntime
    ) -> None:
        _make_handoff_target(runtime, sid="s1", name="agent-one")
        _make_handoff_target(runtime, sid="s2", name="agent-two")

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
        agents = result["agents"]
        assert len(agents) == 2
        names = {a["name"] for a in agents}
        assert "agent-one" in names
        assert "agent-two" in names
        assert all("session_id" not in a for a in agents)
        assert all("running" in a for a in agents)
        assert all("description" in a for a in agents)

    @pytest.mark.asyncio
    async def test_discover_agents_shows_running_status(
        self, runtime: AgentRuntime
    ) -> None:
        _make_handoff_target(runtime, sid="s1", name="agent-one")
        runtime._background_tasks["s1"] = asyncio.create_task(asyncio.sleep(0))

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
        s1_info = next(s for s in result["agents"] if s["name"] == "agent-one")
        assert s1_info["running"] is True
        assert "session_id" not in s1_info

        runtime._background_tasks.pop("s1", None)


class TestRuntimeToolInjection:
    def test_inject_runtime_tools_adds_handoff_and_discover(
        self, runtime: AgentRuntime
    ) -> None:
        tools: list = []

        runtime.inject_runtime_tools(tools)

        tool_names = {t.name for t in tools}
        assert "handoff" in tool_names
        assert "discover_agents" in tool_names

    def test_inject_runtime_tools_skips_existing(self, runtime: AgentRuntime) -> None:
        existing = runtime.create_handoff_tool()
        tools: list = [existing]

        runtime.inject_runtime_tools(tools)

        handoff_tools = [t for t in tools if t.name == "handoff"]
        assert len(handoff_tools) == 1
        assert handoff_tools[0] is existing

    def test_inject_runtime_tools_with_custom_names(
        self, runtime: AgentRuntime
    ) -> None:
        tools: list = []

        runtime.inject_runtime_tools(tools, tool_names=("discover_agents",))

        tool_names = {t.name for t in tools}
        assert "handoff" not in tool_names
        assert "discover_agents" in tool_names


class TestHandoffTargetEventQueue:
    def test_handoff_target_has_event_queue_by_default(self) -> None:
        target = HandoffTarget(
            session_id="test-queue",
            name="Test",
            agent=MagicMock(),
            memory=MagicMock(),
            tools=[],
        )
        assert target.event_queue is not None
        assert target.event_queue.empty()

    def test_handoff_target_event_queue_put_and_get(self) -> None:
        target = HandoffTarget(
            session_id="test-queue2",
            name="Test",
            agent=MagicMock(),
            memory=MagicMock(),
            tools=[],
        )
        mock_event = {"type": "agent_start", "agent": "test"}
        target.event_queue.put_nowait(mock_event)  # type: ignore[reportArgumentType]
        assert not target.event_queue.empty()
        event = target.event_queue.get_nowait()
        assert event == mock_event


class TestRun:
    @pytest.mark.asyncio
    async def test_run_requires_agent_and_memory(self, runtime: AgentRuntime) -> None:
        agent = _fake_agent_factory()
        memory = MagicMock()

        events = []
        async for event in runtime.run(
            agent=agent,
            user_input=[{"type": "text", "text": "hello"}],
            memory=memory,
            tools=[],
        ):
            events.append(event)

        agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_passes_correct_args(self, runtime: AgentRuntime) -> None:
        agent = _fake_agent_factory()
        memory = MagicMock()
        tools = []
        stop = asyncio.Event()

        async for _ in runtime.run(
            agent=agent,
            user_input=[{"type": "text", "text": "hi"}],
            memory=memory,
            tools=tools,
            stop_event=stop,
        ):
            pass

        agent.run.assert_called_once_with(
            user_input=[{"type": "text", "text": "hi"}],
            memory=memory,
            tools=tools,
            stop_event=stop,
        )


class TestRegisterAgent:
    def test_register_agent_creates_handoff_target_and_registers(
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

        assert runtime.get_handoff_target(sid) is not None
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
        names = {a["name"] for a in result["agents"]}
        assert "discover-me" in names
        info = next(a for a in result["agents"] if a["name"] == "discover-me")
        assert info["description"] == "Findable agent"

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


class TestHandoffEventCallback:
    def test_set_on_handoff_event_default_is_none(self, runtime: AgentRuntime) -> None:
        assert runtime._on_handoff_event is None

    def test_set_on_handoff_event_stores_callback(self, runtime: AgentRuntime) -> None:
        calls: list[str] = []

        def cb(sid: str) -> None:
            calls.append(sid)

        runtime.set_on_handoff_event(cb)
        assert runtime._on_handoff_event is cb

    def test_set_on_handoff_event_clears_callback(self, runtime: AgentRuntime) -> None:
        runtime.set_on_handoff_event(lambda sid: None)
        runtime.set_on_handoff_event(None)
        assert runtime._on_handoff_event is None

    def test_set_on_handoff_event_called_from_handoff(
        self, runtime: AgentRuntime
    ) -> None:
        calls: list[str] = []
        runtime.set_on_handoff_event(calls.append)

        target = _make_handoff_target(
            runtime, sid="target-handoff", name="target-handoff"
        )
        target_id = target.session_id

        tool = runtime.create_handoff_tool()
        tool_call = cast(
            ToolCall,
            {"id": "ho_1", "name": "handoff", "input": {}},
        )
        args = {
            "target_agent_name": "target-handoff",
            "context_summary": "ctx",
            "task_description": "task",
        }

        async def _run():
            events = []
            async for e in tool.execute(args, tool_call, stop_event=None):
                events.append(e)
            return events

        asyncio.run(_run())

        assert target_id in calls


class TestListHandoffTargets:
    def test_list_handoff_targets(self, runtime: AgentRuntime) -> None:
        t1 = _make_handoff_target(runtime, sid="a")
        t2 = _make_handoff_target(runtime, sid="b")

        targets = runtime.registered_agents
        assert len(targets) == 2
        assert t1 in targets
        assert t2 in targets

    def test_get_handoff_target(self, runtime: AgentRuntime) -> None:
        t1 = _make_handoff_target(runtime, sid="x")
        _make_handoff_target(runtime, sid="y")

        assert runtime.get_handoff_target("x") is t1
        assert runtime.get_handoff_target("y") is not t1
        assert runtime.get_handoff_target("missing") is None


class TestAgentRuntimeProtocol:
    def test_agent_runtime_conforms_to_protocol(self) -> None:
        from minimal_harness.agent.runtime import AgentRuntimeProtocol

        runtime = AgentRuntime(AgentRegistry())
        assert isinstance(runtime, AgentRuntimeProtocol)
