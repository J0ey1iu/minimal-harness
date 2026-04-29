from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from minimal_harness.agent.registry import AgentRegistry
from minimal_harness.agent.runtime import AgentRuntime
from minimal_harness.client.built_in.buffer import StreamBuffer
from minimal_harness.client.built_in.context import AppContext
from minimal_harness.client.built_in.session_controller import SessionController


@pytest.fixture
def app_context(tmp_path):
    ctx = AppContext(config={"provider": "openai", "model": "test"})
    ctx._all_tools = {}
    ctx.active_tools = []
    ctx.registry = MagicMock()
    ctx._create_llm_provider = MagicMock()
    return ctx


@pytest.fixture
def controller(app_context):
    runtime = MagicMock(spec=AgentRuntime)
    agent_registry = AgentRegistry()
    ctrl = SessionController(runtime, agent_registry, app_context)
    return ctrl


class TestSessionCreation:
    def test_create_session_assigns_new_memory(self, controller):
        session = controller.create_session(
            agent_name="test_agent",
            system_prompt="You are a test assistant.",
            default_tools=None,
        )
        assert session is not None
        assert session.name == "test_agent"
        assert session.session_id is not None
        assert session.memory is not None
        assert session.memory.agent_name == "test_agent"
        msgs = session.memory.get_all_messages()
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are a test assistant."

    def test_create_session_sets_current_session(self, controller):
        session = controller.create_session(agent_name="agent_a")
        assert controller.current_session_id == session.session_id
        assert controller.current_session is session

    def test_create_session_generates_unique_ids(self, controller):
        s1 = controller.create_session(agent_name="agent_a")
        s2 = controller.create_session(agent_name="agent_b")
        assert s1.session_id != s2.session_id

    def test_consecutive_creates_new_memory(self, controller):
        s1 = controller.create_session(agent_name="agent_a")
        m1 = s1.memory
        s2 = controller.create_session(agent_name="agent_b")
        m2 = s2.memory
        assert m1 is not m2
        assert m1.session_id != m2.session_id

    def test_created_at_is_set_on_creation(self, controller):
        session = controller.create_session(agent_name="agent_a")
        created_at = session.memory.created_at
        assert created_at is not None

    def test_consecutive_created_ats_differ(self, controller):
        s1 = controller.create_session(agent_name="agent_a")
        t1 = s1.memory.created_at
        s2 = controller.create_session(agent_name="agent_b")
        t2 = s2.memory.created_at
        assert t1 != t2


class TestPresetAgents:
    def test_register_preset_agents_registers_agents_in_registry(self, controller):
        with (
            patch(
                "minimal_harness.client.built_in.session_controller.load_agents_config"
            ) as mock_load,
            patch(
                "minimal_harness.client.built_in.session_controller.read_system_prompt"
            ) as mock_read,
        ):
            mock_load.return_value = [
                {
                    "name": "assistant_a",
                    "description": "Assistant A",
                    "system_prompt": "a.md",
                    "default_tools": [],
                },
                {
                    "name": "assistant_b",
                    "description": "Assistant B",
                    "system_prompt": "b.md",
                    "default_tools": [],
                },
            ]
            mock_read.return_value = "You are assistant."

            controller.register_preset_agents()

        assert controller._agent_registry.get("assistant_a") is not None
        assert controller._agent_registry.get("assistant_b") is not None
        assert len(controller._agent_registry.get_all()) == 2

    def test_preset_agents_registered_in_registry(self, controller):
        with (
            patch(
                "minimal_harness.client.built_in.session_controller.load_agents_config"
            ) as mock_load,
            patch(
                "minimal_harness.client.built_in.session_controller.read_system_prompt"
            ) as mock_read,
        ):
            mock_load.return_value = [
                {
                    "name": "agent_x",
                    "description": "X",
                    "system_prompt": "x.md",
                    "default_tools": [],
                },
            ]
            mock_read.return_value = "prompt"

            controller.register_preset_agents()

        metadata = controller._agent_registry.get("agent_x")
        assert metadata is not None
        assert metadata.name == "agent_x"


class TestHandoffLifecycle:
    def test_register_handoff_run_creates_new_session(self, controller):
        controller.create_session(agent_name="primary")
        task = MagicMock(spec=asyncio.Task)
        stop_event = asyncio.Event()
        queue: asyncio.Queue = asyncio.Queue()

        controller.register_handoff_run("handoff_target", task, stop_event, queue)

        found = False
        for sid, s in controller._sessions.items():
            if s.name == "handoff_target":
                found = True
                assert sid in controller._active_runs
                break
        assert found, "register_handoff_run should create a new session"

    def test_register_handoff_run_new_session_has_unique_created_at(self, controller):
        controller.create_session(agent_name="parent")
        parent_session = controller.current_session
        assert parent_session is not None
        parent_created = parent_session.memory.created_at

        task = MagicMock(spec=asyncio.Task)
        stop_event = asyncio.Event()
        queue: asyncio.Queue = asyncio.Queue()

        controller.register_handoff_run("child", task, stop_event, queue)

        child_session = None
        for s in controller._sessions.values():
            if s.name == "child":
                child_session = s
                break
        assert child_session is not None
        assert child_session.memory.created_at != parent_created

    def test_make_handoff_memory_creates_new_session_per_call(self, controller):
        mem1 = controller.make_handoff_memory("target")
        sid1 = controller._last_handoff_session_id
        mem2 = controller.make_handoff_memory("target")
        sid2 = controller._last_handoff_session_id

        assert sid1 != sid2
        assert mem1 is not mem2
        assert mem1.session_id != mem2.session_id

    def test_make_handoff_memory_sets_created_at_after_parent(self, controller):
        controller.create_session(agent_name="parent")
        parent_created = controller.current_session.memory.created_at

        controller.make_handoff_memory("child")
        child = controller._sessions[controller._last_handoff_session_id]
        assert child.memory.created_at > parent_created

    def test_register_handoff_run_uses_last_handoff_session(self, controller):
        controller.make_handoff_memory("target")
        expected_sid = controller._last_handoff_session_id

        task = MagicMock(spec=asyncio.Task)
        stop_event = asyncio.Event()
        queue: asyncio.Queue = asyncio.Queue()
        controller.register_handoff_run("target", task, stop_event, queue)

        assert expected_sid in controller._active_runs

    def test_register_handoff_run_tracks_active_run(self, controller):
        controller.create_session(agent_name="primary")
        task = MagicMock(spec=asyncio.Task)
        stop_event = asyncio.Event()
        queue: asyncio.Queue = asyncio.Queue()

        controller.register_handoff_run("handoff_target", task, stop_event, queue)

        assert len(controller._active_runs) == 1
        for sid, (t, s, q) in controller._active_runs.items():
            assert t is task
            assert s is stop_event
            assert q is queue

    def test_handoff_target_ids_excludes_foreground(self, controller):
        controller.create_session(agent_name="primary")
        primary_id = controller.current_session_id

        task = MagicMock(spec=asyncio.Task)
        stop_event = asyncio.Event()
        queue: asyncio.Queue = asyncio.Queue()

        controller.register_handoff_run("handoff_target", task, stop_event, queue)
        controller._foreground_session_id = primary_id

        assert primary_id not in controller.handoff_target_ids
        assert len(controller.handoff_target_ids) == 1

    def test_start_run_sets_foreground(self, controller):
        session = controller.create_session(agent_name="primary")
        controller._runtime.run.return_value = (
            MagicMock(spec=asyncio.Task),
            asyncio.Event(),
            asyncio.Queue(),
        )

        stop_event, queue = controller.start_run(session, "hello")
        assert controller._foreground_session_id == session.session_id

    def test_end_run_clears_foreground(self, controller):
        session = controller.create_session(agent_name="primary")
        controller._runtime.run.return_value = (
            MagicMock(spec=asyncio.Task),
            asyncio.Event(),
            asyncio.Queue(),
        )

        controller.start_run(session, "hello")
        controller.end_run(session.session_id)
        assert controller._foreground_session_id is None

    def test_drain_session_events_skips_foreground(self, controller):
        session = controller.create_session(agent_name="primary")
        controller._foreground_session_id = session.session_id
        events, done = controller.drain_session_events(session.session_id)
        assert events == []
        assert done is False

    def test_drain_session_events_gets_events(self, controller):
        session = controller.create_session(agent_name="primary")
        controller._foreground_session_id = "other"

        q: asyncio.Queue = asyncio.Queue()
        q.put_nowait({"type": "chunk", "data": "hello"})
        controller._active_runs[session.session_id] = (
            MagicMock(spec=asyncio.Task),
            asyncio.Event(),
            q,
        )

        events, done = controller.drain_session_events(session.session_id)
        assert len(events) == 1
        assert events[0]["data"] == "hello"

    def test_drain_none_sentinel_marks_done(self, controller):
        session = controller.create_session(agent_name="primary")
        controller._foreground_session_id = "other"

        q: asyncio.Queue = asyncio.Queue()
        q.put_nowait(None)
        controller._active_runs[session.session_id] = (
            MagicMock(spec=asyncio.Task),
            asyncio.Event(),
            q,
        )

        events, done = controller.drain_session_events(session.session_id)
        assert done is True
        assert session.session_id not in controller._active_runs


class TestSessionManagement:
    def test_switch_session_changes_current(self, controller):
        s1 = controller.create_session(agent_name="agent_a")
        controller.create_session(agent_name="agent_b")
        controller.switch_session(s1.session_id)
        assert controller.current_session_id == s1.session_id

    def test_interrupt_calls_session_interrupt(self, controller):
        session = controller.create_session(agent_name="agent_a")
        session.interrupt = MagicMock()
        controller.interrupt()
        session.interrupt.assert_called_once()

    def test_interrupt_sets_stop_event_for_active_run(self, controller):
        controller.create_session(agent_name="agent_a")
        sid = controller.current_session_id
        stop_event = asyncio.Event()
        controller._active_runs[sid] = (
            MagicMock(spec=asyncio.Task),
            stop_event,
            asyncio.Queue(),
        )

        controller.interrupt()
        assert stop_event.is_set()

    def test_set_streaming_flag(self, controller):
        assert controller.streaming is False
        controller.set_streaming(True)
        assert controller.streaming is True
        controller.set_streaming(False)
        assert controller.streaming is False

    def test_buf_is_stream_buffer(self, controller):
        assert isinstance(controller.buf, StreamBuffer)

    def test_memory_property_returns_current_session_memory(self, controller):
        controller.create_session(agent_name="agent_a")
        memory = controller.memory
        assert memory is controller.current_session.memory

    def test_memory_property_none_when_no_session(self, controller):
        assert controller.memory is None

    def test_active_tools_returns_session_tools(self, controller):
        controller.create_session(agent_name="agent_a")
        session = controller.current_session
        assert session is not None
        assert controller.active_tools is session.tools


class TestGetAllSessionsMetadata:
    def test_combines_memory_and_disk_sessions(self, controller, tmp_path):
        session = controller.create_session(agent_name="mem_agent")
        sid = session.session_id

        metadata = controller.get_all_sessions_metadata()
        meta_ids = {m["session_id"] for m in metadata}

        assert sid in meta_ids

    def test_metadata_sorted_by_created_at_desc(self, controller, tmp_path):
        with (
            patch(
                "minimal_harness.client.built_in.session_controller.PersistentMemory"
            ) as mock_pm,
        ):
            mock_pm.list_sessions.return_value = [
                {
                    "session_id": "old",
                    "title": "Old",
                    "created_at": "2023-01-01",
                    "message_count": 1,
                    "agent_name": "a",
                },
                {
                    "session_id": "new",
                    "title": "New",
                    "created_at": "2024-01-01",
                    "message_count": 1,
                    "agent_name": "b",
                },
            ]

            metadata = controller.get_all_sessions_metadata()
            timestamps = [m["created_at"] for m in metadata]
            assert timestamps == sorted(timestamps, reverse=True)
