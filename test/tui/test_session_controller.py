from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from minimal_harness.agent.registry import AgentRegistry
from minimal_harness.agent.runtime import AgentRuntime
from minimal_harness.client.built_in.buffer import StreamBuffer
from minimal_harness.client.built_in.context import AppContext
from minimal_harness.client.built_in.session_controller import SessionController


@pytest.fixture
def app_context():
    ctx = AppContext(config={"provider": "openai", "model": "test"})
    ctx.create_llm_provider = MagicMock(return_value=MagicMock())
    return ctx


@pytest.fixture
def controller(app_context):
    runtime = AsyncMock(spec=AgentRuntime)
    agent_registry = AgentRegistry()
    ctrl = SessionController(runtime, agent_registry, app_context)
    return ctrl


class TestSessionCreation:
    @pytest.mark.asyncio
    async def test_create_session_returns_valid_session(self, controller):
        session = await controller.create_session(
            agent_name="test_agent",
        )
        assert session is not None
        assert session.name == "test_agent"
        assert session.session_id is not None
        assert session.memory_id is not None

        mem = await controller._ctx.memory_store.get_memory(session.memory_id)
        assert mem is not None

    @pytest.mark.asyncio
    async def test_create_session_sets_current_session(self, controller):
        session = await controller.create_session(agent_name="agent_a")
        assert controller.current_session_id == session.session_id
        assert controller.current_session is session

    @pytest.mark.asyncio
    async def test_create_session_generates_unique_ids(self, controller):
        s1 = await controller.create_session(agent_name="agent_a")
        s2 = await controller.create_session(agent_name="agent_b")
        assert s1.session_id != s2.session_id

    @pytest.mark.asyncio
    async def test_consecutive_creates_new_memory(self, controller):
        s1 = await controller.create_session(agent_name="agent_a")
        m1 = await controller._ctx.memory_store.get_memory(s1.memory_id)
        s2 = await controller.create_session(agent_name="agent_b")
        m2 = await controller._ctx.memory_store.get_memory(s2.memory_id)
        assert m1 is not m2
        assert s1.memory_id != s2.memory_id

    @pytest.mark.asyncio
    async def test_created_at_is_set_on_creation(self, controller):
        session = await controller.create_session(agent_name="agent_a")
        mem = await controller._ctx.memory_store.get_memory(session.memory_id)
        assert mem.created_at is not None


class TestPresetAgents:
    @pytest.mark.asyncio
    async def test_register_preset_agents_registers_agents_in_registry(
        self, controller
    ):
        with (
            patch(
                "minimal_harness.client.built_in.agent_manager.load_agents_config"
            ) as mock_load,
            patch(
                "minimal_harness.client.built_in.agent_manager.read_system_prompt"
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

            await controller.register_preset_agents()

        assert await controller._agent_registry.get("assistant_a") is not None
        assert await controller._agent_registry.get("assistant_b") is not None
        assert len(await controller._agent_registry.get_all()) == 2

    @pytest.mark.asyncio
    async def test_preset_agents_registered_in_registry(self, controller):
        with (
            patch(
                "minimal_harness.client.built_in.agent_manager.load_agents_config"
            ) as mock_load,
            patch(
                "minimal_harness.client.built_in.agent_manager.read_system_prompt"
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

            await controller.register_preset_agents()

        metadata = await controller._agent_registry.get("agent_x")
        assert metadata is not None
        assert metadata.name == "agent_x"


class TestSessionManagement:
    @pytest.mark.asyncio
    async def test_switch_session_changes_current(self, controller):
        s1 = await controller.create_session(agent_name="agent_a")
        await controller.create_session(agent_name="agent_b")
        controller.switch_session(s1.session_id)
        assert controller.current_session_id == s1.session_id

    @pytest.mark.asyncio
    async def test_interrupt_calls_session_interrupt(self, controller):
        session = await controller.create_session(agent_name="agent_a")
        session.interrupt = MagicMock()
        controller.interrupt()
        session.interrupt.assert_called_once()

    @pytest.mark.asyncio
    async def test_interrupt_sets_stop_event_for_active_run(self, controller):
        await controller.create_session(agent_name="agent_a")
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
        buf = controller.get_buf("test")
        assert isinstance(buf, StreamBuffer)

    @pytest.mark.asyncio
    async def test_get_memory_returns_current_session_memory(self, controller):
        await controller.create_session(agent_name="agent_a")
        memory = await controller.get_memory()
        assert memory is not None
        assert memory.memory_id == controller.current_session.memory_id

    @pytest.mark.asyncio
    async def test_get_memory_none_when_no_session(self, controller):
        assert await controller.get_memory() is None


class TestRunManagement:
    @pytest.mark.asyncio
    async def test_start_run_calls_runtime_with_ids(self, controller):
        session = await controller.create_session(agent_name="primary")
        stop_event = asyncio.Event()
        controller._runtime.run = AsyncMock(
            return_value=(MagicMock(spec=asyncio.Task), stop_event, asyncio.Queue())
        )

        result = await controller.start_run(session, "hello world")
        assert result is not None

        controller._runtime.run.assert_awaited_once()
        call_args = controller._runtime.run.await_args
        assert call_args.kwargs["user_input"] == [
            {"type": "text", "text": "hello world"}
        ]
        assert call_args.kwargs["agent_metadata_id"] == session.agent_metadata_id
        assert call_args.kwargs["memory_id"] == session.memory_id

    @pytest.mark.asyncio
    async def test_end_run_removes_from_active(self, controller):
        session = await controller.create_session(agent_name="primary")
        controller._runtime.run = AsyncMock(
            return_value=(
                MagicMock(spec=asyncio.Task),
                asyncio.Event(),
                asyncio.Queue(),
            )
        )

        await controller.start_run(session, "hello")
        await controller.end_run(session.session_id)
        assert session.session_id not in controller._active_runs

    @pytest.mark.asyncio
    async def test_drain_session_events_gets_events(self, controller):
        session = await controller.create_session(agent_name="primary")

        q: asyncio.Queue = asyncio.Queue()
        q.put_nowait({"type": "chunk", "data": "hello"})
        controller._active_runs[session.session_id] = (
            MagicMock(spec=asyncio.Task),
            asyncio.Event(),
            q,
        )

        events, done = await controller.drain_session_events(session.session_id)
        assert len(events) == 1
        assert events[0]["data"] == "hello"

    @pytest.mark.asyncio
    async def test_drain_none_sentinel_marks_done(self, controller):
        session = await controller.create_session(agent_name="primary")

        q: asyncio.Queue = asyncio.Queue()
        q.put_nowait(None)
        controller._active_runs[session.session_id] = (
            MagicMock(spec=asyncio.Task),
            asyncio.Event(),
            q,
        )

        events, done = await controller.drain_session_events(session.session_id)
        assert done is True
        assert session.session_id not in controller._active_runs


class TestGetAllSessionsMetadata:
    @pytest.mark.asyncio
    async def test_sessions_metadata_includes_memory_session(self, controller):
        session = await controller.create_session(agent_name="mem_agent")
        sid = session.session_id

        # Add a message so the session is not filtered out
        mem = await controller.get_memory(session.session_id)
        assert mem is not None
        mem.add_message(
            {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        )

        metadata = await controller.get_all_sessions_metadata()
        meta_ids = {m["session_id"] for m in metadata}

        assert sid in meta_ids
