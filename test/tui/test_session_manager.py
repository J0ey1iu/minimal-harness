from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from minimal_harness.client.built_in.context import AppContext
from minimal_harness.client.built_in.session_replayer import SessionReplayer


def _make_manager() -> tuple[SessionReplayer, MagicMock, MagicMock]:
    ctx = MagicMock(spec=AppContext)
    ctx.session_store = AsyncMock()
    ctx.session_store.create_session = AsyncMock(
        return_value=MagicMock(memory_id="mem1")
    )
    ctx.session_store.get_session = AsyncMock(return_value=MagicMock(memory_id="mem1"))
    display = MagicMock()
    clear_input = MagicMock()

    async def _show_banner() -> None:
        pass

    manager = SessionReplayer(ctx, display, clear_input, _show_banner)
    return manager, ctx, display


class TestExtractUserInputs:
    def test_extracts_user_text(self):
        manager, _, _ = _make_manager()
        memory = MagicMock()
        memory.get_all_messages.return_value = [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": [{"type": "text", "text": "hello there"}]},
            {"role": "user", "content": [{"type": "text", "text": "second msg"}]},
        ]
        result = manager._extract_user_inputs(memory)
        assert result == ["hello there", "second msg"]

    def test_skips_non_text_parts(self):
        manager, _, _ = _make_manager()
        memory = MagicMock()
        memory.get_all_messages.return_value = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image", "source": {"type": "base64", "data": "abc"}},
                ],
            }
        ]
        result = manager._extract_user_inputs(memory)
        assert result == ["hello"]

    def test_skips_empty_text(self):
        manager, _, _ = _make_manager()
        memory = MagicMock()
        memory.get_all_messages.return_value = [
            {"role": "user", "content": [{"type": "text", "text": ""}]}
        ]
        result = manager._extract_user_inputs(memory)
        assert result == []

    def test_returns_empty_when_no_user_messages(self):
        manager, _, _ = _make_manager()
        memory = MagicMock()
        memory.get_all_messages.return_value = [
            {"role": "system", "content": "prompt"},
            {"role": "assistant", "content": "hello"},
        ]
        result = manager._extract_user_inputs(memory)
        assert result == []


class TestReplayMemory:
    def test_skips_system(self):
        manager, _, display = _make_manager()
        memory = MagicMock()
        memory.get_all_messages.return_value = [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ]
        manager._replay_memory(memory)
        display.say.assert_called_once_with("hi", user=True)

    def test_replays_user(self):
        manager, _, display = _make_manager()
        memory = MagicMock()
        memory.get_all_messages.return_value = [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        ]
        manager._replay_memory(memory)
        display.say.assert_called_once_with("hello", user=True)

    def test_replays_assistant_text(self):
        manager, _, display = _make_manager()
        memory = MagicMock()
        memory.get_all_messages.return_value = [
            {"role": "assistant", "content": "I am an AI."}
        ]
        manager._replay_memory(memory)
        display.say.assert_any_call("I am an AI.", "", True)

    def test_replays_assistant_tool_calls(self):
        manager, _, display = _make_manager()
        memory = MagicMock()
        memory.get_all_messages.return_value = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": '{"loc": "NYC"}'}}
                ],
            }
        ]
        manager._replay_memory(memory)
        display.say_tool_call.assert_called_once()

    def test_replays_reasoning(self):
        manager, _, display = _make_manager()
        memory = MagicMock()
        memory.get_all_messages.return_value = [
            {"role": "reasoning", "content": "thinking step..."}
        ]
        manager._replay_memory(memory)
        display.say_reasoning.assert_called_once_with("thinking step...")

    def test_replays_tool_error(self):
        manager, _, display = _make_manager()
        memory = MagicMock()
        memory.get_all_messages.return_value = [
            {"role": "tool", "content": "[Tool Error] Something broke"}
        ]
        manager._replay_memory(memory)
        display.say_tool_result.assert_called_once()

    def test_replays_tool_success_result(self):
        manager, _, display = _make_manager()
        memory = MagicMock()
        memory.get_all_messages.return_value = [
            {"role": "tool", "content": '{"result": "ok"}'}
        ]
        manager._replay_memory(memory)
        display.say_tool_result.assert_called_once()


class TestReplaySession:
    @pytest.mark.asyncio
    async def test_replay_session_success(self):
        manager, ctx, display = _make_manager()
        display.say.return_value = None
        memory = await ctx.session_store.create_session(agent_name="test")
        mock_session = MagicMock()
        mock_session.session = MagicMock()
        mock_session.session.title = "Test Session"
        mock_session.session.memory_id = memory.memory_id
        clear_committed = MagicMock()
        clear_buf = MagicMock()

        ok, inputs = await manager.replay_session(
            mock_session, clear_committed, clear_buf
        )
        assert ok is True

    @pytest.mark.asyncio
    async def test_replay_session_failure(self):
        manager, ctx, display = _make_manager()
        # Make get_memory raise an exception to simulate failure

        original_get_memory = ctx.session_store.get_session

        async def failing_get_memory(memory_id):
            raise Exception("Test error")

        ctx.session_store.get_session = failing_get_memory
        mock_session = MagicMock()
        mock_session.session = MagicMock()
        mock_session.session.memory_id = "nonexistent"
        clear_committed = MagicMock()
        clear_buf = MagicMock()

        ok, inputs = await manager.replay_session(
            mock_session, clear_committed, clear_buf
        )
        assert ok is False
        assert inputs == []
        ctx.session_store.get_session = original_get_memory
