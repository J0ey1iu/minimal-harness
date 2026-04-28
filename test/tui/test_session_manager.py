from __future__ import annotations

from unittest.mock import MagicMock, patch

from minimal_harness.client.built_in.context import AppContext
from minimal_harness.client.built_in.session_manager import SessionManager


def _make_manager() -> tuple[SessionManager, MagicMock, MagicMock]:
    ctx = MagicMock(spec=AppContext)
    display = MagicMock()
    clear_input = MagicMock()
    show_banner = MagicMock()
    manager = SessionManager(ctx, display, clear_input, show_banner)
    return manager, ctx, display


class TestExtractUserInputs:
    def test_extracts_user_text(self):
        manager, _, _ = _make_manager()
        memory = MagicMock()
        memory.get_all_messages.return_value = [
            {"role": "system", "content": "prompt"},
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello there"}],
            },
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
    def test_skips_system_role(self):
        manager, _, display = _make_manager()
        memory = MagicMock()
        memory.get_all_messages.return_value = [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ]
        manager._replay_memory(memory)
        display.say.assert_called_once_with("hi", user=True)

    def test_replays_user_message(self):
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
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"loc": "NYC"}',
                        }
                    }
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

    def test_replays_tool_error_result(self):
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


class TestLoadSession:
    def test_load_session_success(self):
        from minimal_harness.client.built_in.memory import PersistentMemory

        clear_committed = MagicMock()
        clear_buf = MagicMock()
        manager, ctx, display = _make_manager()
        display.say.return_value = None

        mock_memory = MagicMock(spec=PersistentMemory)
        mock_memory.title = "Test Session"
        mock_memory.get_all_messages.return_value = []

        with patch(
            "minimal_harness.client.built_in.session_manager.PersistentMemory"
        ) as mock_cls:
            mock_cls.from_session.return_value = mock_memory
            ok, inputs = manager.load_session("test-id", clear_committed, clear_buf)
            assert ok is True

    def test_load_session_failure(self):
        clear_committed = MagicMock()
        clear_buf = MagicMock()
        manager, _, display = _make_manager()

        ok, inputs = manager.load_session("nonexistent", clear_committed, clear_buf)
        assert ok is False
        assert inputs == []
        display.say.assert_called_once()
