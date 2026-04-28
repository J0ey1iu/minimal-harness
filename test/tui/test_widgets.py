from __future__ import annotations

from unittest.mock import MagicMock

from textual.events import Key

from minimal_harness.client.built_in.widgets import (
    ChatInput,
    ChatInputDump,
    ChatInputSubmit,
    SlashCommandHide,
    SlashCommandNavigateDown,
    SlashCommandNavigateUp,
    SlashCommandSelect,
    SlashCommandShow,
)


def _make_chat_input(text: str = "") -> tuple[ChatInput, MagicMock]:
    ci = ChatInput(text)
    mock_post = MagicMock()
    ci.post_message = mock_post
    return ci, mock_post


class TestChatInputInit:
    def test_default_state(self):
        ci, _ = _make_chat_input()
        assert ci._slash_active is False
        assert ci._input_history == []
        assert ci._history_index == -1
        assert ci._current_input == ""


class TestChatInputSlashDetection:
    def test_slash_show_on_text_starting_with_slash(self):
        ci, mock_post = _make_chat_input()
        ci.text = "/help"
        mock_post.reset_mock()
        ci.on_text_area_changed(MagicMock())
        assert mock_post.call_count >= 1
        msg = mock_post.call_args[0][0]
        assert isinstance(msg, SlashCommandShow)
        assert msg.prefix == "/help"

    def test_slash_hide_when_slash_was_active_and_text_no_longer_starts_with_slash(
        self,
    ):
        ci, mock_post = _make_chat_input()
        ci.text = "no slash"
        ci._slash_active = True
        mock_post.reset_mock()
        ci.on_text_area_changed(MagicMock())
        assert mock_post.call_count >= 1
        msg = mock_post.call_args[0][0]
        assert isinstance(msg, SlashCommandHide)

    def test_no_message_when_text_normal_and_slash_not_active(self):
        ci, mock_post = _make_chat_input()
        mock_post.reset_mock()
        ci.on_text_area_changed(MagicMock())
        mock_post.assert_not_called()


class TestChatInputHistory:
    def test_up_arrow_no_history_does_nothing(self):
        ci, mock_post = _make_chat_input()
        ci.on_key(Key(key="up", character=""))
        mock_post.assert_not_called()

    def test_up_arrow_with_history_navigates_back(self):
        ci, _ = _make_chat_input()
        ci._input_history = ["first", "second"]
        ci.on_key(Key(key="up", character=""))
        assert ci._history_index == 0
        assert ci.text == "second"

    def test_up_arrow_more_history(self):
        ci, _ = _make_chat_input()
        ci._input_history = ["first", "second", "third"]
        ci.on_key(Key(key="up", character=""))
        assert ci.text == "third"
        ci.on_key(Key(key="up", character=""))
        assert ci.text == "second"
        ci.on_key(Key(key="up", character=""))
        assert ci.text == "first"
        ci.on_key(Key(key="up", character=""))
        assert ci.text == "first"

    def test_down_arrow_with_history_at_start(self):
        ci, _ = _make_chat_input()
        ci._input_history = ["first", "second"]
        ci._current_input = "current"
        ci._history_index = 1
        ci.on_key(Key(key="down", character=""))
        assert ci._history_index == 0
        assert ci.text == "second"

    def test_down_arrow_to_bottom_restores_current(self):
        ci, _ = _make_chat_input()
        ci._input_history = ["first", "second"]
        ci._current_input = "typing..."
        ci._history_index = 0
        ci.on_key(Key(key="down", character=""))
        assert ci._history_index == -1
        assert ci.text == "typing..."

    def test_down_arrow_no_history_does_nothing(self):
        ci, mock_post = _make_chat_input()
        ci.on_key(Key(key="down", character=""))
        mock_post.assert_not_called()


class TestChatInputSubmit:
    def test_enter_submits(self):
        ci, mock_post = _make_chat_input()
        ci.text = "hello"
        mock_post.reset_mock()
        ci.on_key(Key(key="enter", character="\n"))
        assert ci._input_history == ["hello"]
        assert mock_post.call_count >= 1
        msg = mock_post.call_args[0][0]
        assert isinstance(msg, ChatInputSubmit)

    def test_enter_empty_text_does_not_add_to_history(self):
        ci, mock_post = _make_chat_input()
        ci.text = "  "
        mock_post.reset_mock()
        ci.on_key(Key(key="enter", character="\n"))
        assert ci._input_history == []
        assert mock_post.call_count >= 1
        msg = mock_post.call_args[0][0]
        assert isinstance(msg, ChatInputSubmit)

    def test_ctrl_enter_inserts_newline(self):
        ci, _ = _make_chat_input()
        ci.insert = MagicMock()
        ci.on_key(Key(key="ctrl+enter", character=""))
        ci.insert.assert_called_once_with("\n")  # type: ignore[union-attr]

    def test_ctrl_j_inserts_newline(self):
        ci, _ = _make_chat_input()
        ci.insert = MagicMock()
        ci.on_key(Key(key="ctrl+j", character=""))
        ci.insert.assert_called_once_with("\n")  # type: ignore[union-attr]


class TestChatInputSlashKeyHandling:
    def test_slash_active_up_navigates(self):
        ci, mock_post = _make_chat_input()
        ci._slash_active = True
        ci.on_key(Key(key="up", character=""))
        assert mock_post.call_count >= 1
        msg = mock_post.call_args[0][0]
        assert isinstance(msg, SlashCommandNavigateUp)

    def test_slash_active_down_navigates(self):
        ci, mock_post = _make_chat_input()
        ci._slash_active = True
        ci.on_key(Key(key="down", character=""))
        assert mock_post.call_count >= 1
        msg = mock_post.call_args[0][0]
        assert isinstance(msg, SlashCommandNavigateDown)

    def test_slash_active_enter_selects(self):
        ci, mock_post = _make_chat_input()
        ci._slash_active = True
        ci.on_key(Key(key="enter", character="\n"))
        assert mock_post.call_count >= 1
        msg = mock_post.call_args[0][0]
        assert isinstance(msg, SlashCommandSelect)

    def test_slash_active_escape_hides(self):
        ci, mock_post = _make_chat_input()
        ci._slash_active = True
        ci.on_key(Key(key="escape", character=""))
        assert mock_post.call_count >= 1
        msg = mock_post.call_args[0][0]
        assert isinstance(msg, SlashCommandHide)

    def test_normal_up_arrow_when_slash_active_does_not_trigger_history(self):
        ci, mock_post = _make_chat_input()
        ci._slash_active = True
        ci._input_history = ["test"]
        ci.on_key(Key(key="up", character=""))
        assert mock_post.call_count >= 1
        msg = mock_post.call_args[0][0]
        assert isinstance(msg, SlashCommandNavigateUp)
        assert ci._history_index == -1


class TestChatInputDump:
    def test_action_dump(self):
        ci, mock_post = _make_chat_input()
        ci.action_dump()
        assert mock_post.call_count >= 1
        msg = mock_post.call_args[0][0]
        assert isinstance(msg, ChatInputDump)

    def test_set_slash_active(self):
        ci, _ = _make_chat_input()
        ci.set_slash_active(True)
        assert ci._slash_active is True
        ci.set_slash_active(False)
        assert ci._slash_active is False

    def test_reset_history_index(self):
        ci, _ = _make_chat_input()
        ci._history_index = 2
        ci._current_input = "text"
        ci.reset_history_index()
        assert ci._history_index == -1
        assert ci._current_input == ""
