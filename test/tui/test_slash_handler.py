from __future__ import annotations

from unittest.mock import MagicMock

from minimal_harness.client.built_in.slash_handler import SlashCommandHandler


def _make_handler() -> tuple[
    SlashCommandHandler, MagicMock, MagicMock, MagicMock, MagicMock
]:
    suggestion_list = MagicMock()
    suggestion_list.children = []
    suggestion_list.index = None
    input_widget = MagicMock()
    get_input_text = MagicMock(return_value="/con")
    set_input_text = MagicMock()
    execute_action = MagicMock()
    handler = SlashCommandHandler(
        suggestion_list, input_widget, get_input_text, set_input_text, execute_action
    )
    return handler, suggestion_list, input_widget, get_input_text, execute_action


class TestSlashCommandHandler:
    def test_slash_commands_defined(self):
        assert len(SlashCommandHandler.SLASH_COMMANDS) == 5

    def test_filter_suggestions_matches(self):
        handler, *_ = _make_handler()
        result = handler._filter_suggestions("/c")
        commands = [c for c, _, _ in result]
        assert "/config" in commands

    def test_filter_suggestions_no_match(self):
        handler, *_ = _make_handler()
        result = handler._filter_suggestions("/z")
        assert result == []

    def test_show_suggestions_with_matches(self):
        handler, sl, _, _, _ = _make_handler()
        handler._show_suggestions("/c")
        sl.clear.assert_called_once()
        sl.append.assert_called()
        sl.add_class.assert_called_once_with("visible")

    def test_show_suggestions_no_matches_hides(self):
        handler, sl, _, _, _ = _make_handler()
        handler._show_suggestions("/z")
        sl.remove_class.assert_called_once_with("visible")

    def test_hide_suggestions(self):
        handler, sl, input_widget, _, _ = _make_handler()
        handler._hide_suggestions()
        sl.remove_class.assert_called_once_with("visible")
        sl.clear.assert_called_once()
        input_widget.set_slash_active.assert_called_once_with(False)

    def test_on_slash_command_show(self):
        handler, sl, _, _, _ = _make_handler()
        handler.on_slash_command_show("/t")
        sl.clear.assert_called_once()

    def test_on_slash_command_hide(self):
        handler, sl, input_widget, _, _ = _make_handler()
        handler.on_slash_command_hide()
        sl.remove_class.assert_called_once_with("visible")

    def test_on_slash_command_navigate_up(self):
        handler, sl, _, _, _ = _make_handler()
        sl.children = [MagicMock()]
        handler.on_slash_command_navigate_up()
        sl.action_cursor_up.assert_called_once()

    def test_on_slash_command_navigate_up_no_children(self):
        handler, sl, _, _, _ = _make_handler()
        handler.on_slash_command_navigate_up()
        sl.action_cursor_up.assert_not_called()

    def test_on_slash_command_navigate_down(self):
        handler, sl, _, _, _ = _make_handler()
        sl.children = [MagicMock()]
        handler.on_slash_command_navigate_down()
        sl.action_cursor_down.assert_called_once()

    def test_on_slash_command_select_valid(self):
        handler, sl, input_widget, get_input_text, execute_action = _make_handler()
        sl.children = [MagicMock()]
        sl.index = 0
        get_input_text.return_value = "/config"
        handler._filter_suggestions = MagicMock(
            return_value=[("/config", "Open configuration", "config")]
        )
        handler.on_slash_command_select()
        input_widget.set_slash_active.assert_called_once_with(False)
        execute_action.assert_called_once_with("config")

    def test_on_slash_command_select_no_children(self):
        handler, sl, _, _, execute_action = _make_handler()
        sl.children = []
        handler.on_slash_command_select()
        execute_action.assert_not_called()

    def test_on_list_view_selected_valid(self):
        handler, sl, _, get_input_text, execute_action = _make_handler()
        sl.has_class.return_value = True
        sl.index = 0
        get_input_text.return_value = "/tools"
        handler._filter_suggestions = MagicMock(
            return_value=[("/tools", "Select tools", "tools")]
        )
        handler.on_list_view_selected(0)
        execute_action.assert_called_once_with("tools")

    def test_on_list_view_selected_not_visible(self):
        handler, sl, _, _, execute_action = _make_handler()
        sl.has_class.return_value = False
        handler.on_list_view_selected(0)
        execute_action.assert_not_called()

    def test_on_list_view_selected_none_index(self):
        handler, sl, _, _, execute_action = _make_handler()
        sl.has_class.return_value = True
        handler.on_list_view_selected(None)
        execute_action.assert_not_called()
