"""Slash command handling for TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from textual.widgets import Label, ListItem, ListView

if TYPE_CHECKING:
    from minimal_harness.client.built_in.widgets import ChatInput


class SlashCommandHandler:
    SLASH_COMMANDS: list[tuple[str, str, str]] = [
        ("/config", "Open configuration", "config"),
        ("/tools", "Select tools", "tools"),
        ("/new", "Start new conversation", "new"),
        ("/sessions", "Resume a past session", "sessions"),
        ("/share", "Export chat as SVG", "share"),
    ]

    def __init__(
        self,
        suggestion_list: ListView,
        input_widget: ChatInput,
        get_input_text: Callable[[], str],
        set_input_text: Callable[[str], None],
        execute_action: Callable[[str], None],
    ) -> None:
        self._suggestion_list = suggestion_list
        self._input = input_widget
        self._get_input_text = get_input_text
        self._set_input_text = set_input_text
        self._execute_action = execute_action

    def _filter_suggestions(self, prefix: str) -> list[tuple[str, str, str]]:
        return [
            (cmd, desc, action)
            for cmd, desc, action in self.SLASH_COMMANDS
            if cmd.startswith(prefix)
        ]

    def _show_suggestions(self, prefix: str) -> None:
        suggestions = self._filter_suggestions(prefix)
        if not suggestions:
            self._hide_suggestions()
            return
        self._suggestion_list.clear()
        for cmd, desc, _ in suggestions:
            self._suggestion_list.append(ListItem(Label(f"{cmd}  {desc}")))
        self._suggestion_list.add_class("visible")
        self._input.set_slash_active(True)
        if self._suggestion_list.children:
            self._suggestion_list.index = 0

    def _hide_suggestions(self) -> None:
        self._suggestion_list.remove_class("visible")
        self._suggestion_list.clear()
        self._input.set_slash_active(False)

    def on_slash_command_show(self, prefix: str) -> None:
        self._show_suggestions(prefix)

    def on_slash_command_hide(self) -> None:
        self._hide_suggestions()

    def on_slash_command_navigate_up(self) -> None:
        sl = self._suggestion_list
        if sl.children:
            sl.action_cursor_up()

    def on_slash_command_navigate_down(self) -> None:
        sl = self._suggestion_list
        if sl.children:
            sl.action_cursor_down()

    def on_slash_command_select(self) -> None:
        sl = self._suggestion_list
        if not sl.children or sl.index is None:
            return
        idx = sl.index
        suggestions = self._filter_suggestions(self._get_input_text())
        if 0 <= idx < len(suggestions):
            _, _, action = suggestions[idx]
            self._set_input_text("")
            self._hide_suggestions()
            self._execute_action(action)

    def on_list_view_selected(self, idx: int | None) -> None:
        if not self._suggestion_list.has_class("visible"):
            return
        if idx is None:
            return
        suggestions = self._filter_suggestions(self._get_input_text())
        if 0 <= idx < len(suggestions):
            _, _, action = suggestions[idx]
            self._set_input_text("")
            self._hide_suggestions()
            self._execute_action(action)