"""@ command handling for TUI — file/directory picker."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from textual.widgets import Label, ListItem, ListView

if TYPE_CHECKING:
    from textual.timer import Timer

    from minimal_harness.client.built_in.widgets import ChatInput


class AtCommandHandler:
    def __init__(
        self,
        suggestion_list: ListView,
        input_widget: ChatInput,
        get_input_text: Callable[[], str],
        set_input_text: Callable[[str], None],
    ) -> None:
        self._suggestion_list = suggestion_list
        self._input = input_widget
        self._get_input_text = get_input_text
        self._set_input_text = set_input_text
        self._cwd = Path.cwd()
        self._entries: list[Path] = []
        self._entries_cache: list[Path] | None = None
        self._debounce_timer: Timer | None = None

    def _get_entries(self) -> list[Path]:
        if self._entries_cache is not None:
            return self._entries_cache
        entries: list[Path] = []
        try:
            for p in self._cwd.rglob("*"):
                entries.append(p)
        except (PermissionError, OSError):
            pass
        self._entries_cache = sorted(entries)
        return self._entries_cache

    def _filter_entries(self, filter_text: str) -> list[Path]:
        entries = self._get_entries()
        if not filter_text:
            return entries
        lower = filter_text.lower()
        return [e for e in entries if lower in str(e.relative_to(self._cwd)).lower()]

    def _show_suggestions(self, filter_text: str) -> None:
        entries = self._filter_entries(filter_text)
        if not entries:
            self._hide_suggestions()
            return
        self._entries = entries[:10]
        self._suggestion_list.clear()
        for entry in self._entries:
            display = str(entry.relative_to(self._cwd))
            self._suggestion_list.append(ListItem(Label(display)))
        self._suggestion_list.add_class("visible")
        self._input.set_at_active(True)
        if self._suggestion_list.children:
            self._suggestion_list.index = 0

    def _cancel_debounce(self) -> None:
        if self._debounce_timer is not None:
            self._debounce_timer.stop()
            self._debounce_timer = None

    def _hide_suggestions(self) -> None:
        self._cancel_debounce()
        self._suggestion_list.remove_class("visible")
        self._suggestion_list.clear()
        self._entries = []
        self._input.set_at_active(False)

    def _insert_path(self, path: Path) -> None:
        text = self._get_input_text()
        at_pos = text.rfind("@")
        if at_pos != -1:
            abs_path = str(path.absolute())
            start = self._offset_to_location(at_pos)
            end = self._offset_to_location(len(text))
            self._input.replace(abs_path, start, end)

    def _offset_to_location(self, offset: int) -> tuple[int, int]:
        before = self._get_input_text()[:offset]
        lines = before.split("\n")
        return (len(lines) - 1, len(lines[-1]))

    def on_at_command_show(self, text: str) -> None:
        self._cancel_debounce()
        self._debounce_timer = self._input.set_timer(
            0.15, lambda t=text: self._do_show(t)
        )

    def _do_show(self, text: str) -> None:
        self._debounce_timer = None
        idx = text.rfind("@")
        if idx == -1:
            self._hide_suggestions()
            return
        filter_text = text[idx + 1 :]
        self._show_suggestions(filter_text)

    def on_at_command_hide(self) -> None:
        self._hide_suggestions()

    def on_at_command_navigate_up(self) -> None:
        sl = self._suggestion_list
        if sl.children:
            sl.action_cursor_up()

    def on_at_command_navigate_down(self) -> None:
        sl = self._suggestion_list
        if sl.children:
            sl.action_cursor_down()

    def on_at_command_select(self) -> None:
        sl = self._suggestion_list
        if not sl.children or sl.index is None:
            return
        idx = sl.index
        if 0 <= idx < len(self._entries):
            self._insert_path(self._entries[idx])
            self._hide_suggestions()

    def on_list_view_selected(self, idx: int | None) -> None:
        if not self._suggestion_list.has_class("visible"):
            return
        if idx is None:
            return
        if 0 <= idx < len(self._entries):
            self._insert_path(self._entries[idx])
            self._hide_suggestions()
