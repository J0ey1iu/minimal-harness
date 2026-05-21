"""Custom widgets for the TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.timer import Timer
from textual.widgets import Static, TextArea

if TYPE_CHECKING:
    pass

from .messages import (
    AtCommandHide,
    AtCommandNavigateDown,
    AtCommandNavigateUp,
    AtCommandSelect,
    AtCommandShow,
    ChatInputDump,
    ChatInputSubmit,
    SessionNotificationClicked,
    SlashCommandHide,
    SlashCommandNavigateDown,
    SlashCommandNavigateUp,
    SlashCommandSelect,
    SlashCommandShow,
)


class Banner(Static):
    pass


class SessionNotification(Static):
    """Notification that a background session has finished. Click to jump to it."""

    def __init__(self, session_id: str, session_name: str, **kwargs) -> None:
        self._session_id = session_id
        self._timer: Timer | None = None
        text = Text.assemble(
            ("\u2713 ", "bold bright_green"),
            (f'Session "{session_name}" finished', "bold"),
            ("  (click to switch)", "dim"),
        )
        super().__init__(text, **kwargs)

    @property
    def target_session_id(self) -> str:
        return self._session_id

    def on_click(self) -> None:
        self.post_message(SessionNotificationClicked(self._session_id))


class ChatInput(TextArea):
    BINDINGS = [Binding("ctrl+d", "dump", "Dump", show=True)]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._slash_active: bool = False
        self._at_active: bool = False
        self._input_history: list[str] = []
        self._history_index: int = -1
        self._current_input: str = ""
        self._change_timer: Timer | None = None

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if self.is_mounted:
            if self._change_timer is not None:
                self._change_timer.reset()
            else:
                self._change_timer = self.set_timer(0.08, self._handle_text_changed)
        else:
            self._handle_text_changed()

    def _handle_text_changed(self) -> None:
        self._change_timer = None
        text = self.text

        if text.startswith("/"):
            if self._at_active:
                self._at_active = False
                self.post_message(AtCommandHide())
            self.post_message(SlashCommandShow(text))
            return

        if self._slash_active:
            self._slash_active = False
            self.post_message(SlashCommandHide())

        at_pos = -1
        for i, ch in enumerate(text):
            if ch == "@":
                if i == 0 or text[i - 1] in (" ", "\t", "\n", "\r"):
                    at_pos = i

        if at_pos >= 0:
            kw_end = at_pos + 1
            while kw_end < len(text) and text[kw_end] not in (" ", "\t", "\n", "\r"):
                kw_end += 1
            space_after = kw_end < len(text) and text[kw_end] in (" ", "\t", "\n", "\r")
            if space_after or kw_end == at_pos + 1:
                if self._at_active:
                    self._at_active = False
                    self.post_message(AtCommandHide())
            else:
                if not self._at_active:
                    self._at_active = True
                self.post_message(AtCommandShow(text))
        elif self._at_active:
            self._at_active = False
            self.post_message(AtCommandHide())

    def on_key(self, event: events.Key) -> None:
        if self._slash_active:
            if event.key in ("up", "down"):
                event.stop()
                event.prevent_default()
                if event.key == "up":
                    self.post_message(SlashCommandNavigateUp())
                else:
                    self.post_message(SlashCommandNavigateDown())
                return
            if event.key == "enter":
                event.stop()
                event.prevent_default()
                self.post_message(SlashCommandSelect())
                return
            if event.key == "escape":
                event.stop()
                event.prevent_default()
                self.post_message(SlashCommandHide())
                return
        if self._at_active:
            if event.key in ("up", "down"):
                event.stop()
                event.prevent_default()
                if event.key == "up":
                    self.post_message(AtCommandNavigateUp())
                else:
                    self.post_message(AtCommandNavigateDown())
                return
            if event.key == "enter":
                event.stop()
                event.prevent_default()
                self.post_message(AtCommandSelect())
                return
            if event.key == "escape":
                event.stop()
                event.prevent_default()
                self.post_message(AtCommandHide())
                return
        if event.key == "up":
            cursor_row, _ = self.cursor_location
            if cursor_row > 0:
                return
            if self._input_history:
                event.stop()
                event.prevent_default()
                if self._history_index == -1:
                    self._current_input = self.text
                if self._history_index < len(self._input_history) - 1:
                    self._history_index += 1
                    self.text = self._input_history[-(self._history_index + 1)]
                return
        if event.key == "down":
            cursor_row, _ = self.cursor_location
            total_lines = len(self.document.lines)
            if cursor_row < total_lines - 1:
                return
            if self._input_history:
                event.stop()
                event.prevent_default()
                if self._history_index == -1:
                    return
                if self._history_index == 0:
                    self._history_index = -1
                    self.text = self._current_input
                else:
                    self._history_index -= 1
                    self.text = self._input_history[-(self._history_index + 1)]
                return
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            text = self.text
            if text.strip():
                self._input_history.append(text)
            self.reset_history_index()
            self.post_message(ChatInputSubmit())
        elif event.key in ("ctrl+enter", "ctrl+j"):
            event.stop()
            event.prevent_default()
            self.insert("\n")

    def action_dump(self) -> None:
        self.post_message(ChatInputDump())

    @property
    def input_history(self) -> list[str]:
        return self._input_history

    @input_history.setter
    def input_history(self, value: list[str]) -> None:
        self._input_history = value

    def set_slash_active(self, active: bool) -> None:
        self._slash_active = active

    def set_at_active(self, active: bool) -> None:
        self._at_active = active

    def reset_history_index(self) -> None:
        self._history_index = -1
        self._current_input = ""
