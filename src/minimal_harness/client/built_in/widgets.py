"""Custom widgets for the TUI."""

from __future__ import annotations

from textual import events
from textual.binding import Binding
from textual.message import Message
from textual.widgets import TextArea


class SlashCommandShow(Message):
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        super().__init__()


class SlashCommandHide(Message):
    pass


class SlashCommandNavigateUp(Message):
    pass


class SlashCommandNavigateDown(Message):
    pass


class SlashCommandSelect(Message):
    pass


class HistoryNavigateUp(Message):
    pass


class HistoryNavigateDown(Message):
    pass


class ChatInputSubmit(Message):
    pass


class ChatInputDump(Message):
    pass


class ChatInput(TextArea):
    BINDINGS = [Binding("ctrl+d", "dump", "Dump", show=True)]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._slash_active: bool = False
        self._input_history: list[str] = []
        self._history_index: int = -1
        self._current_input: str = ""

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        text = self.text
        if text.startswith("/"):
            self.post_message(SlashCommandShow(text))
        elif self._slash_active:
            self.post_message(SlashCommandHide())

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
        if event.key == "up":
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

    def set_slash_active(self, active: bool) -> None:
        self._slash_active = active

    def reset_history_index(self) -> None:
        self._history_index = -1
        self._current_input = ""
