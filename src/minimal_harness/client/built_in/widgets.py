"""Custom widgets for the TUI."""

from __future__ import annotations

from textual import events
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


class ChatInput(TextArea):
    _slash_active: bool = False

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
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.app.action_submit()  # type: ignore[attr-defined]
        elif event.key in ("ctrl+enter", "ctrl+j"):
            event.stop()
            event.prevent_default()
            self.insert("\n")

    def set_slash_active(self, active: bool) -> None:
        self._slash_active = active