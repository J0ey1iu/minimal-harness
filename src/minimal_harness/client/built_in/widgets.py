"""Custom widgets for the TUI."""

from __future__ import annotations

from textual import events
from textual.widgets import TextArea


class ChatInput(TextArea):
    def on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.app.action_submit()  # type: ignore[attr-defined]
        elif event.key in ("ctrl+enter", "ctrl+j"):
            event.stop()
            event.prevent_default()
            self.insert("\n")