from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from textual import events
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.widgets import Input, Static, TextArea

if TYPE_CHECKING:
    pass


class SystemPromptScreen(Screen):
    def __init__(self, current_prompt: str, on_save: Callable[[str], None]):
        super().__init__()
        self._current_prompt = current_prompt
        self._on_save = on_save

    def compose(self) -> "ComposeResult":
        with Container(id="prompt-modal"):
            yield Static("Edit System Prompt", classes="modal-title")
            yield TextArea(
                self._current_prompt,
                id="prompt-editor",
                classes="prompt-editor",
            )
            with Horizontal(id="modal-buttons"):
                yield Input(value="Ctrl+Enter to save", id="save-hint", disabled=True)
                yield Static("Esc to cancel", classes="modal-hint")

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            self.app.pop_screen()
        elif event.key == "ctrl+enter":
            new_prompt = self.query_one("#prompt-editor", TextArea).text
            self._on_save(new_prompt)
            self.app.pop_screen()
