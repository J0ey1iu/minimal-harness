"""Custom Textual Message types for the TUI."""

from __future__ import annotations

from textual.message import Message


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


class ChatInputSubmit(Message):
    pass


class ChatInputDump(Message):
    pass
