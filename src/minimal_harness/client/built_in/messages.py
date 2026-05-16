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


class AtCommandShow(Message):
    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__()


class AtCommandHide(Message):
    pass


class AtCommandNavigateUp(Message):
    pass


class AtCommandNavigateDown(Message):
    pass


class AtCommandSelect(Message):
    pass


class ChatInputSubmit(Message):
    pass


class ChatInputDump(Message):
    pass


class SessionNotificationClicked(Message):
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__()
