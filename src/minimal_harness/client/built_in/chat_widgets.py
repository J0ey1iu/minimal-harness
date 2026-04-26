"""Chat message widgets that natively wrap to container width."""
from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.widgets import Static


class ChatMsg(Static):
    """Base class for all chat messages. Handles text wrapping natively."""

    def __init__(
        self,
        content: Any = "",
        *,
        id: str | None = None,
    ) -> None:
        if isinstance(content, str):
            content = Text(content, no_wrap=False, overflow="fold")
        elif isinstance(content, Text):
            content.no_wrap = False
            content.overflow = "fold"
        super().__init__(content, id=id)

    def update(self, content: Any = "", *, layout: bool = True) -> None:
        if isinstance(content, str):
            content = Text(content, no_wrap=False, overflow="fold")
        elif isinstance(content, Text):
            content.no_wrap = False
            content.overflow = "fold"
        super().update(content, layout=layout)


class UserMsg(ChatMsg):
    """User input message."""


class ReasoningMsg(ChatMsg):
    """Thinking/reasoning content."""


class ToolCallMsg(ChatMsg):
    """Tool call display."""


class ToolResultMsg(ChatMsg):
    """Tool result display."""


class AssistantMsg(ChatMsg):
    """Assistant answer content (streaming or committed)."""
