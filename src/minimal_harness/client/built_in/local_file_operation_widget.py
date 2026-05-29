"""File operation widget — visual feedback for read/write/patch/delete."""

from __future__ import annotations

import json
from typing import Any

from rich.text import Text

from minimal_harness.client.built_in.chat_widgets import ChatMsg
from minimal_harness.client.built_in.tool_widget_provider import ToolWidgetProvider
from minimal_harness.types import ToolResult

_MAX_CONTENT_LINES = 60
_MAX_LINE_LENGTH = 200


def _trunc(s: str) -> str:
    if len(s) > _MAX_LINE_LENGTH:
        return s[:_MAX_LINE_LENGTH] + "\u2026"
    return s


_MODE_ICON: dict[str, str] = {
    "read": "\U0001f4d6",
    "write": "\u270f\ufe0f",
    "patch": "\U0001f527",
    "delete": "\U0001f5d1\ufe0f",
}


class FileOpWidget(ChatMsg):
    """Widget showing file path, mode, output content, and result."""

    def __init__(self, file_path: str, mode: str, msg_id: str = "") -> None:
        self._file_path = file_path
        self._mode = mode
        self._status: str | None = None
        self._content_lines: list[str] = []
        self._meta: dict | None = None
        self._error: str | None = None
        self._done = False
        super().__init__(self._build_content(), id=msg_id)

    def _build_content(self) -> Text:
        text = Text(no_wrap=False, overflow="fold")
        icon = _MODE_ICON.get(self._mode, "\U0001f4c4")
        text.append(f"{icon} File {self._mode.title()}", "bold bright_blue")
        text.append(f"\n  {self._file_path}", "bright_cyan")
        if self._status:
            text.append(f"\n  {self._status}", "dim")
        sep = "\u2500" * 40
        has_body = bool(self._content_lines or self._error or self._done)
        if has_body:
            text.append(f"\n  {sep}", "dim")
        if self._content_lines:
            overflow = len(self._content_lines) - _MAX_CONTENT_LINES
            lines = (
                self._content_lines[-_MAX_CONTENT_LINES:]
                if overflow > 0
                else self._content_lines
            )
            if overflow > 0:
                text.append(f"\n  \u2026 ({overflow} earlier lines truncated)", "dim")
            for line in lines:
                text.append(f"\n  {_trunc(line)}", "")
        if self._error:
            text.append(f"\n  \u2717 {self._error}", "bold bright_red")
        if self._done and not self._error:
            text.append(f"\n  {sep}", "dim")
            if self._meta:
                total = self._meta.get("total_lines")
                line_range = self._meta.get("range")
                if total is not None:
                    text.append(f"\n  {total} line(s) total", "bright_green")
                if line_range:
                    text.append(
                        f"\n  Lines {line_range[0]}\u2013{line_range[1]}", "dim"
                    )
                fp = self._meta.get("file_path")
                if fp and self._mode != "read":
                    text.append("\n  \u2705 OK", "bold bright_green")
        return text

    def set_status(self, status: str) -> None:
        self._status = status
        self.update(self._build_content())

    def append_content(self, content: str) -> None:
        for line in content.split("\n"):
            self._content_lines.append(line)
        self.update(self._build_content())

    def set_done(self, meta: dict | None, error: str | None = None) -> None:
        self._done = True
        self._meta = meta
        self._error = error
        self.update(self._build_content())


class FileOpWidgetProvider(ToolWidgetProvider):
    """Custom rendering for the ``local_file_operation`` tool."""

    @property
    def tool_name(self) -> str:
        return "local_file_operation"

    def make_widget(self, tool_call: dict, msg_id: str) -> ChatMsg:
        args_str = tool_call.get("arguments", "{}") or "{}"
        try:
            args = json.loads(args_str)
        except (json.JSONDecodeError, TypeError):
            args = {}
        return FileOpWidget(
            file_path=args.get("file_path", "?"),
            mode=args.get("mode", "?"),
            msg_id=msg_id,
        )

    def on_progress(self, widget: ChatMsg, chunk: Any) -> bool:
        return False

    def on_end(self, widget: ChatMsg, result: Any) -> bool:
        if not isinstance(widget, FileOpWidget):
            return False
        if isinstance(result, ToolResult):
            meta = result.meta or {}
            if meta.get("error"):
                widget.set_done(meta, error=str(meta["error"]))
            elif result.content and not isinstance(result.content, str):
                widget.set_done(meta, error=str(result.content))
            else:
                content = str(result.content) if result.content else ""
                widget.append_content(content)
                widget.set_done(meta)
            return True
        if isinstance(result, str):
            widget.append_content(result)
            widget.set_done(None)
            return True
        return False
