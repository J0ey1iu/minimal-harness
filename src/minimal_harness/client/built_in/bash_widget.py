"""Bash tool widget — terminal-style visualization for command execution."""

from __future__ import annotations

import json
from typing import Any

from rich.text import Text

from minimal_harness.client.built_in.chat_widgets import ChatMsg
from minimal_harness.client.built_in.tool_widget_provider import ToolWidgetProvider
from minimal_harness.types import ToolResult

_MAX_OUTPUT_LINES = 80
_MAX_LINE_LENGTH = 200


def _truncate_line(line: str) -> str:
    if len(line) > _MAX_LINE_LENGTH:
        return line[:_MAX_LINE_LENGTH] + "\u2026"
    return line


class BashWidget(ChatMsg):
    """Terminal-style widget showing command, live output, and exit status."""

    def __init__(
        self, command: str, msg_id: str = "", timeout: float | None = None
    ) -> None:
        self._command = command
        self._timeout = timeout
        self._output_lines: list[str] = []
        self._exit_code: int | None = None
        self._stderr: str | None = None
        self._running = True
        super().__init__(self._build_content(), id=msg_id)

    def _build_content(self) -> Text:
        text = Text(no_wrap=False, overflow="fold")
        text.append("\U0001f4bb Bash", "bold bright_blue")
        text.append(f"\n  $ {self._command}", "bright_cyan")
        sep = "\u2500" * 40
        if self._output_lines or not self._running:
            text.append(f"\n  {sep}", "dim")
        if self._output_lines:
            overflow = len(self._output_lines) - _MAX_OUTPUT_LINES
            lines = (
                self._output_lines[-_MAX_OUTPUT_LINES:]
                if overflow > 0
                else self._output_lines
            )
            if overflow > 0:
                text.append(f"\n  \u2026 ({overflow} earlier lines truncated)", "dim")
            for line in lines:
                text.append(f"\n  {_truncate_line(line)}", "dim bright_black")
        if not self._running:
            text.append(f"\n  {sep}", "dim")
            if self._exit_code == 0:
                text.append(f"\n  Exit: {self._exit_code}", "bold bright_green")
            else:
                text.append(f"\n  Exit: {self._exit_code}", "bold bright_red")
                if self._stderr:
                    text.append(f"\n  {_truncate_line(self._stderr)}", "bright_red")
        return text

    def append_output(self, output: str) -> None:
        for line in output.split("\n"):
            self._output_lines.append(line)
        self.update(self._build_content())

    def set_complete(self, exit_code: int, stderr: str | None = None) -> None:
        self._running = False
        self._exit_code = exit_code
        self._stderr = stderr
        self.update(self._build_content())


class BashWidgetProvider(ToolWidgetProvider):
    """Custom terminal-style rendering for the ``bash`` tool."""

    @property
    def tool_name(self) -> str:
        return "bash"

    def make_widget(self, tool_call: dict, msg_id: str) -> ChatMsg:
        args_str = tool_call.get("arguments", "{}") or "{}"
        try:
            args = json.loads(args_str)
        except (json.JSONDecodeError, TypeError):
            args = {}
        command = args.get("command", "")
        timeout = args.get("timeout")
        return BashWidget(command, msg_id, timeout)

    def on_progress(self, widget: ChatMsg, chunk: Any) -> bool:
        if not isinstance(widget, BashWidget):
            return False
        if isinstance(chunk, dict) and chunk.get("status") == "progress":
            widget.append_output(chunk.get("message", ""))
            return True
        return False

    def on_end(self, widget: ChatMsg, result: Any) -> bool:
        if not isinstance(widget, BashWidget):
            return False
        if isinstance(result, ToolResult):
            exit_code = -1
            stderr = None
            if result.meta:
                try:
                    exit_code = int(result.meta.get("exit_code", -1))
                except (ValueError, TypeError):
                    exit_code = -1
                stderr = result.meta.get("stderr")
            widget.set_complete(exit_code, stderr)
            return True
        if isinstance(result, str):
            widget.append_output(result)
            widget.set_complete(-1)
            return True
        return False
