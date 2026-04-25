"""Chat rendering components for TUI."""

from __future__ import annotations

import json
from io import StringIO
from typing import TYPE_CHECKING

from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

if TYPE_CHECKING:
    pass


MAX_DISPLAY_LENGTH = 500


class ChatRenderer:
    def __init__(self, committed: list[Text]) -> None:
        self._committed = committed

    @property
    def committed(self) -> list[Text]:
        return self._committed

    def _render_markdown(self, text: str, width: int = 80) -> Text:
        buf = StringIO()
        with Console(file=buf, force_terminal=True, width=width) as console:
            console.print(Markdown(text))
        return Text.from_ansi(buf.getvalue())

    def say(
        self,
        text: str,
        style: str = "",
        is_markdown: bool = False,
        log_width: int = 80,
    ) -> Text:
        if is_markdown:
            t = self._render_markdown(text, log_width)
        elif style:
            t = Text(text, style=style)
        else:
            t = Text(text)
        self._committed.append(t)
        return t

    def format_tool_call(self, call: dict) -> Text:
        try:
            args = json.dumps(
                json.loads(call.get("arguments", "{}")), ensure_ascii=False
            )
        except (json.JSONDecodeError, TypeError):
            args = call.get("arguments", "")
        return Text(f"  ▸ {call.get('name', '?')}({args})", style="bold #f9e2af")

    def format_tool_result(self, result: dict | str) -> Text:
        if isinstance(result, dict) and "error" in result:
            err_msg = result.get("error", "Unknown error")
            tb = result.get("traceback", "") or ""
            stderr = result.get("stderr", "") or ""
            full_err = err_msg
            if tb:
                full_err += "\n\nTraceback:\n" + tb
            if stderr:
                full_err += "\n\nStderr:\n" + stderr
            return Text(f"    ✗ {full_err}", style="bold #f38ba8")
        else:
            if isinstance(result, dict):
                s = json.dumps(result, ensure_ascii=False, default=str)
            elif isinstance(result, str):
                s = result
            else:
                s = str(result)
            if len(s) > MAX_DISPLAY_LENGTH:
                s = s[:MAX_DISPLAY_LENGTH] + "…"
            return Text(f"    ✓ {s}", "#a6e3a1")

    def truncate(self, text: str, max_len: int = MAX_DISPLAY_LENGTH) -> str:
        if len(text) > max_len:
            return text[:max_len] + "…"
        return text


def format_tool_call_static(call: dict) -> Text:
    try:
        args = json.dumps(
            json.loads(call.get("arguments", "{}")), ensure_ascii=False
        )
    except (json.JSONDecodeError, TypeError):
        args = call.get("arguments", "")
    return Text(f"  ▸ {call.get('name', '?')}({args})", style="bold #f9e2af")


def format_tool_result_static(result: dict | str) -> Text:
    if isinstance(result, dict) and "error" in result:
        err_msg = result.get("error", "Unknown error")
        tb = result.get("traceback", "") or ""
        stderr = result.get("stderr", "") or ""
        full_err = err_msg
        if tb:
            full_err += "\n\nTraceback:\n" + tb
        if stderr:
            full_err += "\n\nStderr:\n" + stderr
        return Text(f"    ✗ {full_err}", style="bold #f38ba8")
    else:
        if isinstance(result, dict):
            s = json.dumps(result, ensure_ascii=False, default=str)
        elif isinstance(result, str):
            s = result
        else:
            s = str(result)
        if len(s) > MAX_DISPLAY_LENGTH:
            s = s[:MAX_DISPLAY_LENGTH] + "…"
        return Text(f"    ✓ {s}", "#a6e3a1")


def truncate_static(text: str, max_len: int = MAX_DISPLAY_LENGTH) -> str:
    if len(text) > max_len:
        return text[:max_len] + "…"
    return text