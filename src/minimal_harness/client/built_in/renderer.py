"""Chat rendering components for TUI."""

from __future__ import annotations

import json
from io import StringIO
from typing import TYPE_CHECKING

from rich.console import Console
from rich.text import Text

from .constants import MAX_DISPLAY_LENGTH
from .markdown_styles import MD_THEME, AppMarkdown

if TYPE_CHECKING:
    pass


class ChatRenderer:
    def __init__(self, committed: list[Text]) -> None:
        self._committed = committed

    @property
    def committed(self) -> list[Text]:
        return self._committed

    def _render_markdown(self, text: str, width: int = 80) -> Text:
        buf = StringIO()
        with Console(
            file=buf, force_terminal=True, width=width, theme=MD_THEME
        ) as console:
            console.print(AppMarkdown(text))
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
        return format_tool_call_static(call)

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
            return Text(f"{full_err}", style="bold bright_red")
        else:
            if isinstance(result, dict):
                s = json.dumps(result, ensure_ascii=False, default=str)
            elif isinstance(result, str):
                s = result
            else:
                s = str(result)
            if len(s) > MAX_DISPLAY_LENGTH:
                s = s[:MAX_DISPLAY_LENGTH] + "…"
            return Text(f"{s}", "bright_green")

    def truncate(self, text: str, max_len: int = MAX_DISPLAY_LENGTH) -> str:
        if len(text) > max_len:
            return text[:max_len] + "…"
        return text


def format_tool_call_static(call: dict) -> Text:
    name = call.get("name", "?")
    args_raw = call.get("arguments", "{}")
    try:
        parsed = json.loads(args_raw)
        args_str = json.dumps(parsed, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        args_str = args_raw

    text = Text()
    text.append(name, "bold bright_yellow")

    has_content = args_str and args_str not in ("{}", "")
    if has_content:
        text.append(f"({args_str})", "")
    else:
        text.append("()", "")
    return text


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
        return Text(f"{full_err}", style="bold bright_red")
    else:
        if isinstance(result, dict):
            s = json.dumps(result, ensure_ascii=False, indent=2, default=str)
        elif isinstance(result, str):
            s = result
        else:
            s = str(result)
        if len(s) > MAX_DISPLAY_LENGTH:
            s = s[:MAX_DISPLAY_LENGTH] + "…"
        return Text(f"{s}", "bright_green")


def truncate_static(text: str, max_len: int = MAX_DISPLAY_LENGTH) -> str:
    if len(text) > max_len:
        return text[:max_len] + "…"
    return text
