"""Handoff tool widget — custom visualization for agent delegation."""

from __future__ import annotations

import json
from typing import Any

from rich.text import Text

from minimal_harness.client.built_in.chat_widgets import ChatMsg
from minimal_harness.client.built_in.tool_widget_provider import ToolWidgetProvider

_STATUS_ICONS: dict[str, str] = {
    "llm_start": "\U0001f9e0",
    "llm_end": "\U0001f4ac",
    "execution_start": "\u26a1",
    "execution_end": "\U0001f4e6",
    "tool_start": "\U0001f527",
    "tool_end": "\u2705",
    "agent_end": "\U0001f3c1",
}

_COLLAPSE_PREVIEW_LEN = 100


def _first_line(s: str) -> str:
    line = s.split("\n")[0]
    if len(line) > _COLLAPSE_PREVIEW_LEN:
        return line[:_COLLAPSE_PREVIEW_LEN] + "\u2026"
    return line


class HandoffWidget(ChatMsg):
    """Widget for handoff tool calls, showing delegation status and progress.

    Click the widget to toggle between collapsed (latest line only) and
    expanded (full timeline) views.
    """

    def __init__(self, target_agent: str, task: str = "", msg_id: str = "") -> None:
        self._target_agent = target_agent
        self._handoff_task = task
        self._steps: list[str] = []
        self._result: str | None = None
        self._error: str | None = None
        self._collapsed = True
        super().__init__(self._build_content(), id=msg_id)

    def _build_content(self) -> Text:
        return self._build_collapsed() if self._collapsed else self._build_expanded()

    def _build_header(self) -> Text:
        text = Text(no_wrap=False, overflow="fold")
        text.append("\U0001f91d Handoff \u2192 ", "bold bright_magenta")
        text.append(self._target_agent, "bold bright_cyan")
        return text

    def _build_collapsed(self) -> Text:
        text = self._build_header()
        step_count = len(self._steps)
        if self._error:
            text.append(f"\n  {_first_line(self._error)}", "bold bright_red")
        elif self._result:
            text.append(f"\n  {_first_line(self._result)}", "bright_green")
        elif self._steps:
            text.append(f"\n  {_first_line(self._steps[-1])}", "dim")
        hint_parts = []
        if step_count:
            hint_parts.append(f"{step_count} step(s)")
        hint_parts.append("click to expand")
        text.append(f"\n  \u25b8 {' \u00b7 '.join(hint_parts)}", "dim italic")
        return text

    def _build_expanded(self) -> Text:
        text = self._build_header()
        if self._handoff_task:
            text.append(f"\n  Task: {self._handoff_task}", "")
        if self._steps:
            for step in self._steps:
                text.append(f"\n  {step}", "")
        if self._result:
            text.append(f"\n\n  {self._result}", "bright_green")
        if self._error:
            text.append(f"\n  {self._error}", "bold bright_red")
        if self._steps or self._result or self._error:
            text.append("\n  \u25b8 click to collapse", "dim italic")
        return text

    def _refresh(self) -> None:
        self.update(self._build_content())

    def add_step(self, step: str) -> None:
        self._steps.append(step)
        self._refresh()

    def set_result(self, result: str) -> None:
        self._result = result
        self._refresh()

    def set_error(self, error: str) -> None:
        self._error = error
        self._refresh()

    def on_click(self) -> None:
        self._collapsed = not self._collapsed
        self.set_class(self._collapsed, "collapsed")
        self._refresh()


class HandoffWidgetProvider(ToolWidgetProvider):
    """Custom rendering for the ``handoff`` tool."""

    @property
    def tool_name(self) -> str:
        return "handoff"

    def make_widget(self, tool_call: dict, msg_id: str) -> ChatMsg:
        args_str = tool_call.get("arguments", "{}") or "{}"
        try:
            args = json.loads(args_str)
        except (json.JSONDecodeError, TypeError):
            args = {}
        target = args.get("target_agent_name", "unknown")
        task = args.get("task_description", "")
        return HandoffWidget(target, task, msg_id)

    def on_progress(self, widget: ChatMsg, chunk: Any) -> bool:
        if not isinstance(widget, HandoffWidget):
            return False
        d = chunk
        if isinstance(d, str):
            try:
                d = json.loads(d)
            except (json.JSONDecodeError, TypeError):
                widget.add_step(f"\u2022 {d}")
                return True
        if not isinstance(d, dict):
            widget.add_step(f"\u2022 {d}")
            return True
        status = d.get("status", "")
        msg = d.get("message", str(d))
        if status == "handoff_started":
            widget.add_step(f"\U0001f4cb {msg}")
        elif status == "progress":
            type_ = d.get("type", "")
            icon = _STATUS_ICONS.get(type_, "\u2022")
            widget.add_step(f"{icon} {msg}")
        elif status == "error":
            widget.add_step(f"\u274c {msg}")
        else:
            widget.add_step(f"\u2022 {msg}")
        return True

    def on_end(self, widget: ChatMsg, result: Any) -> bool:
        if not isinstance(widget, HandoffWidget):
            return False
        d = result
        if isinstance(d, str):
            try:
                d = json.loads(d)
            except (json.JSONDecodeError, TypeError):
                widget.set_result(f"\u2705 {d}")
                return True
        if isinstance(d, dict):
            status = d.get("status", "")
            if status == "handoff_complete":
                r = d.get("result", "") or d.get("message", "")
                widget.set_result(f"\u2705 Handoff complete \u2014 {r}")
            elif status == "error":
                widget.set_error(f"\u274c {d.get('message', '')}")
            elif "error" in d:
                widget.set_error(f"\u274c {d['error']}")
            else:
                widget.set_result(
                    f"\u2705 {json.dumps(d, ensure_ascii=False, default=str)}"
                )
            return True
        return False
