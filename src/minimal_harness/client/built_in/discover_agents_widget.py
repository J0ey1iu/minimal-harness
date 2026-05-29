"""Discover agents widget — lists available agents for handoff."""

from __future__ import annotations

import json
from typing import Any

from rich.text import Text

from minimal_harness.client.built_in.chat_widgets import ChatMsg
from minimal_harness.client.built_in.tool_widget_provider import ToolWidgetProvider

_COLLAPSE_PREVIEW_LEN = 100


def _first_line(s: str) -> str:
    line = s.split("\n")[0]
    if len(line) > _COLLAPSE_PREVIEW_LEN:
        return line[:_COLLAPSE_PREVIEW_LEN] + "\u2026"
    return line


class DiscoverAgentsWidget(ChatMsg):
    """Widget listing discovered agents with name and description.

    Click the widget to toggle between collapsed and expanded views.
    """

    def __init__(self, msg_id: str = "") -> None:
        self._agents: list[dict[str, str]] = []
        self._error: str | None = None
        self._collapsed = True
        super().__init__(self._build_content(), id=msg_id)

    def _build_content(self) -> Text:
        return self._build_collapsed() if self._collapsed else self._build_expanded()

    def _build_header(self) -> Text:
        text = Text(no_wrap=False, overflow="fold")
        text.append("\U0001f50d Discover Agents", "bold bright_blue")
        return text

    def _build_collapsed(self) -> Text:
        text = self._build_header()
        if self._error:
            text.append(f"\n  {_first_line(self._error)}", "bold bright_red")
        elif self._agents:
            text.append(f"\n  {len(self._agents)} agent(s) found", "bright_green")
        else:
            text.append("\n  No agents available", "dim")
        text.append("\n  \u25b8 click to expand", "dim italic")
        return text

    def _build_expanded(self) -> Text:
        text = self._build_header()
        text.append(f"\n  {'\u2500' * 40}", "dim")
        if self._error:
            text.append(f"\n  \u2717 {self._error}", "bold bright_red")
        elif self._agents:
            for a in self._agents:
                name = a.get("display_name", a.get("name", "?"))
                desc = a.get("description", "")
                text.append(f"\n  \U0001f916 {name}", "bold bright_cyan")
                if desc:
                    text.append(f"\n     {desc}", "dim")
        else:
            text.append("\n  No agents available", "dim")
        text.append(f"\n  {'\u2500' * 40}", "dim")
        if self._agents:
            text.append(f"\n  {len(self._agents)} agent(s) found", "bright_green")
        text.append("\n  \u25b8 click to collapse", "dim italic")
        return text

    def _refresh(self) -> None:
        self.update(self._build_content())

    def set_agents(self, agents: list[dict[str, str]]) -> None:
        self._agents = agents
        self._refresh()

    def set_error(self, error: str) -> None:
        self._error = error
        self._refresh()

    def on_click(self) -> None:
        self._collapsed = not self._collapsed
        self.set_class(self._collapsed, "collapsed")
        self._refresh()


class DiscoverAgentsWidgetProvider(ToolWidgetProvider):
    """Custom rendering for the ``discover_agents`` tool."""

    @property
    def tool_name(self) -> str:
        return "discover_agents"

    def make_widget(self, tool_call: dict, msg_id: str) -> ChatMsg:
        return DiscoverAgentsWidget(msg_id)

    def on_progress(self, widget: ChatMsg, chunk: Any) -> bool:
        if not isinstance(widget, DiscoverAgentsWidget):
            return False
        d = chunk
        if isinstance(d, str):
            try:
                d = json.loads(d)
            except (json.JSONDecodeError, TypeError):
                return False
        if isinstance(d, dict) and d.get("status") == "ok":
            agents = d.get("agents", [])
            if isinstance(agents, list):
                widget.set_agents(agents)
                return True
        return False

    def on_end(self, widget: ChatMsg, result: Any) -> bool:
        if not isinstance(widget, DiscoverAgentsWidget):
            return False
        d = result
        if isinstance(d, str):
            try:
                d = json.loads(d)
            except (json.JSONDecodeError, TypeError):
                widget.set_error(str(d))
                return True
        if isinstance(d, dict):
            if d.get("status") == "ok":
                agents = d.get("agents", [])
                if isinstance(agents, list):
                    widget.set_agents(agents)
                    return True
            if "error" in d:
                widget.set_error(str(d["error"]))
                return True
        return False
