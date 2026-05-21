"""Modal screens for the TUI."""

from __future__ import annotations

import re
from datetime import datetime
from typing import TYPE_CHECKING, Any

from textual import events
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    Label,
    ListItem,
    ListView,
    Select,
    Static,
    TextArea,
)

from minimal_harness.client.built_in.config import DEFAULT_CONFIG, load_models
from minimal_harness.client.built_in.constants import THEMES

if TYPE_CHECKING:
    from minimal_harness.types import ToolMetadata


class ConfigScreen(ModalScreen[dict | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "Cancel")]

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = config

    def compose(self):
        with Vertical(classes="modal"):
            yield Label("⚙  Configuration", classes="modal-title")
            with VerticalScroll(classes="modal-body"):
                yield Label("Base URL")
                yield Input(
                    self.cfg.get("base_url", ""), id="f-base", placeholder="https://..."
                )
                yield Label("API Key")
                yield Input(
                    self.cfg.get("api_key", ""),
                    id="f-key",
                    password=True,
                    placeholder="sk-...",
                )
                yield Label("Model")
                current_model = self.cfg.get("model", "")
                models = load_models()
                if current_model and current_model not in models:
                    models.insert(0, current_model)
                if not models:
                    models = [current_model] if current_model else [""]
                model_options = [(m, m) for m in models]
                default_model = current_model if current_model in models else models[0]
                yield Select(
                    model_options,
                    value=default_model,
                    id="f-model",
                    allow_blank=False,
                )
                yield Label("Tools Path")
                yield Input(self.cfg.get("tools_path", ""), id="f-tools")
                yield Label("Theme")
                yield Select(
                    [(t, t) for t in THEMES],
                    value=self.cfg.get("theme", DEFAULT_CONFIG["theme"]),
                    id="f-theme",
                    allow_blank=False,
                )
            with Horizontal(classes="modal-buttons"):
                yield Button("Save", variant="primary", id="ok")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ok":
            theme = self.query_one("#f-theme", Select).value
            model = self.query_one("#f-model", Select).value
            self.dismiss(
                {
                    "base_url": self.query_one("#f-base", Input).value,
                    "api_key": self.query_one("#f-key", Input).value,
                    "model": model if isinstance(model, str) else "",
                    "tools_path": self.query_one("#f-tools", Input).value,
                    "theme": theme
                    if isinstance(theme, str)
                    else DEFAULT_CONFIG["theme"],
                }
            )
        else:
            self.dismiss(None)


class ConfirmScreen(ModalScreen[bool]):
    BINDINGS = [Binding("escape", "dismiss(False)", "Cancel")]

    def __init__(
        self, title: str, message: str, ok: str = "OK", variant: str = "primary"
    ) -> None:
        super().__init__()
        self.t, self.m, self.ok_label, self.variant = title, message, ok, variant

    def compose(self):
        with Vertical(classes="modal small"):
            yield Label(self.t, classes="modal-title")
            yield Label(self.m, classes="modal-message")
            with Horizontal(classes="modal-buttons"):
                yield Button(self.ok_label, variant=self.variant, id="ok")  # type: ignore[arg-type]
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "ok")


class PromptScreen(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "Cancel")]

    def __init__(self, title: str, default: str = "") -> None:
        super().__init__()
        self.t, self.default = title, default

    def compose(self):
        with Vertical(classes="modal small"):
            yield Label(self.t, classes="modal-title")
            yield Input(value=self.default, id="value")
            with Horizontal(classes="modal-buttons"):
                yield Button("Save", variant="primary", id="ok")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ok":
            self.dismiss(self.query_one("#value", Input).value.strip() or None)
        else:
            self.dismiss(None)


class ToolSelectScreen(ModalScreen[list[str] | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "Cancel")]

    def __init__(self, tools: dict[str, ToolMetadata], selected: set[str]) -> None:
        super().__init__()
        self.tools, self.selected = tools, selected
        self._id_map: dict[str, str] = {}

    @staticmethod
    def _safe_id(name: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_-]", "_", name)

    def compose(self):
        with Vertical(classes="modal"):
            yield Label("🔧  Select Tools", classes="modal-title")
            with VerticalScroll(classes="modal-body"):
                for name in sorted(self.tools):
                    tool = self.tools[name]
                    resolve_dn = getattr(tool, "resolve_display_name", None)
                    display_name = (
                        resolve_dn()
                        if resolve_dn
                        else (getattr(tool, "display_name", None) or name)
                    )
                    resolve_desc = getattr(tool, "resolve_description", None)
                    desc = (resolve_desc() if resolve_desc else tool.description) or ""
                    safe = self._safe_id(name)
                    self._id_map[safe] = name
                    with Vertical(classes="tool-item"):
                        yield Checkbox(
                            display_name, value=name in self.selected, id=f"cb-{safe}"
                        )
                        if desc:
                            yield Static(desc, classes="tool-desc")
            with Horizontal(classes="modal-buttons"):
                yield Button("Save", variant="primary", id="ok")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ok":
            chosen = [
                name
                for safe, name in self._id_map.items()
                if self.query_one(f"#cb-{safe}", Checkbox).value
            ]
            self.dismiss(chosen)
        else:
            self.dismiss(None)


class AgentSelectScreen(ModalScreen[dict[str, str] | None]):
    BINDINGS = [
        Binding("escape", "dismiss(None)", "Cancel"),
        Binding("enter", "select_agent", "Select", show=False),
    ]

    def __init__(self, agents: list[dict[str, str]]) -> None:
        super().__init__()
        self.agents = agents

    def on_mount(self) -> None:
        if self.agents:
            lv = self.query_one("#agent-list", ListView)
            lv.focus()

    def compose(self):
        with Vertical(classes="modal session-select"):
            yield Label("🤖  Select Agent", classes="modal-title")
            with Vertical(classes="modal-body"):
                if not self.agents:
                    yield Label("No agents configured.", classes="modal-message")
                else:
                    with ListView(id="agent-list"):
                        for i, agent in enumerate(self.agents):
                            name = agent.get("name", "Unknown")
                            display_name = agent.get("display_name") or name
                            desc = agent.get("description", "")
                            with ListItem(id=f"agent-{i}"):
                                with Vertical():
                                    yield Label(display_name, classes="session-title")
                                    if desc:
                                        yield Label(desc, classes="tool-desc")
            with Horizontal(classes="modal-buttons"):
                yield Button("Start Chat", variant="primary", id="ok")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ok":
            try:
                lv = self.query_one("#agent-list", ListView)
                if lv.index is not None and 0 <= lv.index < len(self.agents):
                    self.dismiss(self.agents[lv.index])
                    return
            except Exception:
                pass
        self.dismiss(None)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx = event.list_view.index
        if idx is not None and 0 <= idx < len(self.agents):
            self.dismiss(self.agents[idx])


class SessionSelectScreen(ModalScreen[str | None]):
    BINDINGS = [
        Binding("escape", "dismiss(None)", "Cancel"),
        Binding("enter", "select_session", "Load", show=False),
    ]

    def __init__(
        self,
        sessions: list[dict[str, Any]],
        controller: Any | None = None,
    ) -> None:
        super().__init__()
        self.sessions = sessions
        self._controller = controller

    async def on_mount(self) -> None:
        if self._controller is not None:
            self._controller.add_status_listener(self._on_status_changed)
        if self.sessions:
            lv = self.query_one("#session-list", ListView)
            lv.focus()

    async def on_unmount(self) -> None:
        if self._controller is not None:
            self._controller.remove_status_listener(self._on_status_changed)

    async def _on_status_changed(self, session_id: str, status: Any) -> None:
        await self._refresh_sessions()

    async def _refresh_sessions(self) -> None:
        if self._controller is None:
            return
        self.sessions = await self._controller.get_all_sessions_metadata()
        lv = self.query_one("#session-list", ListView)
        await lv.clear()
        for i, session in enumerate(self.sessions):
            lv.append(self._build_item(i, session))

    def _format_title(self, title: str, max_len: int = 100) -> str:
        if len(title) > max_len:
            return title[: max_len - 3] + "..."
        return title

    @staticmethod
    def _format_relative_time(iso_str: str) -> str:
        if not iso_str:
            return ""
        try:
            dt = datetime.fromisoformat(iso_str)
            now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
            diff = now - dt
            secs = int(diff.total_seconds())
            if secs < 0:
                return "just now"
            if secs < 60:
                return f"{secs}s ago"
            if secs < 3600:
                return f"{secs // 60}m ago"
            if secs < 86400:
                return f"{secs // 3600}h ago"
            days = secs // 86400
            if days == 1:
                return "yesterday"
            if days < 30:
                return f"{days}d ago"
            return iso_str[:10]
        except (ValueError, TypeError):
            return iso_str[:19].replace("T", " ") if iso_str else ""

    @staticmethod
    def _build_item(i: int, session: dict[str, Any]) -> ListItem:
        title = session.get("title", "Untitled") or "Untitled"
        if len(title) > 100:
            title = title[:97] + "..."
        created = session.get("created_at", "")
        relative_time = SessionSelectScreen._format_relative_time(created)
        msg_count = session.get("message_count", 0)
        agent_name = session.get("agent_name", "")
        status = session.get("status", "idle")

        if status == "running":
            display_title = f"[bold $warning]\u25cf Running[/]  {title}"
        else:
            display_title = title

        meta_children: list[Label] = [
            Label(relative_time, classes="session-date"),
            Label(f"{msg_count} msgs", classes="session-count"),
        ]
        if agent_name:
            meta_children.append(Label(agent_name, classes="session-agent"))

        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", session.get("session_id", str(i)))
        return ListItem(
            Vertical(
                Label(display_title, classes="session-title", markup=True),
                Horizontal(*meta_children, classes="session-meta"),
            ),
            id=f"session-{safe_id}",
        )

    def compose(self):
        with Vertical(classes="modal session-select"):
            yield Label("\U0001f4c1  Select Session", classes="modal-title")
            with Vertical(classes="modal-body"):
                if not self.sessions:
                    yield Label("No saved sessions found.", classes="modal-message")
                else:
                    with ListView(id="session-list"):
                        for i, session in enumerate(self.sessions):
                            yield self._build_item(i, session)
            with Horizontal(classes="modal-buttons"):
                yield Button("Load", variant="primary", id="ok")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ok":
            try:
                lv = self.query_one("#session-list", ListView)
                if lv.index is not None and 0 <= lv.index < len(self.sessions):
                    self.dismiss(self.sessions[lv.index]["session_id"])
                    return
            except Exception:
                pass
            self.dismiss(None)
        else:
            self.dismiss(None)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx = event.list_view.index
        if idx is not None and 0 <= idx < len(self.sessions):
            self.dismiss(self.sessions[idx]["session_id"])


class CopySelectScreen(ModalScreen[str | None]):
    BINDINGS = [
        Binding("escape", "dismiss(None)", "Cancel"),
    ]

    def __init__(self, messages: list[tuple[str, str]]) -> None:
        super().__init__()
        self.messages = messages

    def compose(self):
        with Vertical(classes="modal session-select"):
            yield Label("📋  Copy Message", classes="modal-title")
            with Vertical(classes="modal-body"):
                if not self.messages:
                    yield Label(
                        "No assistant messages to copy.", classes="modal-message"
                    )
                else:
                    with ListView(id="copy-list"):
                        for i, (preview, _) in enumerate(self.messages):
                            with ListItem(id=f"copy-{i}"):
                                yield Label(
                                    preview,
                                    classes="session-title",
                                )
            with Horizontal(classes="modal-buttons"):
                yield Button("Copy Selected", variant="primary", id="ok")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ok":
            try:
                lv = self.query_one("#copy-list", ListView)
                if lv.index is not None and 0 <= lv.index < len(self.messages):
                    self.dismiss(self.messages[lv.index][1])
                    return
            except Exception:
                pass
        self.dismiss(None)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx = event.list_view.index
        if idx is not None and 0 <= idx < len(self.messages):
            self.dismiss(self.messages[idx][1])


class ErrorScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape", "dismiss", "Close")]

    def __init__(self, errors: list[dict[str, Any]]) -> None:
        super().__init__()
        self.errors = errors

    def compose(self):
        with Vertical(classes="modal error-modal"):
            yield Label(f"\u26a0  Errors ({len(self.errors)})", classes="modal-title")
            with VerticalScroll(classes="modal-body"):
                for i, err in enumerate(self.errors):
                    yield Label(
                        f"[{err.get('timestamp', '')}] {err.get('brief', '')}",
                        classes="error-brief",
                    )
                    ta = TextArea(
                        err.get("formatted", ""),
                        read_only=True,
                        show_line_numbers=False,
                        id=f"error-detail-{i}",
                    )
                    ta.border_title = f"Error #{i + 1}"
                    yield ta
            with Horizontal(classes="modal-buttons"):
                yield Button("Copy All", variant="primary", id="copy")
                yield Button("Close", id="close")

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            self.dismiss()
            event.stop()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "copy":
            import asyncio

            all_text = "\n\n---\n\n".join(e.get("formatted", "") for e in self.errors)
            asyncio.create_task(self._copy_to_clipboard(all_text))
        else:
            self.dismiss()

    async def _copy_to_clipboard(self, text: str) -> None:
        import asyncio

        try:
            proc = await asyncio.create_subprocess_exec(
                "pbcopy",
                stdin=asyncio.subprocess.PIPE,
            )
            if proc.stdin:
                proc.stdin.write(text.encode("utf-8"))
                await proc.stdin.drain()
                proc.stdin.close()
            await proc.wait()
        except Exception:
            pass
