"""Modal screens for the TUI."""

from __future__ import annotations

from typing import Any

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
)

from minimal_harness.client.built_in.config import (
    DEFAULT_CONFIG,
    SYSTEM_PROMPTS_DIR,
    list_system_prompts,
    load_models,
)
from minimal_harness.client.built_in.constants import THEMES
from minimal_harness.tool.base import StreamingTool


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
                yield Label("System Prompt")
                current_prompt_path = self.cfg.get("system_prompt", "")
                system_prompts = list_system_prompts()
                prompt_options = [(p.name, str(p)) for p in system_prompts]
                if not prompt_options:
                    prompt_options = (
                        [(system_prompts[0].name, str(system_prompts[0]))]
                        if system_prompts
                        else [("default.md", str(SYSTEM_PROMPTS_DIR / "default.md"))]
                    )
                default_value = (
                    current_prompt_path
                    if current_prompt_path in [str(p) for p in system_prompts]
                    else prompt_options[0][1]
                )
                yield Select(
                    prompt_options,
                    value=default_value,
                    id="f-prompt",
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
                    "system_prompt": self.query_one("#f-prompt", Select).value,
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

    def __init__(self, tools: dict[str, StreamingTool], selected: set[str]) -> None:
        super().__init__()
        self.tools, self.selected = tools, selected
        self._id_map: dict[str, str] = {}

    @staticmethod
    def _safe_id(name: str) -> str:
        import re

        return re.sub(r"[^a-zA-Z0-9_-]", "_", name)

    def compose(self):
        with Vertical(classes="modal"):
            yield Label("🔧  Select Tools", classes="modal-title")
            with VerticalScroll(classes="modal-body"):
                for name in sorted(self.tools):
                    desc = self.tools[name].description or ""
                    safe = self._safe_id(name)
                    self._id_map[safe] = name
                    with Vertical(classes="tool-item"):
                        yield Checkbox(
                            name, value=name in self.selected, id=f"cb-{safe}"
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


class SessionSelectScreen(ModalScreen[str | None]):
    BINDINGS = [
        Binding("escape", "dismiss(None)", "Cancel"),
        Binding("enter", "select_session", "Load", show=False),
    ]

    def __init__(self, sessions: list[dict[str, Any]]) -> None:
        super().__init__()
        self.sessions = sessions

    def on_mount(self) -> None:
        if self.sessions:
            lv = self.query_one("#session-list", ListView)
            lv.focus()

    def _format_title(self, title: str, max_len: int = 30) -> str:
        if len(title) > max_len:
            return title[: max_len - 3] + "..."
        return title

    def compose(self):
        with Vertical(classes="modal"):
            yield Label("📁  Select Session", classes="modal-title")
            with VerticalScroll(classes="modal-body"):
                if not self.sessions:
                    yield Label(
                        "No saved sessions found.", classes="modal-message"
                    )
                else:
                    with ListView(id="session-list"):
                        for i, session in enumerate(self.sessions):
                            title = self._format_title(
                                session.get("title", "Untitled")
                            )
                            created = (
                                session.get("created_at", "")[:19].replace("T", " ")
                            )
                            msg_count = session.get("message_count", 0)
                            with ListItem(id=f"session-{i}"):
                                with Horizontal(classes="session-row"):
                                    yield Label(
                                        title, classes="session-title"
                                    )
                                    yield Label(
                                        created, classes="session-date"
                                    )
                                    yield Label(
                                        f"{msg_count} msgs",
                                        classes="session-count",
                                    )
            with Horizontal(classes="modal-buttons"):
                yield Button("Load", variant="primary", id="ok")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ok":
            try:
                lv = self.query_one("#session-list", ListView)
                if (
                    lv.index is not None
                    and 0 <= lv.index < len(self.sessions)
                ):
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
