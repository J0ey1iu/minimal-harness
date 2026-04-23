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
    Select,
    Static,
)

from minimal_harness.client.built_in.config import (
    DEFAULT_CONFIG,
    SYSTEM_PROMPTS_DIR,
    THEMES,
    list_system_prompts,
)
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
                yield Input(self.cfg.get("model", ""), id="f-model")
                yield Label("System Prompt")
                current_prompt_path = self.cfg.get("system_prompt", "")
                system_prompts = list_system_prompts()
                prompt_options = [(p.name, str(p)) for p in system_prompts]
                if not prompt_options:
                    prompt_options = [(system_prompts[0].name, str(system_prompts[0]))] if system_prompts else [("default.md", str(SYSTEM_PROMPTS_DIR / "default.md"))]
                default_value = current_prompt_path if current_prompt_path in [str(p) for p in system_prompts] else prompt_options[0][1]
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
            self.dismiss(
                {
                    "base_url": self.query_one("#f-base", Input).value,
                    "api_key": self.query_one("#f-key", Input).value,
                    "model": self.query_one("#f-model", Input).value,
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

    def compose(self):
        with Vertical(classes="modal"):
            yield Label("🔧  Select Tools", classes="modal-title")
            with VerticalScroll(classes="modal-body"):
                for name in sorted(self.tools):
                    desc = self.tools[name].description or ""
                    with Vertical(classes="tool-item"):
                        yield Checkbox(
                            name, value=name in self.selected, id=f"cb-{name}"
                        )
                        if desc:
                            yield Static(desc, classes="tool-desc")
            with Horizontal(classes="modal-buttons"):
                yield Button("OK", variant="primary", id="ok")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ok":
            chosen = [
                n for n in self.tools if self.query_one(f"#cb-{n}", Checkbox).value
            ]
            self.dismiss(chosen)
        else:
            self.dismiss(None)