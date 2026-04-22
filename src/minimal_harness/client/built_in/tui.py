"""Terminal UI client for the minimal-harness framework."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from openai import AsyncOpenAI
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    TextArea,
)

from minimal_harness.agent.openai import OpenAIAgent
from minimal_harness.client.client import FrameworkClient
from minimal_harness.client.events import (
    AgentEndEvent,
    Event,
    ExecutionStartEvent,
    LLMChunkEvent,
    LLMEndEvent,
    ToolEndEvent,
    ToolProgressEvent,
    ToolStartEvent,
)
from minimal_harness.llm.openai import OpenAILLMProvider
from minimal_harness.memory import ConversationMemory
from minimal_harness.tool.base import StreamingTool
from minimal_harness.tool.built_in.bash import get_tools as get_bash_tools
from minimal_harness.tool.built_in.patch_file import get_tools as get_patch_file_tools
from minimal_harness.tool.external_loader import load_external_tools
from minimal_harness.tool.registry import ToolRegistry

# --- Config -----------------------------------------------------------------

CONFIG_FILE = Path.home() / ".minimal_harness" / "config.json"
FLUSH_INTERVAL = 0.25  # seconds; render cadence during streaming

DEFAULT_CONFIG: dict[str, Any] = {
    "base_url": "https://aihubmix.com/v1",
    "api_key": "",
    "model": "qwen3.5-27b",
    "system_prompt": "You are a helpful assistant.",
    "tools_path": "",
    "theme": "tokyo-night",
    "selected_tools": [],
}

THEMES = [
    "textual-dark",
    "nord",
    "gruvbox",
    "monokai",
    "tokyo-night",
    "dracula",
    "catppuccin-mocha",
    "solarized-dark",
    "solarized-light",
]


def load_config() -> dict[str, Any]:
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            return {
                **DEFAULT_CONFIG,
                **{k: data[k] for k in DEFAULT_CONFIG if k in data},
            }
        except (json.JSONDecodeError, OSError):
            pass
    save_config(dict(DEFAULT_CONFIG))
    return dict(DEFAULT_CONFIG)


def save_config(config: dict[str, Any]) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# --- Tools ------------------------------------------------------------------


def collect_tools(
    config: dict[str, Any], extra: Sequence[StreamingTool] = ()
) -> dict[str, StreamingTool]:
    if path := config.get("tools_path", "").strip():
        load_external_tools(path)
    tools: dict[str, StreamingTool] = {}
    for getter in (get_bash_tools, get_patch_file_tools):
        tools.update(getter())
    for t in ToolRegistry.get_instance().get_all():
        tools[t.name] = t
    for t in extra:
        tools.setdefault(t.name, t)
    return tools


# --- Streaming buffer -------------------------------------------------------


@dataclass
class StreamBuffer:
    """Holds the current streaming LLM output."""

    content: str = ""
    reasoning: str = ""
    tool_calls: dict[int, dict[str, str]] = field(default_factory=dict)
    _flushed: bool = False

    def render(self) -> Text:
        out = Text()
        if self.reasoning:
            out.append("▼ thinking\n", "dim italic #89b4fa")
            out.append(self.reasoning, "dim italic #89b4fa")
        if self.content:
            if self.reasoning:
                out.append("\n\n")
            out.append(self.content)
        return out

    def clear(self) -> None:
        self.content = ""
        self.reasoning = ""
        self.tool_calls.clear()
        self._flushed = False


# --- Modals -----------------------------------------------------------------


class ConfigScreen(ModalScreen[dict | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "Cancel")]

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = config

    def compose(self) -> ComposeResult:
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
                yield TextArea(self.cfg.get("system_prompt", ""), id="f-prompt")
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
                    "system_prompt": self.query_one("#f-prompt", TextArea).text,
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

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal small"):
            yield Label(self.t, classes="modal-title")
            yield Label(self.m, classes="modal-message")
            with Horizontal(classes="modal-buttons"):
                yield Button(self.ok_label, variant=self.variant, id="ok")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "ok")


class PromptScreen(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "Cancel")]

    def __init__(self, title: str, default: str = "") -> None:
        super().__init__()
        self.t, self.default = title, default

    def compose(self) -> ComposeResult:
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

    def compose(self) -> ComposeResult:
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


# --- Chat input -------------------------------------------------------------


class ChatInput(TextArea):
    def on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.app.action_submit()  # type: ignore[attr-defined]
        elif event.key in ("ctrl+enter", "ctrl+j"):
            event.stop()
            event.prevent_default()
            self.insert("\n")


# --- App --------------------------------------------------------------------


class TUIApp(App):
    TITLE = "Minimal Harness"
    ENABLE_COMMAND_PALETTE = False

    CSS = """
    Screen { align: center middle; background: $background; }

    #top-bar {
        height: 1; padding: 0 2; background: $primary 20%; color: $primary;
        text-style: bold;
    }

    #chat-container { height: 1fr; padding: 0 1; }

    #chat-log {
        height: 1fr; background: $background;
        border: none; padding: 1 2; scrollbar-size: 1 1;
    }

    #input-wrap {
        height: auto; max-height: 12; margin: 0 1 1 1;
        border: round $accent; padding: 0 1;
        background: $surface;
    }
    #input-wrap.streaming { border: round $warning; }

    #chat-input {
        height: auto; max-height: 10; background: $surface;
        border: none; padding: 0;
    }

    /* Modals ----------------------------------------------------------- */
    .modal {
        width: 72; max-height: 80%; padding: 1 2;
        background: $surface; border: round $accent;
    }
    .modal.small { width: 52; height: auto; }
    .modal-title { text-style: bold; color: $accent; margin-bottom: 1; width: 100%; text-align: center; }
    .modal-message { margin: 1 0 2 0; width: 100%; text-align: center; }
    .modal-body { height: auto; max-height: 24; padding-right: 1; }
    .modal-body Label { margin-top: 1; color: $text-muted; }
    .modal-body Input, .modal-body TextArea, .modal-body Select {
        margin-bottom: 1; background: $background; border: tall $surface-lighten-2;
    }
    .modal-body TextArea { height: 5; }
    .modal-body Checkbox { background: transparent; border: none; height: auto; padding: 0 1; }
    .tool-item { height: auto; margin-bottom: 1; }
    .tool-desc { color: $text-muted; text-style: italic; margin: 0 3; height: auto; }
    .modal-buttons { height: 3; align: center middle; margin-top: 1; }
    .modal-buttons Button { margin: 0 1; min-width: 12; }
    """

    BINDINGS = [
        Binding("ctrl+o", "config", "Config"),
        Binding("ctrl+t", "tools", "Tools"),
        Binding("ctrl+d", "dump", "Dump"),
        Binding("escape", "interrupt", "Interrupt", show=False),
        Binding("ctrl+c", "request_quit", "Quit"),
    ]

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        tools: Sequence[StreamingTool] | None = None,
    ) -> None:
        super().__init__()
        self.config = config or load_config()
        self._injected: tuple[StreamingTool, ...] = tuple(tools or ())
        self._all_tools: dict[str, StreamingTool] = {}
        self.active_tools: list[StreamingTool] = []
        self.memory: ConversationMemory | None = None
        self.client: FrameworkClient | None = None
        self.stop_event: asyncio.Event | None = None
        self.streaming = False
        self.buf = StreamBuffer()
        self._committed: list[Text] = []
        self._first = True

    # -- layout ------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Static(
            "  Minimal Harness  ·  Ctrl+O Config  ·  Ctrl+T Tools  ·  Esc Interrupt  ",
            id="top-bar",
        )
        with Vertical(id="chat-container"):
            yield RichLog(id="chat-log", markup=True, wrap=True, highlight=False)
        with Vertical(id="input-wrap"):
            yield ChatInput(
                id="chat-input",
                placeholder="Type a message — Enter to send, Ctrl+Enter for newline",
            )
        yield Footer()

    def on_mount(self) -> None:
        theme = self.config.get("theme", DEFAULT_CONFIG["theme"])
        if theme in THEMES:
            self.theme = theme
        self._rebuild()
        self.set_interval(FLUSH_INTERVAL, self._tick)
        self._input.focus()
        self._banner()

    # -- shortcuts --------------------------------------------------------
    @property
    def _rlog(self) -> RichLog:
        return self.query_one("#chat-log", RichLog)

    @property
    def _input(self) -> ChatInput:
        return self.query_one("#chat-input", ChatInput)

    @property
    def _wrap(self) -> Vertical:
        return self.query_one("#input-wrap", Vertical)

    # -- display ---------------------------------------------------------
    def say(self, text: str, style: str = "") -> None:
        t = Text(text, style=style) if style else Text(text)
        self._committed.append(t)
        if not self.streaming:
            self._rlog.write(t)

    def _tick(self) -> None:
        if not self.streaming:
            return
        self._rlog.clear()
        for line in self._committed:
            self._rlog.write(line)
        if self.buf.reasoning or self.buf.content:
            self._rlog.write(self.buf.render())
        self._rlog.scroll_end(animate=False)

    def _banner(self) -> None:
        self.say("Minimal Harness TUI", "bold #a6e3a1")
        self.say("")
        if not self.config.get("api_key"):
            self.say("⚠  No API key configured — press Ctrl+O", "bold #f9e2af")
        built_in = {
            n for getter in (get_bash_tools, get_patch_file_tools) for n in getter()
        }
        ext = [t for t in self._all_tools.values() if t.name not in built_in]
        if ext:
            self.say(
                f"Loaded {len(ext)} external tool(s): "
                + ", ".join(t.name for t in ext),
                "dim",
            )
        active = ", ".join(t.name for t in self.active_tools) or "(none)"
        self.say(f"Active tools: {active}", "dim")
        self.say("")

    # -- agent lifecycle --------------------------------------------------
    def _rebuild(self) -> None:
        cfg = self.config
        self._all_tools = collect_tools(cfg, self._injected)
        selected = cfg.get("selected_tools") or []
        if selected:
            self.active_tools = [
                self._all_tools[n] for n in selected if n in self._all_tools
            ]
        else:
            self.active_tools = list(self._all_tools.values())

        kwargs: dict[str, Any] = {}
        if cfg.get("base_url"):
            kwargs["base_url"] = cfg["base_url"]
        if cfg.get("api_key"):
            kwargs["api_key"] = cfg["api_key"]
        llm = OpenAILLMProvider(
            client=AsyncOpenAI(**kwargs), model=cfg.get("model", "")
        )

        prompt = cfg.get("system_prompt", DEFAULT_CONFIG["system_prompt"])
        if self.memory is None:
            self.memory = ConversationMemory(system_prompt=prompt)
        else:
            msgs = self.memory.get_all_messages()
            if (
                msgs
                and msgs[0].get("role") == "system"
                and msgs[0].get("content") != prompt
            ):
                self.memory = ConversationMemory(system_prompt=prompt)

        self.client = FrameworkClient(
            agent=OpenAIAgent(
                llm_provider=llm, tools=self.active_tools or None, memory=self.memory
            )
        )

    # -- input ------------------------------------------------------------
    def action_submit(self) -> None:
        text = self._input.text.strip()
        if not text or self.streaming:
            return
        self._input.text = ""
        if self._first:
            self._first = False
            self._committed.clear()
            self._rlog.clear()
            self.buf.clear()
        self.say(f"\n❯ {text}", "bold #89b4fa")
        self.say("")
        self._run(text)

    def _set_streaming(self, active: bool) -> None:
        self.streaming = active
        self._wrap.set_class(active, "streaming")
        self._input.disabled = active
        if not active:
            self._input.focus()

    @work(exclusive=True)
    async def _run(self, user_input: str) -> None:
        if self.client is None:
            self.say("Framework client not initialized.", "bold #f38ba8")
            return
        self.buf.clear()
        self.stop_event = asyncio.Event()
        self._set_streaming(True)
        try:
            async for event in self.client.run(
                user_input=[{"type": "text", "text": user_input}],
                stop_event=self.stop_event,
                memory=self.memory,
                tools=self.active_tools or None,
            ):
                if self.stop_event.is_set():
                    break
                self._on_event(event)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.say(f"\nError: {e}", "bold #f38ba8")
        finally:
            if not self.buf._flushed:
                rendered = self.buf.render()
                if rendered.plain:
                    self._committed.append(rendered)
            self.buf.clear()
            self._rlog.clear()
            for line in self._committed:
                self._rlog.write(line)
            self._rlog.scroll_end(animate=False)
            self.stop_event = None
            self._set_streaming(False)

    # -- event handling --------------------------------------------------
    def _on_event(self, event: Event) -> None:
        b = self.buf
        if isinstance(event, LLMChunkEvent):
            self._on_chunk(event)
        elif isinstance(event, LLMEndEvent):
            rendered = b.render()
            if rendered.plain:
                self._committed.append(rendered)
                b._flushed = True
            b.reasoning = ""
            b.content = ""
            if b.tool_calls:
                self.say("")
                self.say("")
                for _, call in sorted(b.tool_calls.items()):
                    try:
                        args = json.dumps(
                            json.loads(call.get("arguments", "{}")), ensure_ascii=False
                        )
                    except (json.JSONDecodeError, TypeError):
                        args = call.get("arguments", "")
                    self.say(f"  ▸ {call.get('name', '?')}({args})", "bold #f9e2af")
                b.tool_calls.clear()
            if event.usage:
                u = event.usage
                self.say(
                    f"  [{u['prompt_tokens']}+{u['completion_tokens']}={u['total_tokens']} tok]",
                    "dim",
                )
            self.say("")
            self.say("")
        elif isinstance(event, ExecutionStartEvent):
            self.say("")
            names = ", ".join(tc["function"]["name"] for tc in event.tool_calls)
            self.say(f"  ⚡ {names}", "bold #fab387")
        elif isinstance(event, ToolStartEvent):
            pass  # already announced
        elif isinstance(event, ToolProgressEvent):
            chunk = event.chunk
            msg = (
                chunk.get("message", str(chunk))
                if isinstance(chunk, dict)
                else str(chunk)
            )
            self.say(f"    · {msg}", "dim")
        elif isinstance(event, ToolEndEvent):
            r = event.result
            if isinstance(r, dict) and "error" in r:
                err_msg = r.get("error", "Unknown error")
                tb = r.get("traceback", "")
                stderr = r.get("stderr", "")
                full_err = err_msg
                if tb:
                    full_err += "\n\nTraceback:\n" + tb
                if stderr:
                    full_err += "\n\nStderr:\n" + stderr
                self.say(f"    ✗ {full_err}", "bold #f38ba8")
            else:
                s = (
                    json.dumps(r, ensure_ascii=False, default=str)
                    if isinstance(r, dict)
                    else str(r)
                )
                if len(s) > 500:
                    s = s[:500] + "…"
                self.say(f"    ✓ {s}", "#a6e3a1")
            self.say("")
        elif isinstance(event, AgentEndEvent):
            self.say("")

    def _on_chunk(self, event: LLMChunkEvent) -> None:
        b = self.buf
        try:
            delta = event.chunk.choices[0].delta
        except (AttributeError, IndexError):
            return
        reasoning = getattr(delta, "reasoning_content", None) or ""
        content = getattr(delta, "content", None) or getattr(delta, "text", None) or ""
        tcs = getattr(delta, "tool_calls", None) or []

        if reasoning:
            b.reasoning += reasoning

        if content:
            b.content += content

        for tc in tcs:
            call = b.tool_calls.setdefault(
                tc.index, {"id": "", "name": "", "arguments": ""}
            )
            if tc.id:
                call["id"] += tc.id
            if tc.function:
                if tc.function.name:
                    call["name"] += tc.function.name
                if tc.function.arguments:
                    call["arguments"] += tc.function.arguments

    # -- actions ---------------------------------------------------------
    def action_interrupt(self) -> None:
        if self.streaming and self.stop_event is not None:
            self.stop_event.set()
            self.say("  ✗ interrupted", "bold #f38ba8")

    def action_config(self) -> None:
        if self.streaming:
            return

        def done(result: dict | None) -> None:
            if result is None:
                return
            self.config.update(result)
            save_config(self.config)
            if (t := result.get("theme")) in THEMES:
                self.theme = t
            self._rebuild()
            self.say("✓ Configuration saved", "bold #a6e3a1")

        self.push_screen(ConfigScreen(self.config), done)

    def action_tools(self) -> None:
        if self.streaming or not self._all_tools:
            return
        selected = {t.name for t in self.active_tools}

        def done(chosen: list[str] | None) -> None:
            if chosen is None:
                return
            self.active_tools = [
                self._all_tools[n] for n in chosen if n in self._all_tools
            ]
            self.config["selected_tools"] = chosen
            save_config(self.config)
            self._rebuild()
            names = ", ".join(t.name for t in self.active_tools) or "(none)"
            self.say(f"✓ Tools: {names}", "bold #a6e3a1")

        self.push_screen(ToolSelectScreen(self._all_tools, selected), done)

    def action_dump(self) -> None:
        if self.memory is None:
            return

        def done(path: str | None) -> None:
            if not path:
                return
            try:
                p = Path(path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(
                    json.dumps(
                        {
                            "messages": self.memory.get_all_messages(),  # type: ignore[union-attr]
                            "usage": self.memory.get_total_usage(),  # type: ignore[union-attr]
                        },
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    ),
                    encoding="utf-8",
                )
                self.say(f"✓ Memory dumped → {path}", "bold #a6e3a1")
            except Exception as e:
                self.say(f"✗ {e}", "bold #f38ba8")

        self.push_screen(
            PromptScreen("💾  Dump memory to file", "./memory_dump.json"), done
        )

    def action_request_quit(self) -> None:
        def done(ok: bool) -> None:
            if ok:
                self.exit()

        self.push_screen(
            ConfirmScreen("Quit?", "Memory will be lost.", ok="Quit", variant="error"),
            done,
        )


# --- Entry ------------------------------------------------------------------


def main() -> None:
    config = load_config()
    TUIApp(config=config, tools=list(collect_tools(config).values())).run()


if __name__ == "__main__":
    main()
