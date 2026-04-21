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
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
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
from minimal_harness.memory import ConversationMemory, Memory
from minimal_harness.tool.base import StreamingTool
from minimal_harness.tool.built_in.bash import get_tools as get_bash_tools
from minimal_harness.tool.built_in.create_file import get_tools as get_create_file_tools
from minimal_harness.tool.built_in.delete_file import get_tools as get_delete_file_tools
from minimal_harness.tool.built_in.patch_file import get_tools as get_patch_file_tools
from minimal_harness.tool.built_in.read_file import get_tools as get_read_file_tools
from minimal_harness.tool.external_loader import load_external_tools
from minimal_harness.tool.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REFRESH_INTERVAL = 1 / 3

CONFIG_DIR = Path.home() / ".minimal_harness"
CONFIG_FILE = CONFIG_DIR / "config.json"

CONFIG_KEYS: tuple[str, ...] = (
    "base_url",
    "api_key",
    "model",
    "system_prompt",
    "tools_path",
    "theme",
)

DEFAULT_CONFIG: dict[str, str] = {
    "base_url": "https://aihubmix.com/v1",
    "api_key": "put-your-api-key-here",
    "model": "qwen3.5-27b",
    "system_prompt": "You are a helpful assistant.",
    "tools_path": "",
    "theme": "textual-dark",
}

AVAILABLE_THEMES: list[str] = [
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

# ---------------------------------------------------------------------------
# Tool helpers
# ---------------------------------------------------------------------------

_BUILT_IN_TOOL_GETTERS = (
    get_bash_tools,
    get_create_file_tools,
    get_delete_file_tools,
    get_patch_file_tools,
    get_read_file_tools,
)


def get_all_built_in_tools() -> list[StreamingTool]:
    """Return a deduplicated list of every built-in tool."""
    merged: dict[str, StreamingTool] = {}
    for getter in _BUILT_IN_TOOL_GETTERS:
        merged.update(getter())
    return list(merged.values())


def _collect_all_tools(
    config: dict[str, str],
    extra: Sequence[StreamingTool] = (),
) -> dict[str, StreamingTool]:
    """Build the full name→tool map from built-ins, external registry, and any
    extra tools passed in programmatically.  External tools override built-ins
    with the same name; *extra* tools are added only when no other tool already
    claims the name."""
    tools_path = config.get("tools_path", "")
    if tools_path:
        load_external_tools(tools_path)

    by_name: dict[str, StreamingTool] = {t.name: t for t in get_all_built_in_tools()}

    for t in ToolRegistry.get_instance().get_all():
        by_name[t.name] = t  # external overrides built-in

    for t in extra:
        by_name.setdefault(t.name, t)  # extras fill gaps only

    return by_name


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------


def load_config(path: Path = CONFIG_FILE) -> dict[str, str]:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            merged = {
                **DEFAULT_CONFIG,
                **{k: v for k, v in data.items() if k in DEFAULT_CONFIG},
            }
            return merged
        except (json.JSONDecodeError, OSError):
            pass
    save_config(dict(DEFAULT_CONFIG), path)
    return dict(DEFAULT_CONFIG)


def save_config(config: dict[str, str], path: Path = CONFIG_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Memory dump
# ---------------------------------------------------------------------------


def dump_memory(memory: Memory, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "messages": memory.get_all_messages(),
        "usage": memory.get_total_usage(),
    }
    target.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Chunk parsing
# ---------------------------------------------------------------------------


def extract_chunk_deltas(chunk: Any) -> tuple[str, str, list[Any]]:
    """Return ``(content_delta, reasoning_delta, tool_call_deltas)``."""
    if chunk is None:
        return "", "", []
    try:
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None) or ""
        reasoning = getattr(delta, "reasoning_content", None) or ""
        tool_calls = getattr(delta, "tool_calls", None) or []
        return content, reasoning, tool_calls
    except (AttributeError, IndexError):
        return "", "", []


# ---------------------------------------------------------------------------
# Streaming-state accumulator
# ---------------------------------------------------------------------------


@dataclass
class ToolCallAccumulator:
    id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class StreamingState:
    """All mutable state that lives only for a single agent run."""

    content_buf: str = ""
    reasoning_buf: str = ""
    pending_lines: list[tuple[str, str]] = field(default_factory=list)
    tool_calls_acc: dict[int, ToolCallAccumulator] = field(default_factory=dict)
    is_reasoning: bool = False
    had_tool_calls: bool = False

    # ---- helpers ----------------------------------------------------------

    def queue(self, text: str, style: str = "") -> None:
        self.pending_lines.append((text, style))

    def flush_buffers(self) -> None:
        """Move whatever is left in the streaming buffers into *pending_lines*."""
        if self.reasoning_buf:
            for line in self.reasoning_buf.split("\n"):
                self.pending_lines.append((f"  {line}", "dim italic cyan"))
            self.reasoning_buf = ""
        if self.content_buf:
            for line in self.content_buf.split("\n"):
                self.pending_lines.append((line, ""))
            self.content_buf = ""

    @staticmethod
    def extract_complete_lines(buf: str) -> tuple[list[str], str]:
        """Split *buf* into fully-received lines and a remaining fragment."""
        if "\n" not in buf:
            return [], buf
        last_nl = buf.rfind("\n")
        lines = buf[: last_nl + 1].split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        return lines, buf[last_nl + 1 :]

    def drain_complete_lines(self) -> None:
        """Move complete lines from the streaming buffers into *pending_lines*."""
        if self.reasoning_buf:
            lines, self.reasoning_buf = self.extract_complete_lines(self.reasoning_buf)
            for line in lines:
                self.pending_lines.append((f"  {line}", "dim italic cyan"))
        if self.content_buf:
            lines, self.content_buf = self.extract_complete_lines(self.content_buf)
            for line in lines:
                self.pending_lines.append((line, ""))


# ---------------------------------------------------------------------------
# Modal screens
# ---------------------------------------------------------------------------


class _DismissableModal(ModalScreen):
    """Shared base: Escape always dismisses with ``None``."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def action_cancel(self) -> None:
        self.dismiss(None)


class ConfigScreen(_DismissableModal):
    """Edit base_url / api_key / model / system_prompt / tools_path / theme."""

    def __init__(self, current_config: dict[str, str]) -> None:
        super().__init__()
        self.current_config = current_config

    def compose(self) -> ComposeResult:
        with Vertical(id="config-container"):
            yield Static("Configuration", id="config-title")
            yield Static("Base URL:")
            yield Input(
                value=self.current_config.get("base_url", ""),
                placeholder="https://api.openai.com/v1",
                id="config-base-url",
            )
            yield Static("API Key:")
            yield Input(
                value=self.current_config.get("api_key", ""),
                placeholder="sk-...",
                id="config-api-key",
                password=True,
            )
            yield Static("Model:")
            yield Input(
                value=self.current_config.get("model", ""),
                placeholder="gpt-4o",
                id="config-model",
            )
            yield Static("System Prompt:")
            yield TextArea(
                self.current_config.get("system_prompt", ""),
                id="config-system-prompt",
                placeholder="You are a helpful assistant.",
            )
            yield Static("Tools Path:")
            yield Input(
                value=self.current_config.get("tools_path", ""),
                placeholder="~/.minimal_harness/tools/ or path to .py file",
                id="config-tools-path",
            )
            yield Static("Theme:")
            yield Select(
                [(theme, theme) for theme in AVAILABLE_THEMES],
                value=self.current_config.get("theme", DEFAULT_CONFIG["theme"]),
                id="config-theme",
                allow_blank=False,
            )
            with Horizontal(id="config-buttons"):
                yield Button("Save", variant="primary", id="config-save")
                yield Button("Cancel", variant="default", id="config-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "config-save":
            theme_select = self.query_one("#config-theme", Select)
            theme_value = theme_select.value
            self.dismiss(
                {
                    "base_url": self.query_one("#config-base-url", Input).value,
                    "api_key": self.query_one("#config-api-key", Input).value,
                    "model": self.query_one("#config-model", Input).value,
                    "system_prompt": self.query_one(
                        "#config-system-prompt", TextArea
                    ).text,
                    "tools_path": self.query_one("#config-tools-path", Input).value,
                    "theme": theme_value if isinstance(theme_value, str) else DEFAULT_CONFIG["theme"],
                }
            )
        else:
            self.dismiss(None)


class QuitConfirmScreen(_DismissableModal):
    def compose(self) -> ComposeResult:
        with Vertical(id="quit-container"):
            yield Static("Quit Minimal Harness?", id="quit-title")
            yield Static(
                "Warning: Memory will be lost and cannot be recovered.",
                id="quit-warning",
            )
            with Horizontal(id="quit-buttons"):
                yield Button("Quit", variant="error", id="quit-confirm")
                yield Button("Cancel", variant="default", id="quit-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "quit-confirm")


class DumpMemoryScreen(_DismissableModal):
    def compose(self) -> ComposeResult:
        with Vertical(id="dump-container"):
            yield Static("Dump Memory", id="dump-title")
            yield Static("File path:")
            yield Input(
                value="./memory_dump.json",
                placeholder="Enter file path...",
                id="dump-path",
            )
            with Horizontal(id="dump-buttons"):
                yield Button("Save", variant="primary", id="dump-save")
                yield Button("Cancel", variant="default", id="dump-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "dump-save":
            path = self.query_one("#dump-path", Input).value.strip()
            self.dismiss(path or None)
        else:
            self.dismiss(None)


class ToolSelectScreen(_DismissableModal):
    def __init__(
        self,
        tool_names: list[str],
        tool_descriptions: dict[str, str],
        selected: set[str],
    ) -> None:
        super().__init__()
        self.tool_names = tool_names
        self.tool_descriptions = tool_descriptions
        self.selected = selected

    def compose(self) -> ComposeResult:
        with Vertical(id="tool-select-container"):
            yield Static("Select Tools", id="tool-select-title")
            for name in self.tool_names:
                desc = self.tool_descriptions.get(name, "")
                with Horizontal(classes="tool-row"):
                    yield Checkbox(
                        value=name in self.selected, id=f"tool-cb-{name}"
                    )
                    with Vertical(classes="tool-info"):
                        name_static = Static(name, classes="tool-name")
                        name_static.shrink = False
                        yield name_static
                        if desc:
                            desc_static = Static(desc, classes="tool-desc")
                            desc_static.shrink = False
                            yield desc_static
            with Horizontal(id="tool-select-buttons"):
                yield Button("OK", variant="primary", id="tool-select-ok")
                yield Button("Cancel", variant="default", id="tool-select-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "tool-select-ok":
            chosen = [
                n
                for n in self.tool_names
                if self.query_one(f"#tool-cb-{n}", Checkbox).value
            ]
            self.dismiss(chosen or None)
        else:
            self.dismiss(None)


# ---------------------------------------------------------------------------
# Custom widgets
# ---------------------------------------------------------------------------


class ChatInput(TextArea):
    """TextArea that submits on Enter and inserts newline on Ctrl+Enter / Ctrl+J."""

    def on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.app.action_submit_message()  # type: ignore[attr-defined]
            return
        if event.key in ("ctrl+enter", "ctrl+j"):
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return


# ---------------------------------------------------------------------------
# Main TUI application
# ---------------------------------------------------------------------------


class TUIApp(App):
    TITLE = "Minimal Harness TUI"
    ENABLE_COMMAND_PALETTE = False
    CSS = """
    /* Global screen alignment for modals */
    Screen { align: center middle; }

    /* Main chat layout */
    #chat-container { height: 1fr; layout: vertical; }

    #chat-log {
        height: 1fr;
        border: none;
        scrollbar-size: 1 1;
        padding: 0 2;
    }

    /* Input bar at the bottom */
    #input-bar {
        height: auto;
        padding: 1 2;
        background: $surface-darken-1;
        border-top: solid $surface-lighten-1;
    }

    #chat-input {
        width: 1fr;
        height: auto;
        max-height: 10;
        border: solid $surface-lighten-2;
        background: $surface;
        padding: 0 1;
        color: $text;
    }

    /* Streaming status label */
    #streaming-label {
        color: $warning;
        text-style: italic;
        height: auto;
        padding: 0 2;
        background: $surface-darken-1;
    }

    /* Modal screens */
    #config-container, #dump-container, #tool-select-container, #quit-container {
        padding: 1 2;
        width: 70;
        height: auto;
        border: round $primary;
        background: $surface;
    }

    #config-title, #dump-title, #tool-select-title, #quit-title {
        text-style: bold;
        margin-bottom: 1;
        color: $primary;
        text-align: center;
    }

    #config-buttons, #dump-buttons, #tool-select-buttons, #quit-buttons {
        margin-top: 1;
        align: center middle;
        height: auto;
    }

    #quit-warning {
        color: $warning;
        text-align: center;
        margin: 1 0;
        text-style: italic;
    }

    /* Form elements in modals */
    #config-container Input, #config-container TextArea, #dump-container Input {
        margin-bottom: 1;
        border: solid $surface-lighten-1;
    }

    #config-container Static {
        margin-top: 1;
        color: $text;
        text-style: dim;
    }

    /* Tool selection rows */
    .tool-row {
        height: auto;
        margin-bottom: 1;
    }
    .tool-info {
        width: 1fr;
        height: auto;
        padding-left: 1;
    }
    .tool-name {
        width: 1fr;
        height: auto;
        text-style: bold;
    }
    .tool-desc {
        width: 1fr;
        height: auto;
        text-style: dim;
    }
    """

    BINDINGS = [
        Binding("ctrl+o", "open_config", "Config", show=True),
        Binding("cmd+o", "open_config", "", show=False),
        Binding("ctrl+t", "select_tools", "Tools", show=True),
        Binding("cmd+t", "select_tools", "", show=False),
        Binding("ctrl+d", "dump_memory", "Dump Memory", show=True),
        Binding("cmd+d", "dump_memory", "", show=False),
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+q", "quit", "", show=False),
        Binding("cmd+q", "quit", "", show=False),
    ]

    # ------------------------------------------------------------------ init

    def __init__(
        self,
        config: dict[str, str] | None = None,
        tools: Sequence[StreamingTool] | None = None,
    ) -> None:
        super().__init__()
        self.config = config if config is not None else load_config()

        # Tools injected from outside (e.g. programmatic use)
        self._injected_tools: tuple[StreamingTool, ...] = tuple(tools) if tools else ()
        # Full universe of available tools (built-in + external + injected)
        self._all_tools_map: dict[str, StreamingTool] = {}
        # User's currently selected subset
        self.tools: list[StreamingTool] = []

        self.framework_client: FrameworkClient | None = None
        self.memory: ConversationMemory | None = None
        self.stop_event: asyncio.Event | None = None
        self.is_streaming: bool = False

        self._stream: StreamingState = StreamingState()
        self._first_message: bool = True

    # ------------------------------------------------------------ lifecycle

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="chat-container"):
            yield RichLog(id="chat-log", highlight=True, markup=True, wrap=True)
            yield Static("", id="streaming-label")
        with Horizontal(id="input-bar"):
            yield ChatInput(
                id="chat-input",
                placeholder="Type a message... (Enter to send, Ctrl+Enter for newline)",
            )
        yield Footer()

    def on_mount(self) -> None:
        theme = self.config.get("theme", DEFAULT_CONFIG["theme"])
        if theme in AVAILABLE_THEMES:
            self.theme = theme
        self._rebuild_agent()
        self.set_interval(REFRESH_INTERVAL, self._refresh_display)
        self._chat_input.focus()
        if not self.config.get("api_key") and not self.config.get("base_url"):
            self._log(
                "No API key or base URL configured. Press Ctrl+O to configure.",
                "bold bright_yellow",
            )
        self._show_intro()

    # ------------------------------------------------ cached widget lookups

    @property
    def _chat_log(self) -> RichLog:
        return self.query_one("#chat-log", RichLog)

    @property
    def _chat_input(self) -> TextArea:
        return self.query_one("#chat-input", TextArea)

    @property
    def _streaming_label(self) -> Static:
        return self.query_one("#streaming-label", Static)

    # --------------------------------------------------------- intro / info

    def _show_intro(self) -> None:
        lines: list[tuple[str, str]] = [
            ("=== Minimal Harness TUI ===", "bold bright_green"),
            ("Type your message and press Enter to send.", ""),
            ("", ""),
            ("Keyboard shortcuts:", "bold"),
            ("  Enter  — Send message", "dim"),
            ("  Ctrl+Enter / Ctrl+J  — Insert newline", "dim"),
            ("  Ctrl+O / Cmd+O  — Open configuration", "dim"),
            ("  Ctrl+T / Cmd+T  — Select tools", "dim"),
            ("  Ctrl+D / Cmd+D  — Dump memory to file", "dim"),
            ("  Ctrl+C / Ctrl+Q / Cmd+Q  — Quit", "dim"),
            ("  Escape  — Interrupt current response", "dim"),
            ("", ""),
            ("Getting started:", "bold"),
            (
                "  1. Press Ctrl+O (Cmd+O on macOS) to configure your API key and base URL",
                "dim",
            ),
            ("  2. Press Ctrl+T (Cmd+T on macOS) to enable tools (optional)", "dim"),
            ("  3. Type a message and press Enter to start", "dim"),
            ("", ""),
        ]
        for text, style in lines:
            self._log(text, style)
        self._show_tool_status()

    def _show_tool_status(self) -> None:
        built_in_names = {t.name for t in get_all_built_in_tools()}
        external_tools = [t for t in self.tools if t.name not in built_in_names]
        if external_tools:
            self._log("External tools registered:", "bold")
            for t in external_tools:
                self._log(f"  - {t.name}", "dim")
        elif self.config.get("tools_path", "").strip():
            self._log(
                "External tools path configured but no tools loaded.", "bold bright_yellow"
            )
        else:
            self._log(
                "No external tools path configured. Press Ctrl+O to set Tools Path.",
                "dim",
            )

    # ---------------------------------------------------- tool / agent init

    def _reload_tools(self) -> None:
        """Refresh ``_all_tools_map`` and prune ``self.tools`` to still-valid entries."""
        self._all_tools_map = _collect_all_tools(self.config, self._injected_tools)
        self.tools = [
            self._all_tools_map[n]
            for n in (t.name for t in self.tools)
            if n in self._all_tools_map
        ]

    def _rebuild_agent(self) -> None:
        """(Re)create LLM provider, agent, and framework client.

        Preserves conversation memory unless the system prompt changed.
        """
        cfg = self.config
        base_url = cfg.get("base_url") or None
        api_key = cfg.get("api_key") or None
        model = cfg.get("model", DEFAULT_CONFIG["model"])
        system_prompt = cfg.get("system_prompt", DEFAULT_CONFIG["system_prompt"])

        self._reload_tools()

        # Build AsyncOpenAI client — only pass params that are set
        client_kwargs: dict[str, Any] = {}
        if base_url:
            client_kwargs["base_url"] = base_url
        if api_key:
            client_kwargs["api_key"] = api_key
        client = AsyncOpenAI(**client_kwargs)

        llm_provider = OpenAILLMProvider(client=client, model=model)

        # Preserve memory across config changes unless system prompt differs
        if self.memory is not None:
            msgs = self.memory.get_all_messages()
            if (
                msgs
                and msgs[0].get("role") == "system"
                and msgs[0].get("content") != system_prompt
            ):
                self.memory = ConversationMemory(system_prompt=system_prompt)
        else:
            self.memory = ConversationMemory(system_prompt=system_prompt)

        agent = OpenAIAgent(
            llm_provider=llm_provider,
            tools=self.tools or None,
            memory=self.memory,
        )
        self.framework_client = FrameworkClient(agent=agent)

    # -------------------------------------------------------- display layer

    def _log(self, text: str, style: str = "") -> None:
        """Immediately write a line to the chat log."""
        self._stream.queue(text, style)
        self._stream.flush_buffers()
        self._write_pending()

    def _write_pending(self) -> None:
        """Flush ``_stream.pending_lines`` into the RichLog widget."""
        log = self._chat_log
        for text, style in self._stream.pending_lines:
            log.write(Text(text, style=style) if style else Text(text))
        self._stream.pending_lines.clear()

    def _refresh_display(self) -> None:
        """Timer callback: push completed lines and auto-scroll while streaming."""
        self._stream.drain_complete_lines()
        self._write_pending()
        if self.is_streaming:
            self._chat_log.scroll_end(animate=False)

    # ----------------------------------------------------- input handling

    def on_key(self, event: events.Key) -> None:
        # ChatInput handles Enter / Ctrl+Enter locally; App only needs to
        # guard against global shortcuts that should not trigger while typing.
        pass

    def action_submit_message(self) -> None:
        text_area = self._chat_input
        text = text_area.text.strip()
        if not text or self.is_streaming:
            return
        text_area.text = ""
        if self._first_message:
            self._first_message = False
            self._chat_log.clear()
        self._log(f"\nYou: {text}", "bold cyan")
        self._run_agent(text)

    # ----------------------------------------------------- agent execution

    def _set_streaming(self, active: bool) -> None:
        """Toggle UI elements tied to the streaming state."""
        self.is_streaming = active
        self._streaming_label.update("Streaming..." if active else "")
        self._chat_input.disabled = active
        if not active:
            self._chat_input.focus()

    @work(exclusive=True)
    async def _run_agent(self, user_input: str) -> None:
        if self.framework_client is None:
            self._log("Error: Framework client not initialized.", "bold bright_red")
            return

        self._reload_tools()
        self._stream = StreamingState()
        self.stop_event = asyncio.Event()
        self._set_streaming(True)

        try:
            async for event in self.framework_client.run(
                user_input=[{"type": "text", "text": user_input}],
                stop_event=self.stop_event,
                memory=self.memory,
                tools=self.tools or None,
            ):
                if self.stop_event.is_set():
                    break
                self._handle_event(event)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._stream.queue(f"\nError: {e}", "bold bright_red")
        finally:
            self._stream.flush_buffers()
            self._write_pending()
            self.stop_event = None
            self._set_streaming(False)

    # ---------------------------------------------------- event dispatching

    def _handle_event(self, event: Event) -> None:
        s = self._stream

        if isinstance(event, LLMChunkEvent):
            self._handle_chunk(event)

        elif isinstance(event, LLMEndEvent):
            self._handle_llm_end(event)

        elif isinstance(event, ExecutionStartEvent):
            names = [tc["function"]["name"] for tc in event.tool_calls]
            s.queue(f"\n  [Executing: {', '.join(names)}]", "bold bright_yellow")

        elif isinstance(event, ToolStartEvent):
            s.queue(f"  [Tool: {event.tool_call['function']['name']}]", "bright_yellow")

        elif isinstance(event, ToolProgressEvent):
            chunk = event.chunk
            msg = (
                chunk.get("message", str(chunk))
                if isinstance(chunk, dict)
                else str(chunk)
            )
            s.queue(f"    {msg}", "dim")

        elif isinstance(event, ToolEndEvent):
            result = event.result
            if isinstance(result, dict):
                formatted = json.dumps(result, ensure_ascii=False, default=str)
            else:
                formatted = str(result)
            s.queue(f"    Result: {formatted}", "bright_green")

        elif isinstance(event, AgentEndEvent):
            s.queue("", "")

        # AgentStartEvent, LLMStartEvent, ExecutionEndEvent — intentionally ignored

    def _handle_chunk(self, event: LLMChunkEvent) -> None:
        s = self._stream
        content_delta, reasoning_delta, tool_call_deltas = extract_chunk_deltas(
            event.chunk
        )

        if reasoning_delta:
            if not s.is_reasoning:
                s.is_reasoning = True
                # flush any partial content before switching to reasoning
                if s.content_buf:
                    for line in s.content_buf.split("\n"):
                        s.pending_lines.append((line, ""))
                    s.content_buf = ""
                s.queue("  [Thinking...]", "dim italic cyan")
            s.reasoning_buf += reasoning_delta

        if content_delta:
            need_tag = False
            if s.had_tool_calls:
                s.had_tool_calls = False
                need_tag = True
            if s.is_reasoning:
                s.is_reasoning = False
                if s.reasoning_buf:
                    for line in s.reasoning_buf.split("\n"):
                        s.pending_lines.append((f"  {line}", "dim italic cyan"))
                    s.reasoning_buf = ""
                need_tag = True
            if need_tag:
                s.queue("  [Response]", "bold")
            s.content_buf += content_delta

        for tc_delta in tool_call_deltas:
            idx = tc_delta.index
            acc = s.tool_calls_acc.setdefault(idx, ToolCallAccumulator())
            if tc_delta.id:
                acc.id += tc_delta.id
            if tc_delta.function:
                if tc_delta.function.name:
                    acc.name += tc_delta.function.name
                if tc_delta.function.arguments:
                    acc.arguments += tc_delta.function.arguments

    def _handle_llm_end(self, event: LLMEndEvent) -> None:
        s = self._stream
        s.flush_buffers()
        s.is_reasoning = False

        if s.tool_calls_acc:
            for _idx, acc in sorted(s.tool_calls_acc.items()):
                try:
                    args_str = json.dumps(json.loads(acc.arguments), ensure_ascii=False)
                except (json.JSONDecodeError, TypeError):
                    args_str = acc.arguments
                s.queue(f"  [Call: {acc.name}({args_str})]", "bold bright_yellow")
            s.tool_calls_acc.clear()
            s.had_tool_calls = True

        if event.usage:
            s.queue(
                f"  [Tokens: prompt={event.usage['prompt_tokens']}, "
                f"completion={event.usage['completion_tokens']}, "
                f"total={event.usage['total_tokens']}]",
                "dim",
            )
        self._write_pending()

    # ------------------------------------------------------------ key: Esc

    def key_escape(self) -> None:
        if self.is_streaming and self.stop_event is not None:
            self.stop_event.set()
            self._stream.flush_buffers()
            self._stream.queue("\n  [Interrupted by user]", "bold bright_red")
            self._write_pending()
            self.stop_event = None
            self._set_streaming(False)

    # ----------------------------------------------------- action: config

    def action_open_config(self) -> None:
        if self.is_streaming:
            self._log("Cannot configure while streaming.", "bold bright_yellow")
            return

        def on_result(config: dict[str, str] | None) -> None:
            if config is not None:
                save_config(config)
                self.config = config
                new_theme = config.get("theme", DEFAULT_CONFIG["theme"])
                if new_theme in AVAILABLE_THEMES:
                    self.theme = new_theme
                self._rebuild_agent()
                self._log("Configuration saved and agent reinitialized.", "bold bright_green")

        self.push_screen(ConfigScreen(self.config), on_result)

    # ------------------------------------------------- action: dump memory

    def action_dump_memory(self) -> None:
        if self.memory is None:
            self._log("No memory available.", "bold bright_red")
            return

        def on_result(path: str | None) -> None:
            if path is not None:
                try:
                    dump_memory(self.memory, path)  # type: ignore[arg-type]
                    self._log(f"Memory dumped to: {path}", "bold bright_green")
                except Exception as e:
                    self._log(f"Failed to dump memory: {e}", "bold bright_red")

        self.push_screen(DumpMemoryScreen(), on_result)

    # ------------------------------------------------- action: select tools

    def action_select_tools(self) -> None:
        if self.is_streaming:
            self._log("Cannot change tools while streaming.", "bold bright_yellow")
            return
        if not self._all_tools_map:
            self._log("No tools available.", "bold bright_red")
            return

        tool_names = sorted(self._all_tools_map)
        tool_descs = {n: t.description for n, t in self._all_tools_map.items()}
        selected = {t.name for t in self.tools}

        def on_result(chosen: list[str] | None) -> None:
            if chosen is None:
                return
            self.tools = [
                self._all_tools_map[n] for n in chosen if n in self._all_tools_map
            ]
            self._rebuild_agent()
            names = ", ".join(t.name for t in self.tools) or "(none)"
            self._log(f"Tools updated: {names}", "bold bright_green")

        self.push_screen(ToolSelectScreen(tool_names, tool_descs, selected), on_result)

    # -------------------------------------------------------- action: quit

    async def action_quit(self) -> None:
        def on_result(confirmed: bool | None) -> None:
            if confirmed:
                self.exit()

        self.push_screen(QuitConfirmScreen(), on_result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    config = load_config()
    all_tools = list(_collect_all_tools(config).values())
    app = TUIApp(config=config, tools=all_tools)
    app.run()


if __name__ == "__main__":
    main()
