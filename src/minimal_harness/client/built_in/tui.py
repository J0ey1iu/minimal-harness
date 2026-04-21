"""Terminal UI client for the minimal-harness framework."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from openai import AsyncOpenAI
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Footer, Input, RichLog, Static, TextArea

from minimal_harness.agent.openai import OpenAIAgent
from minimal_harness.client.client import FrameworkClient
from minimal_harness.client.events import (
    AgentEndEvent,
    AgentStartEvent,
    Event,
    ExecutionEndEvent,
    ExecutionStartEvent,
    LLMChunkEvent,
    LLMEndEvent,
    LLMStartEvent,
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

REFRESH_INTERVAL = 1 / 3


def get_all_built_in_tools() -> list[StreamingTool]:
    all_tools: dict[str, StreamingTool] = {}
    all_tools.update(get_bash_tools())
    all_tools.update(get_create_file_tools())
    all_tools.update(get_delete_file_tools())
    all_tools.update(get_patch_file_tools())
    all_tools.update(get_read_file_tools())
    return list(all_tools.values())


def load_all_tools(config: dict[str, str]) -> list[StreamingTool]:
    tools = get_all_built_in_tools()
    tools_path = config.get("tools_path", "")
    if tools_path:
        load_external_tools(tools_path)
    registry = ToolRegistry.get_instance()
    external = registry.get_all()
    ext_names = {t.name for t in external}
    tools = [t for t in tools if t.name not in ext_names]
    tools.extend(external)
    return tools


CONFIG_DIR = Path.home() / ".minimal_harness"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_CONFIG: dict[str, str] = {
    "base_url": "https://aihubmix.com/v1",
    "api_key": "put-your-api-key-here",
    "model": "qwen3.5-27b",
    "system_prompt": "You are a helpful assistant.",
    "tools_path": "",
}


def load_config(path: Path = CONFIG_FILE) -> dict[str, str]:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {
                **DEFAULT_CONFIG,
                **{k: v for k, v in data.items() if k in DEFAULT_CONFIG},
            }
        except (json.JSONDecodeError, OSError):
            pass
    save_config(dict(DEFAULT_CONFIG), path)
    return dict(DEFAULT_CONFIG)


def save_config(config: dict[str, str], path: Path = CONFIG_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


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


def extract_chunk_deltas(chunk: Any) -> tuple[str, str, list[Any]]:
    content_delta = ""
    reasoning_delta = ""
    tool_call_deltas: list[Any] = []

    if chunk is None:
        return content_delta, reasoning_delta, tool_call_deltas

    try:
        if chunk.choices:
            delta = chunk.choices[0].delta
            content_delta = getattr(delta, "content", None) or ""
            reasoning_delta = getattr(delta, "reasoning_content", None) or ""
            tool_call_deltas = getattr(delta, "tool_calls", None) or []
    except (AttributeError, IndexError):
        pass

    return content_delta, reasoning_delta, tool_call_deltas


@dataclass
class ToolCallAccumulator:
    id: str = ""
    name: str = ""
    arguments: str = ""


class ConfigScreen(ModalScreen[dict[str, str] | None]):
    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

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
            with Horizontal(id="config-buttons"):
                yield Button("Save", variant="primary", id="config-save")
                yield Button("Cancel", variant="default", id="config-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "config-save":
            config = {
                "base_url": self.query_one("#config-base-url", Input).value,
                "api_key": self.query_one("#config-api-key", Input).value,
                "model": self.query_one("#config-model", Input).value,
                "system_prompt": self.query_one("#config-system-prompt", TextArea).text,
                "tools_path": self.query_one("#config-tools-path", Input).value,
            }
            self.dismiss(config)
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class QuitConfirmScreen(ModalScreen[bool]):
    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

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
        if event.button.id == "quit-confirm":
            self.dismiss(True)
        else:
            self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)


class DumpMemoryScreen(ModalScreen[str | None]):
    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

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
            if path:
                self.dismiss(path)
            else:
                self.dismiss(None)
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ToolSelectScreen(ModalScreen[list[str] | None]):
    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

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
                label = f"{name}" if not desc else f"{name} — {desc}"
                yield Checkbox(
                    label=label, value=name in self.selected, id=f"tool-cb-{name}"
                )
            with Horizontal(id="tool-select-buttons"):
                yield Button("OK", variant="primary", id="tool-select-ok")
                yield Button("Cancel", variant="default", id="tool-select-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "tool-select-ok":
            selected: list[str] = []
            for name in self.tool_names:
                cb = self.query_one(f"#tool-cb-{name}", Checkbox)
                if cb.value:
                    selected.append(name)
            self.dismiss(selected if selected else None)
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class TUIApp(App):
    TITLE = "Minimal Harness TUI"
    ENABLE_COMMAND_PALETTE = False
    CSS = """
    #chat-container {
        height: 1fr;
    }
    #chat-log {
        height: 1fr;
        border: solid green;
        scrollbar-size: 0 0;
    }
    #input-bar {
        height: auto;
        padding: 0 1;
    }
    #chat-input {
        width: 1fr;
        height: auto;
        max-height: 10;
    }
    #streaming-label {
        color: yellow;
        text-style: italic;
    }
    #config-container, #dump-container, #tool-select-container, #quit-container {
        padding: 1 2;
        width: 60;
        height: auto;
        border: solid blue;
        background: $surface;
    }
    #config-title, #dump-title, #tool-select-title, #quit-title {
        text-style: bold;
        margin-bottom: 1;
    }
    #config-buttons, #dump-buttons, #tool-select-buttons, #quit-buttons {
        margin-top: 1;
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

    def __init__(
        self,
        config: dict[str, str] | None = None,
        tools: Sequence[StreamingTool] | None = None,
    ) -> None:
        super().__init__()
        self.config = config if config is not None else load_config()
        self._all_tools_map: dict[str, StreamingTool] = (
            {t.name: t for t in tools} if tools else {}
        )
        self.tools: list[StreamingTool] = list(tools) if tools else []
        self.framework_client: FrameworkClient | None = None
        self.memory: ConversationMemory | None = None
        self.stop_event: asyncio.Event | None = None
        self.is_streaming: bool = False

        self._streaming_content: str = ""
        self._streaming_reasoning: str = ""
        self._pending_lines: list[tuple[str, str]] = []
        self._tool_calls_acc: dict[int, ToolCallAccumulator] = {}
        self._is_reasoning: bool = False
        self._had_tool_calls: bool = False
        self._first_message: bool = True

    def compose(self) -> ComposeResult:
        with Vertical(id="chat-container"):
            yield RichLog(id="chat-log", highlight=True, markup=True, wrap=True)
            yield Static("", id="streaming-label")
        with Horizontal(id="input-bar"):
            yield TextArea(
                id="chat-input",
                placeholder="Type a message... (Ctrl+Enter or Ctrl+J to send)",
            )
        yield Footer()

    def on_mount(self) -> None:
        self._init_agent()
        self.set_interval(REFRESH_INTERVAL, self._refresh_display)
        self.query_one("#chat-input", TextArea).focus()
        if not self.config.get("api_key") and not self.config.get("base_url"):
            self.log_message(
                "No API key or base URL configured. Press Ctrl+O to configure.",
                style="bold yellow",
            )
        self._show_intro()

    def _show_intro(self) -> None:
        self.log_message("=== Minimal Harness TUI ===", style="bold green")
        self.log_message(
            "Type your message and press Enter to start a conversation.", style=""
        )
        self.log_message("", "")
        self.log_message("Keyboard shortcuts:", style="bold")
        self.log_message("  Ctrl+O / Cmd+O  - Open configuration", style="dim")
        self.log_message("  Ctrl+T / Cmd+T  - Select tools", style="dim")
        self.log_message("  Ctrl+D / Cmd+D  - Dump memory to file", style="dim")
        self.log_message("  Ctrl+C / Ctrl+Q / Cmd+Q  - Quit", style="dim")
        self.log_message("  Escape  - Interrupt current response", style="dim")
        self.log_message("", "")
        self.log_message("Getting started:", style="bold")
        self.log_message(
            "  1. Press Ctrl+O (Cmd+O on macOS) to configure your API key and base URL",
            style="dim",
        )
        self.log_message(
            "  2. Press Ctrl+T (Cmd+T on macOS) to enable tools (optional)", style="dim"
        )
        self.log_message("  3. Type a message and press Enter to start", style="dim")
        self.log_message("", "")
        self._show_tool_status()

    def _show_tool_status(self) -> None:
        built_in_names = {t.name for t in get_all_built_in_tools()}
        external_tools = [t for t in self.tools if t.name not in built_in_names]
        if external_tools:
            self.log_message("External tools registered:", style="bold")
            for t in external_tools:
                self.log_message(f"  - {t.name}", style="dim")
        elif self.config.get("tools_path", "").strip():
            self.log_message(
                "External tools path configured but no tools loaded.",
                style="bold yellow",
            )
        else:
            self.log_message(
                "No external tools path configured. Press Ctrl+O to set Tools Path.",
                style="dim",
            )

    def _reload_tools(self) -> None:
        """Reload external tools from disk and refresh the available tool maps."""
        tools_path = self.config.get("tools_path", "")
        if tools_path:
            load_external_tools(tools_path)

        # Rebuild the full available tool map from built-in + external registry.
        # self.tools holds the user's selected subset; _all_tools_map must
        # contain ALL tools so the selector modal can show them.
        all_tools = get_all_built_in_tools()
        external = ToolRegistry.get_instance().get_all()
        ext_names = {t.name for t in external}
        all_tools = [t for t in all_tools if t.name not in ext_names]
        all_tools.extend(external)

        # Merge tools that were passed in directly (not built-in, not external)
        existing_names = {t.name for t in all_tools}
        for tool in self.tools:
            if tool.name not in existing_names:
                all_tools.append(tool)
                existing_names.add(tool.name)

        self._all_tools_map = {t.name: t for t in all_tools}

        # Preserve current selection for tools that still exist
        selected_names = {t.name for t in self.tools}
        self.tools = [
            self._all_tools_map[name]
            for name in selected_names
            if name in self._all_tools_map
        ]

    def _init_agent(self) -> None:
        base_url = self.config.get("base_url") or None
        api_key = self.config.get("api_key") or None
        model = self.config.get("model", DEFAULT_CONFIG["model"])
        system_prompt = self.config.get(
            "system_prompt", DEFAULT_CONFIG["system_prompt"]
        )

        self._reload_tools()

        if api_key and base_url:
            client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        elif base_url:
            client = AsyncOpenAI(base_url=base_url)
        elif api_key:
            client = AsyncOpenAI(api_key=api_key)
        else:
            client = AsyncOpenAI()

        llm_provider = OpenAILLMProvider(client=client, model=model)

        # Preserve existing conversation memory when only tools or model change.
        # Only reset memory if this is the first init or the system prompt changed.
        if self.memory is None:
            self.memory = ConversationMemory(system_prompt=system_prompt)
        else:
            existing_messages = self.memory.get_all_messages()
            if (
                existing_messages
                and existing_messages[0].get("role") == "system"
                and existing_messages[0].get("content") != system_prompt
            ):
                self.memory = ConversationMemory(system_prompt=system_prompt)

        agent = OpenAIAgent(
            llm_provider=llm_provider,
            tools=self.tools if self.tools else None,
            memory=self.memory,
        )
        self.framework_client = FrameworkClient(agent=agent)

    @staticmethod
    def _extract_complete_lines(buf: str) -> tuple[list[str], str]:
        if "\n" not in buf:
            return [], buf
        last_nl = buf.rfind("\n")
        complete_portion = buf[: last_nl + 1]
        remainder = buf[last_nl + 1 :]
        lines = complete_portion.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        return lines, remainder

    def _queue_message(self, text: str, style: str = "") -> None:
        self._pending_lines.append((text, style))

    def _flush_streaming_to_pending(self) -> None:
        if self._streaming_reasoning:
            for line in self._streaming_reasoning.split("\n"):
                self._pending_lines.append((f"  {line}", "dim italic cyan"))
            self._streaming_reasoning = ""
        if self._streaming_content:
            for line in self._streaming_content.split("\n"):
                self._pending_lines.append((line, ""))
            self._streaming_content = ""

    def _refresh_display(self) -> None:
        chat_log = self.query_one("#chat-log", RichLog)

        for text, style in self._pending_lines:
            if style:
                chat_log.write(Text(text, style=style))
            else:
                chat_log.write(Text(text))
        self._pending_lines.clear()

        if self._streaming_reasoning:
            lines, self._streaming_reasoning = self._extract_complete_lines(
                self._streaming_reasoning
            )
            for line in lines:
                chat_log.write(Text(f"  {line}", style="dim italic cyan"))

        if self._streaming_content:
            lines, self._streaming_content = self._extract_complete_lines(
                self._streaming_content
            )
            for line in lines:
                chat_log.write(Text(line))

        if self.is_streaming:
            chat_log.scroll_end(animate=False)

    def log_message(self, text: str, style: str = "") -> None:
        self._queue_message(text, style)
        self._flush_streaming_to_pending()
        self._refresh_display()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        pass

    def on_key(self, event: events.Key) -> None:
        if event.key in ("ctrl+enter", "ctrl+j"):
            text_area = self.query_one("#chat-input", TextArea)
            if text_area.has_focus:
                self.action_submit_message()

    def action_submit_message(self) -> None:
        text_area = self.query_one("#chat-input", TextArea)
        text = text_area.text.strip()
        if not text:
            return
        if self.is_streaming:
            return
        text_area.text = ""
        if self._first_message:
            self._first_message = False
            self.query_one("#chat-log", RichLog).clear()
        self.log_message(f"\nYou: {text}", style="bold cyan")
        self._run_agent(text)

    @work(exclusive=True)
    async def _run_agent(self, user_input: str) -> None:
        if self.framework_client is None:
            self.log_message(
                "Error: Framework client not initialized.", style="bold red"
            )
            return

        self._reload_tools()

        self.is_streaming = True
        self.stop_event = asyncio.Event()
        streaming_label = self.query_one("#streaming-label", Static)
        input_widget = self.query_one("#chat-input", TextArea)
        input_widget.disabled = True

        self._streaming_content = ""
        self._streaming_reasoning = ""
        self._pending_lines = []
        self._tool_calls_acc = {}
        self._is_reasoning = False
        self._had_tool_calls = False

        streaming_label.update("Streaming...")

        try:
            async for event in self.framework_client.run(
                user_input=[{"type": "text", "text": user_input}],
                stop_event=self.stop_event,
                memory=self.memory,
                tools=self.tools if self.tools else None,
            ):
                if self.stop_event.is_set():
                    break
                self._handle_event(event)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._queue_message(f"\nError: {e}", "bold red")
        finally:
            self._flush_streaming_to_pending()
            self._refresh_display()
            self.is_streaming = False
            self.stop_event = None
            streaming_label.update("")
            self.query_one("#streaming-label", Static).update("")
            self.query_one("#chat-input", TextArea).disabled = False
            self.query_one("#chat-input", TextArea).focus()

    def _handle_event(self, event: Event) -> None:
        if isinstance(event, AgentStartEvent):
            pass

        elif isinstance(event, LLMStartEvent):
            pass

        elif isinstance(event, LLMChunkEvent):
            self._handle_chunk(event)

        elif isinstance(event, LLMEndEvent):
            self._handle_llm_end(event)

        elif isinstance(event, ExecutionStartEvent):
            names = [tc["function"]["name"] for tc in event.tool_calls]
            self._queue_message(f"\n  [Executing: {', '.join(names)}]", "bold yellow")

        elif isinstance(event, ToolStartEvent):
            tc = event.tool_call
            self._queue_message(f"  [Tool: {tc['function']['name']}]", "yellow")

        elif isinstance(event, ToolProgressEvent):
            chunk = event.chunk
            if isinstance(chunk, dict):
                msg = chunk.get("message", str(chunk))
            else:
                msg = str(chunk)
            self._queue_message(f"    {msg}", "dim")

        elif isinstance(event, ToolEndEvent):
            result = event.result
            if isinstance(result, dict):
                self._queue_message(
                    f"    Result: {json.dumps(result, ensure_ascii=False, default=str)}",
                    "green",
                )
            else:
                self._queue_message(f"    Result: {result}", "green")

        elif isinstance(event, ExecutionEndEvent):
            pass

        elif isinstance(event, AgentEndEvent):
            self._queue_message("", "")

    def _handle_chunk(self, event: LLMChunkEvent) -> None:
        chunk = event.chunk
        content_delta, reasoning_delta, tool_call_deltas = extract_chunk_deltas(chunk)

        if reasoning_delta:
            if not self._is_reasoning:
                self._is_reasoning = True
                if self._streaming_content:
                    for line in self._streaming_content.split("\n"):
                        self._pending_lines.append((line, ""))
                    self._streaming_content = ""
                self._queue_message("  [Thinking...]", "dim italic cyan")
            self._streaming_reasoning += reasoning_delta

        if content_delta:
            need_response_tag = False
            if self._had_tool_calls:
                self._had_tool_calls = False
                need_response_tag = True
            if self._is_reasoning:
                self._is_reasoning = False
                if self._streaming_reasoning:
                    for line in self._streaming_reasoning.split("\n"):
                        self._pending_lines.append((f"  {line}", "dim italic cyan"))
                    self._streaming_reasoning = ""
                need_response_tag = True
            if need_response_tag:
                self._queue_message("  [Response]", "bold")
            self._streaming_content += content_delta

        for tc_delta in tool_call_deltas:
            idx = tc_delta.index
            if idx not in self._tool_calls_acc:
                self._tool_calls_acc[idx] = ToolCallAccumulator()
            acc = self._tool_calls_acc[idx]
            if tc_delta.id:
                acc.id += tc_delta.id
            if tc_delta.function:
                if tc_delta.function.name:
                    acc.name += tc_delta.function.name
                if tc_delta.function.arguments:
                    acc.arguments += tc_delta.function.arguments

    def _handle_llm_end(self, event: LLMEndEvent) -> None:
        self._flush_streaming_to_pending()
        self._is_reasoning = False

        if self._tool_calls_acc:
            for _idx, acc in sorted(self._tool_calls_acc.items()):
                try:
                    args = json.loads(acc.arguments)
                    args_str = json.dumps(args, ensure_ascii=False)
                except (json.JSONDecodeError, TypeError):
                    args_str = acc.arguments
                self._queue_message(f"  [Call: {acc.name}({args_str})]", "bold yellow")
            self._tool_calls_acc = {}
            self._had_tool_calls = True

        if event.usage:
            self._queue_message(
                f"  [Tokens: prompt={event.usage['prompt_tokens']}, "
                f"completion={event.usage['completion_tokens']}, "
                f"total={event.usage['total_tokens']}]",
                "dim",
            )

        self._refresh_display()

    def key_escape(self) -> None:
        if self.is_streaming and self.stop_event is not None:
            self.stop_event.set()
            self._flush_streaming_to_pending()
            self._queue_message("\n  [Interrupted by user]", "bold red")
            self._refresh_display()
            self.is_streaming = False
            self.query_one("#streaming-label", Static).update("")
            self.query_one("#chat-input", TextArea).disabled = False
            self.query_one("#chat-input", TextArea).focus()

    def action_open_config(self) -> None:
        if self.is_streaming:
            self.log_message("Cannot configure while streaming.", style="bold yellow")
            return

        def on_config_result(config: dict[str, str] | None) -> None:
            if config is not None:
                save_config(config)
                self.config = config
                self._init_agent()
                self.log_message(
                    "Configuration saved and agent reinitialized.", style="bold green"
                )

        self.push_screen(ConfigScreen(self.config), on_config_result)

    def action_dump_memory(self) -> None:
        if self.memory is None:
            self.log_message("No memory available.", style="bold red")
            return

        def on_dump_result(path: str | None) -> None:
            if path is not None:
                try:
                    dump_memory(self.memory, path)  # type: ignore[arg-type]
                    self.log_message(f"Memory dumped to: {path}", style="bold green")
                except Exception as e:
                    self.log_message(f"Failed to dump memory: {e}", style="bold red")

        self.push_screen(DumpMemoryScreen(), on_dump_result)

    def action_select_tools(self) -> None:
        if self.is_streaming:
            self.log_message(
                "Cannot change tools while streaming.", style="bold yellow"
            )
            return

        if not self._all_tools_map:
            self.log_message("No tools available.", style="bold red")
            return

        tool_names = sorted(self._all_tools_map.keys())
        tool_descriptions = {
            name: t.description for name, t in self._all_tools_map.items()
        }
        selected = {t.name for t in self.tools}

        def on_tools_result(selected_names: list[str] | None) -> None:
            if selected_names is None:
                return
            self.tools = [
                self._all_tools_map[name]
                for name in selected_names
                if name in self._all_tools_map
            ]
            self._init_agent()
            names = ", ".join(t.name for t in self.tools) if self.tools else "(none)"
            self.log_message(f"Tools updated: {names}", style="bold green")

        self.push_screen(
            ToolSelectScreen(tool_names, tool_descriptions, selected),
            on_tools_result,
        )

    async def action_quit(self) -> None:
        async def on_quit_result(confirmed: bool | None) -> None:
            if confirmed:
                self.exit()

        self.push_screen(QuitConfirmScreen(), on_quit_result)


def main() -> None:
    config = load_config()
    tools = load_all_tools(config)
    app = TUIApp(config=config, tools=tools)
    app.run()


if __name__ == "__main__":
    main()
