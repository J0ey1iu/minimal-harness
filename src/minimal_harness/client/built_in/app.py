"""Main TUI application."""

from __future__ import annotations

import asyncio
import json
import random
from io import StringIO
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, Label, ListItem, ListView, RichLog, Static

from minimal_harness.agent.openai import OpenAIAgent
from minimal_harness.client.built_in.buffer import StreamBuffer
from minimal_harness.client.built_in.config import (
    DEFAULT_CONFIG,
    J0EY1IU_QUOTES,
    THEMES,
    collect_tools,
    load_config,
    read_system_prompt,
    save_config,
)
from minimal_harness.client.built_in.modals import (
    ConfigScreen,
    ConfirmScreen,
    PromptScreen,
    ToolSelectScreen,
)
from minimal_harness.client.built_in.widgets import (
    ChatInput,
    DumpRequest,
    SlashCommandHide,
    SlashCommandNavigateDown,
    SlashCommandNavigateUp,
    SlashCommandSelect,
    SlashCommandShow,
)
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
from minimal_harness.tool.registry import ToolRegistry

FLUSH_INTERVAL = 0.25
MAX_DISPLAY_LENGTH = 500


class TUIApp(App):
    TITLE = "Minimal Harness"
    ENABLE_COMMAND_PALETTE = False

    CSS = """
    Screen { align: center middle; background: $background; }

    #top-bar {
        height: 1; padding: 0 2; background: $primary 20%; color: $primary;
        text-style: bold;
    }

    #chat-container { height: 1fr; width: 100%; padding: 0 1; }

    #chat-log {
        height: 1fr; width: 100%; background: $background;
        border: none; padding: 1 2; scrollbar-size: 1 1;
    }

    #input-area {
        height: auto; width: 100%; margin: 0 1 1 1;
    }

    #suggestion-list {
        height: auto; max-height: 8; width: 100%;
        background: $surface; border: round $accent;
        margin-bottom: 1;
        display: none;
    }
    #suggestion-list.visible { display: block; }

    #input-wrap {
        height: auto; max-height: 12;
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
        Binding("escape", "interrupt", "Interrupt", show=False),
        Binding("ctrl+c", "request_quit", "Quit"),
    ]

    SLASH_COMMANDS: list[tuple[str, str, str]] = [
        ("/config", "Open configuration", "config"),
        ("/tools", "Select tools", "tools"),
        ("/new", "Start new conversation", "new"),
        ("/share", "Export chat as SVG", "share"),
    ]

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        super().__init__()
        self.config = config or load_config()
        self.registry: ToolRegistry = registry or ToolRegistry()
        self._all_tools: dict[str, StreamingTool] = {}
        self.active_tools: list[StreamingTool] = []
        self.memory: ConversationMemory | None = None
        self.client: FrameworkClient | None = None
        self.stop_event: asyncio.Event | None = None
        self.streaming = False
        self.buf = StreamBuffer()
        self._committed: list[Text] = []
        self._first = True

    def compose(self) -> ComposeResult:
        yield Static(
            "  Minimal Harness  ·  Ctrl+O Config  ·  Ctrl+T Tools  ·  Ctrl+D Dump  ·  Esc Interrupt  ",
            id="top-bar",
        )
        with Vertical(id="chat-container"):
            yield RichLog(id="chat-log", markup=True, wrap=True, highlight=False)
        with Vertical(id="input-area"):
            yield ListView(id="suggestion-list")
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

    def on_click(self) -> None:
        self._input.focus()

    @property
    def _rlog(self) -> RichLog:
        return self.query_one("#chat-log", RichLog)

    @property
    def _input(self) -> ChatInput:
        return self.query_one("#chat-input", ChatInput)

    @property
    def _wrap(self) -> Vertical:
        return self.query_one("#input-wrap", Vertical)

    @property
    def _suggestion_list(self) -> ListView:
        return self.query_one("#suggestion-list", ListView)

    def _filter_suggestions(self, prefix: str) -> list[tuple[str, str, str]]:
        return [
            (cmd, desc, action)
            for cmd, desc, action in self.SLASH_COMMANDS
            if cmd.startswith(prefix)
        ]

    def _show_suggestions(self, prefix: str) -> None:
        suggestions = self._filter_suggestions(prefix)
        if not suggestions:
            self._hide_suggestions()
            return
        self._suggestion_list.clear()
        for cmd, desc, _ in suggestions:
            self._suggestion_list.append(ListItem(Label(f"{cmd}  {desc}")))
        self._suggestion_list.add_class("visible")
        self._input.set_slash_active(True)
        if self._suggestion_list.children:
            self._suggestion_list.index = 0

    def _hide_suggestions(self) -> None:
        self._suggestion_list.remove_class("visible")
        self._suggestion_list.clear()
        self._input.set_slash_active(False)

    def on_slash_command_show(self, event: SlashCommandShow) -> None:
        self._show_suggestions(event.prefix)

    def on_slash_command_hide(self, event: SlashCommandHide) -> None:
        self._hide_suggestions()

    def on_slash_command_navigate_up(self, event: SlashCommandNavigateUp) -> None:
        sl = self._suggestion_list
        if sl.children:
            sl.action_cursor_up()

    def on_slash_command_navigate_down(self, event: SlashCommandNavigateDown) -> None:
        sl = self._suggestion_list
        if sl.children:
            sl.action_cursor_down()

    def on_slash_command_select(self, event: SlashCommandSelect) -> None:
        sl = self._suggestion_list
        if not sl.children or sl.index is None:
            return
        idx = sl.index
        suggestions = self._filter_suggestions(self._input.text)
        if 0 <= idx < len(suggestions):
            _, _, action = suggestions[idx]
            self._input.text = ""
            self._hide_suggestions()
            getattr(self, f"action_{action}")()

    def on_dump_request(self, event: DumpRequest) -> None:
        self.action_dump()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if not self._suggestion_list.has_class("visible"):
            return
        idx = event.list_view.index
        suggestions = self._filter_suggestions(self._input.text)
        if idx is not None and 0 <= idx < len(suggestions):
            _, _, action = suggestions[idx]
            self._input.text = ""
            self._hide_suggestions()
            getattr(self, f"action_{action}")()

    @property
    def _log_width(self) -> int:
        return max(self._rlog.content_size.width, 40)

    def _render_markdown(self, text: str, width: int = 80) -> Text:
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=width)
        console.print(Markdown(text))
        return Text.from_ansi(buf.getvalue())

    def say(self, text: str, style: str = "", is_markdown: bool = False) -> None:
        if is_markdown:
            t = self._render_markdown(text, self._log_width)
        elif style:
            t = Text(text, style=style)
        else:
            t = Text(text)
        self._committed.append(t)
        if not self.streaming:
            self._rlog.write(t)

    def _tick(self) -> None:
        if not self.streaming:
            return
        rlog = self._rlog
        max_scroll = rlog.max_scroll_y
        at_bottom = max_scroll == 0 or rlog.scroll_y >= max_scroll
        if not at_bottom:
            return
        rlog.clear()
        for line in self._committed:
            rlog.write(line)
        if self.buf.reasoning or self.buf.content:
            rlog.write(self.buf.render(width=self._log_width))
        if self.buf.tool_calls:
            for _, call in sorted(self.buf.tool_calls.items()):
                try:
                    args = json.dumps(
                        json.loads(call.get("arguments", "{}")), ensure_ascii=False
                    )
                except (json.JSONDecodeError, TypeError):
                    args = call.get("arguments", "")
                rlog.write(Text(f"  ▸ {call.get('name', '?')}({args})", style="bold #f9e2af"))
        rlog.scroll_end(animate=False)

    def _banner(self) -> None:
        self.say("Minimal Harness TUI", "bold #a6e3a1")
        self.say(f'  "{random.choice(J0EY1IU_QUOTES)}"  --J0ey1iu', "dim italic")
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

    def _rebuild(self) -> None:
        cfg = self.config
        self._all_tools = collect_tools(cfg, self.registry)
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

        prompt_path = cfg.get("system_prompt", DEFAULT_CONFIG["system_prompt"])
        prompt = read_system_prompt(Path(prompt_path)) if prompt_path else ""
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
                rendered = self.buf.render(width=self._log_width)
                if rendered.plain:
                    self._committed.append(rendered)
            self.buf.clear()
            self._rlog.clear()
            for line in self._committed:
                self._rlog.write(line)
            self._rlog.scroll_end(animate=False)
            self.stop_event = None
            self._set_streaming(False)

    def _on_event(self, event: Event) -> None:
        b = self.buf
        if isinstance(event, LLMChunkEvent):
            self._on_chunk(event)
        elif isinstance(event, LLMEndEvent):
            w = max(self._rlog.size.width, 40)
            rendered = b.render(width=w)
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
            pass
        elif isinstance(event, ToolProgressEvent):
            chunk = event.chunk
            if isinstance(chunk, dict):
                msg = chunk.get("message")
                if msg is None:
                    msg = json.dumps(chunk, ensure_ascii=False, default=str)
            else:
                msg = str(chunk)
            if len(msg) > MAX_DISPLAY_LENGTH:
                msg = msg[:MAX_DISPLAY_LENGTH] + "…"
            self.say(f"    · {msg}", "dim")
        elif isinstance(event, ToolEndEvent):
            r = event.result
            if isinstance(r, dict) and "error" in r:
                err_msg = r.get("error", "Unknown error")
                tb = r.get("traceback", "") or ""
                stderr = r.get("stderr", "") or ""
                full_err = err_msg
                if tb:
                    full_err += "\n\nTraceback:\n" + tb
                if stderr:
                    full_err += "\n\nStderr:\n" + stderr
                self.say(f"    ✗ {full_err}", "bold #f38ba8")
            else:
                if isinstance(r, dict):
                    s = json.dumps(r, ensure_ascii=False, default=str)
                elif isinstance(r, str):
                    s = r
                else:
                    s = str(r)
                if len(s) > MAX_DISPLAY_LENGTH:
                    s = s[:MAX_DISPLAY_LENGTH] + "…"
                self.say(f"    ✓ {s}", "#a6e3a1")
            self.say("")
        elif isinstance(event, AgentEndEvent):
            self.say("")

    def _on_chunk(self, event: LLMChunkEvent) -> None:
        b = self.buf
        try:
            delta = event.chunk.choices[0].delta  # type: ignore[OptionalMemberAccess]
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

    def action_interrupt(self) -> None:
        if self.streaming and self.stop_event is not None:
            self.stop_event.set()
            self.say("  ✗ interrupted", "bold #f38ba8")

    def action_new(self) -> None:
        if self.streaming:
            return
        self._committed.clear()
        self._rlog.clear()
        self.buf.clear()
        self._first = True
        if self.memory is not None:
            prompt_path = self.config.get("system_prompt", DEFAULT_CONFIG["system_prompt"])
            prompt = read_system_prompt(Path(prompt_path)) if prompt_path else ""
            self.memory = ConversationMemory(system_prompt=prompt)
        self._rebuild()
        self._banner()

    def action_share(self) -> None:
        if self.streaming:
            return

        def done(path: str | None) -> None:
            if not path:
                return
            try:
                rlog = self._rlog
                width = rlog.content_size.width or 80
                height = rlog.content_size.height or 24
                buf = StringIO()
                console = Console(
                    file=buf,
                    force_terminal=True,
                    width=width,
                    height=height,
                    record=True,
                    legacy_windows=False,
                    color_system="truecolor",
                )
                for text in self._committed:
                    console.print(text)
                svg = console.export_svg(title="Minimal Harness Chat")
                p = Path(path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(svg, encoding="utf-8")
                self.say(f"✓ Chat exported → {path}", "bold #a6e3a1")
            except Exception as e:
                self.say(f"✗ {e}", "bold #f38ba8")

        self.push_screen(PromptScreen("📸  Export chat as SVG", "./chat-container.svg"), done)

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
        def done(ok: bool | None) -> None:
            if ok:
                self.exit()

        self.push_screen(
            ConfirmScreen("Quit?", "Memory will be lost.", ok="Quit", variant="error"),
            done,
        )


def main() -> None:
    config = load_config()
    TUIApp(config=config).run()