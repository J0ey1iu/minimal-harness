"""Main TUI application."""

from __future__ import annotations

import asyncio
import json
import random
from io import StringIO
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, Label, ListItem, ListView, RichLog, Static

from minimal_harness.client.built_in.buffer import StreamBuffer
from minimal_harness.client.built_in.config import (
    DEFAULT_CONFIG,
    J0EY1IU_QUOTES,
    THEMES,
)
from minimal_harness.client.built_in.context import AppContext
from minimal_harness.client.built_in.memory import PersistentMemory
from minimal_harness.client.built_in.modals import (
    ConfigScreen,
    ConfirmScreen,
    PromptScreen,
    SessionSelectScreen,
    ToolSelectScreen,
)
from minimal_harness.client.built_in.widgets import (
    ChatInput,
    ChatInputDump,
    ChatInputSubmit,
    SlashCommandHide,
    SlashCommandNavigateDown,
    SlashCommandNavigateUp,
    SlashCommandSelect,
    SlashCommandShow,
)
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

FLUSH_INTERVAL = 0.25
MAX_DISPLAY_LENGTH = 500

_CSS_PATH = Path(__file__).parent / "app.css"

_BUILT_IN_TOOL_NAMES: set[str] | None = None


def _get_built_in_tool_names() -> set[str]:
    global _BUILT_IN_TOOL_NAMES
    if _BUILT_IN_TOOL_NAMES is None:
        from minimal_harness.tool.built_in.bash import get_tools as get_bash_tools
        from minimal_harness.tool.built_in.patch_file import (
            get_tools as get_patch_file_tools,
        )

        _BUILT_IN_TOOL_NAMES = {
            n for getter in (get_bash_tools, get_patch_file_tools) for n in getter()
        }
    return _BUILT_IN_TOOL_NAMES


class TUIApp(App):
    TITLE = "Minimal Harness"
    ENABLE_COMMAND_PALETTE = False

    CSS_PATH = _CSS_PATH

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
        ("/sessions", "Resume a past session", "sessions"),
        ("/share", "Export chat as SVG", "share"),
    ]

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        registry: Any = None,
    ) -> None:
        super().__init__()
        self.ctx = AppContext(config=config, registry=registry)
        self.stop_event: asyncio.Event | None = None
        self.streaming = False
        self.buf = StreamBuffer()
        self._committed: list[Text] = []
        self._first = True

    @property
    def config(self) -> dict[str, Any]:
        return self.ctx.config

    @property
    def memory(self):
        return self.ctx.memory

    @property
    def active_tools(self):
        return self.ctx.active_tools

    @property
    def agent(self):
        return self.ctx.agent

    @property
    def _all_tools(self):
        return self.ctx._all_tools

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
        theme = self.ctx.config.get("theme", DEFAULT_CONFIG["theme"])
        if theme in THEMES:
            self.theme = theme
        self.ctx.rebuild()
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

    def on_chat_input_submit(self, event: ChatInputSubmit) -> None:
        self.action_submit()

    def on_chat_input_dump(self, event: ChatInputDump) -> None:
        self.action_dump()

    @property
    def _log_width(self) -> int:
        return max(self._rlog.content_size.width, 40)

    def _render_markdown(self, text: str, width: int = 80) -> Text:
        buf = StringIO()
        with Console(file=buf, force_terminal=True, width=width) as console:
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
                rlog.write(
                    Text(f"  ▸ {call.get('name', '?')}({args})", style="bold #f9e2af")
                )
        rlog.scroll_end(animate=False)

    def _banner(self) -> None:
        self.say("Minimal Harness TUI", "bold #a6e3a1")
        self.say(f'  "{random.choice(J0EY1IU_QUOTES)}"  --J0ey1iu', "dim italic")
        self.say("")
        if not self.ctx.config.get("api_key"):
            self.say("⚠  No API key configured — press Ctrl+O", "bold #f9e2af")

        built_in = _get_built_in_tool_names()
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
        if self.agent is None:
            self.say("Agent not initialized.", "bold #f38ba8")
            return
        self.buf.clear()
        self.stop_event = asyncio.Event()
        self._set_streaming(True)
        try:
            async for event in self.agent.run(
                user_input=[{"type": "text", "text": user_input}],
                stop_event=self.stop_event,
                memory=self.memory,
                tools=self.active_tools or None,
            ):
                if self.stop_event.is_set():
                    break
                self._on_event(event.to_client_event())
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
        delta = event.chunk
        if delta is None:
            return

        if delta.reasoning:
            b.reasoning += delta.reasoning

        if delta.content:
            b.content += delta.content

        if delta.tool_calls:
            for tc in delta.tool_calls:
                call = b.tool_calls.setdefault(
                    tc.index, {"id": "", "name": "", "arguments": ""}
                )
                if tc.id:
                    call["id"] += tc.id
                if tc.name:
                    call["name"] += tc.name
                if tc.arguments:
                    call["arguments"] += tc.arguments

    def action_interrupt(self) -> None:
        if self.streaming and self.stop_event is not None:
            self.stop_event.set()
            self.say("  ✗ interrupted", "bold #f38ba8")

    def _replay_memory(self) -> None:
        """Replay all non-system messages from memory into the chat log."""
        if self.memory is None:
            return
        messages = self.memory.get_all_messages()

        tool_names: dict[str, str] = {}
        for msg in messages:
            if msg.get("role") == "assistant":
                tcs = msg.get("tool_calls")
                if isinstance(tcs, list):
                    for tc in tcs:
                        tid = tc.get("id", "") if isinstance(tc, dict) else ""
                        name = (
                            tc.get("function", {}).get("name", "?")
                            if isinstance(tc, dict)
                            else "?"
                        )
                        if tid:
                            tool_names[tid] = name

        for msg in messages:
            role = msg.get("role")
            if role == "system":
                continue
            if role == "user":
                parts = msg.get("content")
                if not isinstance(parts, list):
                    continue
                texts = []
                for part in parts:
                    if isinstance(part, dict) and part.get("type") == "text":
                        texts.append(part.get("text", ""))
                text = " ".join(texts)
                if text:
                    self.say(f"\n❯ {text}", "bold #89b4fa")
                    self.say("")
            elif role == "assistant":
                content = msg.get("content")
                if isinstance(content, str) and content:
                    self.say(content, is_markdown=True)
                tcs = msg.get("tool_calls")
                if isinstance(tcs, list):
                    for tc in tcs:
                        if not isinstance(tc, dict):
                            continue
                        func = tc.get("function", {})
                        if not isinstance(func, dict):
                            continue
                        try:
                            args = json.dumps(
                                json.loads(func.get("arguments", "{}")),
                                ensure_ascii=False,
                            )
                        except (json.JSONDecodeError, TypeError):
                            args = func.get("arguments", "")
                        self.say(
                            f"  ▸ {func.get('name', '?')}({args})",
                            "bold #f9e2af",
                        )
                self.say("")
                self.say("")
            elif role == "tool":
                content = msg.get("content")
                if not isinstance(content, str):
                    continue
                if content.startswith(("[Tool Error]", "[Tool Execution Stopped]")):
                    self.say(f"    ✗ {content}", "bold #f38ba8")
                else:
                    s = content
                    if len(s) > MAX_DISPLAY_LENGTH:
                        s = s[:MAX_DISPLAY_LENGTH] + "…"
                    self.say(f"    ✓ {s}", "#a6e3a1")
                self.say("")

    def action_new(self) -> None:
        if self.streaming:
            return
        self._committed.clear()
        self._rlog.clear()
        self.buf.clear()
        self._first = True
        self.ctx.reset_memory()
        self.ctx.rebuild()
        self._banner()

    def action_sessions(self) -> None:
        if self.streaming:
            return
        sessions = PersistentMemory.list_sessions()

        def done(session_id: str | None) -> None:
            if not session_id:
                return
            try:
                memory = PersistentMemory.from_session(session_id)
                self.ctx.memory = memory
                self._committed.clear()
                self._rlog.clear()
                self.buf.clear()
                self._first = True
                self._banner()
                title = memory._title or "Untitled"
                self.say(
                    f"✓ Session resumed: {title}",
                    "bold #a6e3a1",
                )
                self._replay_memory()
                self._first = False
                self._rlog.scroll_end(animate=False)
                # Populate input history so up/down arrows work for resumed sessions
                inputs: list[str] = []
                for msg in memory.get_all_messages():
                    if msg.get("role") == "user":
                        parts = msg.get("content")
                        if isinstance(parts, list):
                            texts = [
                                p.get("text", "")
                                for p in parts
                                if isinstance(p, dict) and p.get("type") == "text"
                            ]
                            text = " ".join(texts)
                            if text:
                                inputs.append(text)
                self._input._input_history = inputs
                self._input.reset_history_index()
            except Exception as e:
                self.say(f"✗ {e}", "bold #f38ba8")

        self.push_screen(SessionSelectScreen(sessions), done)

    def action_share(self) -> None:
        if self.streaming:
            return

        def done(path: str | None) -> None:
            if not path:
                return
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
            try:
                with console:
                    for text in self._committed:
                        console.print(text)
                    svg = console.export_svg(title="Minimal Harness Chat")
                p = Path(path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(svg, encoding="utf-8")
                self.say(f"✓ Chat exported → {path}", "bold #a6e3a1")
            except Exception as e:
                self.say(f"✗ {e}", "bold #f38ba8")

        self.push_screen(
            PromptScreen("📸  Export chat as SVG", "./chat-container.svg"), done
        )

    def action_config(self) -> None:
        if self.streaming:
            return

        def done(result: dict | None) -> None:
            if result is None:
                return
            self.ctx.update_config(result)
            if (t := result.get("theme")) in THEMES:
                self.theme = t
            self.ctx.rebuild()
            self.say("✓ Configuration saved", "bold #a6e3a1")

        self.push_screen(ConfigScreen(self.ctx.config), done)

    def action_tools(self) -> None:
        if self.streaming or not self._all_tools:
            return
        selected = {t.name for t in self.active_tools}

        def done(chosen: list[str] | None) -> None:
            if chosen is None:
                return
            self.ctx.select_tools(chosen)
            self.ctx.rebuild()
            names = ", ".join(t.name for t in self.active_tools) or "(none)"
            self.say(f"✓ Tools: {names}", "bold #a6e3a1")

        self.push_screen(ToolSelectScreen(self._all_tools, selected), done)

    def action_dump(self) -> None:
        if self.memory is None:
            return
        memory = self.memory

        def done(path: str | None) -> None:
            if not path:
                return
            try:
                p = Path(path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(
                    memory.dump_memory_json(indent=2),
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
            ConfirmScreen("Quit?", "Session is saved.", ok="Quit", variant="error"),
            done,
        )


def main() -> None:
    from minimal_harness.client.built_in.config import load_config

    config = load_config()
    TUIApp(config=config).run()
