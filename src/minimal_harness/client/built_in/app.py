"""Main TUI application."""

from __future__ import annotations

import asyncio
import json
import random
from io import StringIO
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Footer, ListView, Static

from minimal_harness.agent import Agent
from minimal_harness.client.built_in.buffer import StreamBuffer
from minimal_harness.client.built_in.chat_widgets import (
    AssistantMsg,
    ChatMsg,
    ReasoningMsg,
    ToolCallMsg,
)
from minimal_harness.client.built_in.config import DEFAULT_CONFIG
from minimal_harness.client.built_in.constants import (
    FLUSH_INTERVAL,
    J0EY1IU_QUOTES,
    THEMES,
)
from minimal_harness.client.built_in.context import AppContext
from minimal_harness.client.built_in.markdown_styles import LazyMarkdown
from minimal_harness.client.built_in.memory import PersistentMemory
from minimal_harness.client.built_in.modals import (
    ConfigScreen,
    ConfirmScreen,
    PromptScreen,
    SessionSelectScreen,
    ToolSelectScreen,
)
from minimal_harness.client.built_in.renderer import (
    format_tool_call_static,
    format_tool_result_static,
    truncate_static,
)
from minimal_harness.client.built_in.session_manager import SessionManager
from minimal_harness.client.built_in.slash_handler import SlashCommandHandler
from minimal_harness.client.built_in.widgets import (
    Banner,
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
from minimal_harness.tool.base import StreamingTool
from minimal_harness.tool.registry import ToolRegistry

_CSS_PATH = Path(__file__).parent / "app.tcss"

_BUILT_IN_TOOL_NAMES: set[str] | None = None


def _get_built_in_tool_names() -> set[str]:
    global _BUILT_IN_TOOL_NAMES
    if _BUILT_IN_TOOL_NAMES is None:
        from minimal_harness.tool.built_in.bash import get_tools as get_bash_tools
        from minimal_harness.tool.built_in.local_file_operation import (
            get_tools as get_local_file_operation_tools,
        )

        _BUILT_IN_TOOL_NAMES = {
            n
            for getter in (get_bash_tools, get_local_file_operation_tools)
            for n in getter()
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

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        super().__init__()
        self.ctx = AppContext(config=config, registry=registry)
        self.stop_event: asyncio.Event | None = None
        self.streaming = False
        self.buf = StreamBuffer()
        self._msg_counter: int = 0
        self._first = True
        self._streaming_reasoning: ReasoningMsg | None = None
        self._streaming_content: AssistantMsg | None = None
        self._streaming_tool_widgets: dict[int, ToolCallMsg] = {}
        self._export_history: list[tuple[str, str | None, bool]] = []
        self._slash_handler: SlashCommandHandler | None = None
        self._session_manager: SessionManager | None = None

    @property
    def config(self) -> dict[str, Any]:
        return self.ctx.config

    @property
    def memory(self) -> PersistentMemory | None:
        return self.ctx.memory

    @property
    def active_tools(self) -> list[StreamingTool]:
        return self.ctx.active_tools

    @property
    def agent(self) -> Agent | None:
        return self.ctx.agent

    @property
    def _all_tools(self) -> dict[str, StreamingTool]:
        return self.ctx._all_tools

    def compose(self) -> ComposeResult:
        yield Static(
            "  Minimal Harness  ·  Ctrl+O Config  ·  Ctrl+T Tools  ·  Ctrl+D Dump  ·  Esc Interrupt  ",
            id="top-bar",
        )
        with Vertical(id="chat-container"):
            yield Banner(id="banner")
            yield VerticalScroll(id="chat-scroll")
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
        self._slash_handler = SlashCommandHandler(
            suggestion_list=self._suggestion_list,
            input_widget=self._input,
            get_input_text=lambda: self._input.text,
            set_input_text=lambda t: setattr(self._input, "text", t),
            execute_action=lambda a: getattr(self, f"action_{a}")(),
        )
        self._session_manager = SessionManager(
            ctx=self.ctx,
            say=self.say,
            scroll_end=lambda animate=True: self._chat.scroll_end(animate=animate),
            clear_rlog=lambda: None,
            clear_input=lambda: setattr(self._input, "text", ""),
            set_input_history=lambda h: (
                setattr(self._input, "_input_history", h)
                or self._input.reset_history_index()
            ),
        )
        self.set_interval(FLUSH_INTERVAL, self._tick)
        self._input.focus()
        self._chat.display = False
        self._banner()

    def on_click(self) -> None:
        self._input.focus()

    @property
    def _chat(self) -> VerticalScroll:
        return self.query_one("#chat-scroll", VerticalScroll)

    @property
    def _chat_width(self) -> int:
        w = self._chat.size.width
        return max(w - 4, 20) if w > 0 else 80

    def _next_msg_id(self) -> str:
        self._msg_counter += 1
        return f"msg-{self._msg_counter}"

    @property
    def _input(self) -> ChatInput:
        return self.query_one("#chat-input", ChatInput)

    @property
    def _wrap(self) -> Vertical:
        return self.query_one("#input-wrap", Vertical)

    @property
    def _suggestion_list(self) -> ListView:
        return self.query_one("#suggestion-list", ListView)

    @property
    def _banner_widget(self) -> Banner:
        return self.query_one("#banner", Banner)

    def on_slash_command_show(self, event: SlashCommandShow) -> None:
        if self._slash_handler:
            self._slash_handler.on_slash_command_show(event.prefix)

    def on_slash_command_hide(self, event: SlashCommandHide) -> None:
        if self._slash_handler:
            self._slash_handler.on_slash_command_hide()

    def on_slash_command_navigate_up(self, event: SlashCommandNavigateUp) -> None:
        if self._slash_handler:
            self._slash_handler.on_slash_command_navigate_up()

    def on_slash_command_navigate_down(self, event: SlashCommandNavigateDown) -> None:
        if self._slash_handler:
            self._slash_handler.on_slash_command_navigate_down()

    def on_slash_command_select(self, event: SlashCommandSelect) -> None:
        if self._slash_handler:
            self._slash_handler.on_slash_command_select()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if self._slash_handler:
            self._slash_handler.on_list_view_selected(event.list_view.index)

    def on_chat_input_submit(self, event: ChatInputSubmit) -> None:
        self.action_submit()

    def on_chat_input_dump(self, event: ChatInputDump) -> None:
        self.action_dump()

    def _render_markdown(self, text: str, width: int = 80) -> LazyMarkdown:
        return LazyMarkdown(text)

    def say(
        self, text: str | Text, style: str = "", is_markdown: bool = False
    ) -> None:
        mid = self._next_msg_id()
        if isinstance(text, Text):
            w = ChatMsg(text, id=mid)
            self._export_history.append(
                (text.plain, str(text.style) if text.style else None, False)
            )
        elif is_markdown:
            w = AssistantMsg(self._render_markdown(text), id=mid)
            self._export_history.append((text, None, True))
        elif style:
            w = ChatMsg(
                Text(text, style=style, no_wrap=False, overflow="fold"), id=mid
            )
            self._export_history.append((text, style, False))
        else:
            w = ChatMsg(text, id=mid)
            self._export_history.append((text, None, False))
        self._chat.mount(w)
        w.scroll_visible()

    def _flush_buffer_to_committed(self) -> None:
        b = self.buf
        had_content = bool(b.reasoning or b.content)
        if self._streaming_reasoning is not None:
            self._streaming_reasoning.remove()
            self._streaming_reasoning = None
        if self._streaming_content is not None:
            self._streaming_content.remove()
            self._streaming_content = None
        for w in self._streaming_tool_widgets.values():
            w.remove()
        self._streaming_tool_widgets.clear()
        if b.reasoning:
            mid = self._next_msg_id()
            w = ReasoningMsg(b.reasoning, id=mid)
            self._chat.mount(w)
            self._export_history.append((b.reasoning, "dim", False))
        if b.content:
            rendered = self._render_markdown(b.content, self._chat_width)
            mid = self._next_msg_id()
            w = AssistantMsg(rendered, id=mid)
            self._chat.mount(w)
            self._export_history.append((b.content, None, True))
        if b.tool_calls:
            for _, call in sorted(b.tool_calls.items()):
                tw = format_tool_call_static(call)
                tw.no_wrap = False
                tw.overflow = "fold"
                mid = self._next_msg_id()
                w = ToolCallMsg(tw, id=mid)
                self._chat.mount(w)
                self._export_history.append(
                    (tw.plain, str(tw.style) if tw.style else None, False)
                )
            b.tool_calls.clear()
        if had_content:
            b._flushed = True
        b.reasoning = ""
        b.content = ""

    def _tick(self) -> None:
        if not self.streaming:
            return
        chat = self._chat
        max_scroll = chat.max_scroll_y
        at_bottom = max_scroll == 0 or chat.scroll_y >= max_scroll
        if not at_bottom:
            return
        b = self.buf
        width = self._chat_width
        if b.reasoning:
            if self._streaming_reasoning is None:
                self._streaming_reasoning = ReasoningMsg(
                    b.reasoning, id=self._next_msg_id()
                )
                chat.mount(self._streaming_reasoning)
            else:
                self._streaming_reasoning.update(b.reasoning)
        elif self._streaming_reasoning is not None:
            self._streaming_reasoning.remove()
            self._streaming_reasoning = None
        if b.content:
            rendered = self._render_markdown(b.content, width)
            if self._streaming_content is None:
                self._streaming_content = AssistantMsg(
                    rendered, id=self._next_msg_id()
                )
                chat.mount(self._streaming_content)
            else:
                self._streaming_content.update(rendered)
        elif self._streaming_content is not None:
            self._streaming_content.remove()
            self._streaming_content = None
        if b.tool_calls:
            prev_ids = set(self._streaming_tool_widgets.keys())
            cur_ids = set(b.tool_calls.keys())
            for idx in prev_ids - cur_ids:
                self._streaming_tool_widgets[idx].remove()
                del self._streaming_tool_widgets[idx]
            for idx, call in sorted(b.tool_calls.items()):
                tw = format_tool_call_static(call)
                tw.no_wrap = False
                tw.overflow = "fold"
                if idx in self._streaming_tool_widgets:
                    self._streaming_tool_widgets[idx].update(tw)
                else:
                    w = ToolCallMsg(tw, id=self._next_msg_id())
                    chat.mount(w)
                    self._streaming_tool_widgets[idx] = w
        elif self._streaming_tool_widgets:
            for w in self._streaming_tool_widgets.values():
                w.remove()
            self._streaming_tool_widgets.clear()
        chat.scroll_end(animate=False)

    def _banner(self) -> None:
        lines: list[Text] = []
        lines.append(Text("  Minimal Harness TUI", style="bold bright_green"))
        lines.append(
            Text(f'  "{random.choice(J0EY1IU_QUOTES)}"  --J0ey1iu', style="dim italic")
        )
        lines.append(Text(""))
        if not self.ctx.config.get("api_key"):
            lines.append(
                Text(
                    "⚠  No API key configured — press Ctrl+O",
                    style="bold bright_yellow",
                )
            )

        built_in = _get_built_in_tool_names()
        ext = [t for t in self._all_tools.values() if t.name not in built_in]
        if ext:
            lines.append(
                Text(
                    f"Loaded {len(ext)} external tool(s): "
                    + ", ".join(t.name for t in ext),
                    style="dim",
                )
            )
        active = ", ".join(t.name for t in self.active_tools) or "(none)"
        lines.append(Text(f"Active tools: {active}", style="dim"))
        self._banner_widget.update(Text("\n").join(lines))
        self._banner_widget.display = True
        self._chat.display = False

    def action_submit(self) -> None:
        text = self._input.text.strip()
        if not text or self.streaming:
            return
        self._input.text = ""
        if self._first:
            self._first = False
            self._export_history.clear()
            self._chat.query("ChatMsg").remove()
            self.buf.clear()
            self._banner_widget.display = False
            self._chat.display = True
        self.say("")
        self.say(f"❯ {text}", "bold bright_blue")
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
            self.say("Agent not initialized.", "bold bright_red")
            return
        self.buf.clear()
        self.stop_event = asyncio.Event()
        self._set_streaming(True)
        try:
            async for event in self.agent.run(
                user_input=[{"type": "text", "text": user_input}],
                stop_event=self.stop_event,
                memory=self.memory,
                tools=self.active_tools,
            ):
                if self.stop_event.is_set():
                    break
                self._on_event(event.to_client_event())
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.say(f"\nError: {e}", "bold bright_red")
        finally:
            if not self.buf._flushed:
                self._flush_buffer_to_committed()
            self.buf.clear()
            self.stop_event = None
            self._set_streaming(False)

    def _on_event(self, event: Event) -> None:
        if isinstance(event, LLMChunkEvent):
            self._on_chunk(event)
        elif isinstance(event, LLMEndEvent):
            self._flush_buffer_to_committed()
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
            self.say(f"  ⚡ Executing: {names}", "bold bright_yellow")
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
            self.say(f"    · {truncate_static(msg)}", "dim")
        elif isinstance(event, ToolEndEvent):
            self.say(format_tool_result_static(event.result))
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
            self.say("  ✗ interrupted", "bold bright_red")

    def action_new(self) -> None:
        if self.streaming:
            return
        self._export_history.clear()
        self._chat.query("ChatMsg").remove()
        self.buf.clear()
        self._first = True
        self.ctx.reset_memory()
        self.ctx.rebuild()
        self._banner_widget.display = True
        self._chat.display = False
        self._banner()

    def action_sessions(self) -> None:
        if self.streaming:
            return
        sessions = PersistentMemory.list_sessions()

        def done(session_id: str | None) -> None:
            if not session_id or self._session_manager is None:
                return
            self._first = True
            success = self._session_manager.load_session(
                session_id,
                clear_committed=lambda: (
                    self._export_history.clear(),
                    self._chat.query("ChatMsg").remove(),
                    None,
                )[2],
                clear_buf=self.buf.clear,
            )
            if success:
                self._first = False
                self._banner_widget.display = False
                self._chat.display = True

        self.push_screen(SessionSelectScreen(sessions), done)

    def action_share(self) -> None:
        if self.streaming:
            return

        def done(path: str | None) -> None:
            if not path:
                return
            width = self._chat_width or 80
            height = 24
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
                    for text, style, is_md in self._export_history:
                        if is_md:
                            console.print(self._render_markdown(text, width))
                        elif style:
                            console.print(Text(text, style=style))
                        else:
                            console.print(Text(text))
                svg = console.export_svg(title="Minimal Harness Chat")
                p = Path(path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(svg, encoding="utf-8")
                self.say(f"✓ Chat exported → {path}", "bold bright_green")
            except Exception as e:
                self.say(f"✗ {e}", "bold bright_red")

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
            self.say("✓ Configuration saved", "bold bright_green")
            if self._first:
                self._banner()

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
            self.say(f"✓ Tools: {names}", "bold bright_green")
            if self._first:
                self._banner()

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
                self.say(f"✓ Memory dumped → {path}", "bold bright_green")
            except Exception as e:
                self.say(f"✗ {e}", "bold bright_red")

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
