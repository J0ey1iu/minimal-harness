"""Main TUI application."""

from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Footer, ListView, Static

from minimal_harness.agent import (
    AgentRegistry,
    AgentRegistryProtocol,
    AgentRuntime,
)
from minimal_harness.client.built_in.config import DEFAULT_CONFIG
from minimal_harness.client.built_in.constants import (
    FLUSH_INTERVAL,
    J0EY1IU_QUOTES,
    THEMES,
)
from minimal_harness.client.built_in.context import AppContext
from minimal_harness.client.built_in.display import ChatDisplay
from minimal_harness.client.built_in.export_presenter import ExportPresenter
from minimal_harness.client.built_in.modals import (
    AgentSelectScreen,
    ConfigScreen,
    ConfirmScreen,
    PromptScreen,
    SessionSelectScreen,
    ToolSelectScreen,
)
from minimal_harness.client.built_in.session_controller import SessionController
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
from minimal_harness.client.events import to_client_event
from minimal_harness.tool.base import Tool
from minimal_harness.tool.registry import ToolRegistry

if TYPE_CHECKING:
    from minimal_harness.memory import Memory

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
        self._agent_registry: AgentRegistryProtocol = AgentRegistry()
        self._runtime: AgentRuntime = AgentRuntime()
        self._ctrl = SessionController(self._runtime, self._agent_registry, self.ctx)
        self._announced_delegates: set[str] = set()
        self._first = True
        self._chat_display: ChatDisplay | None = None
        self._exporter: ExportPresenter | None = None
        self._slash_handler: SlashCommandHandler | None = None
        self._session_manager: SessionManager | None = None

    @property
    def config(self) -> dict[str, Any]:
        return self.ctx.config

    @property
    def memory(self) -> Memory | None:
        return self._ctrl.memory

    @property
    def active_tools(self) -> list[Tool]:
        return self._ctrl.active_tools

    @property
    def _all_tools(self) -> dict[str, Tool]:
        return self.ctx.all_tools

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
        self._ctrl.register_preset_agents()
        d = ChatDisplay(
            chat_container=self._chat,
            theme=self.theme,
        )
        self._chat_display = d
        self._exporter = ExportPresenter(
            get_theme=lambda: self.theme,
            say=d.say,
        )
        self._slash_handler = SlashCommandHandler(
            suggestion_list=self._suggestion_list,
            input_widget=self._input,
            get_input_text=lambda: self._input.text,
            set_input_text=lambda t: setattr(self._input, "text", t),
            execute_action=lambda a: getattr(self, f"action_{a}")(),
        )
        self._session_manager = SessionManager(
            runtime=self._runtime,
            ctx=self.ctx,
            display=d,
            clear_input=lambda: setattr(self._input, "text", ""),
            show_banner=self._banner,
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

    def _tick(self) -> None:
        if self._chat_display is not None:
            self._chat_display.tick(self._ctrl.buf, self._ctrl.streaming)
        self._poll_handoff_events()

    def _banner(self, show: bool = True) -> None:
        lines: list[Text] = []
        lines.append(Text("  Minimal Harness TUI", style="bold bright_green"))
        lines.append(
            Text(f'  "{random.choice(J0EY1IU_QUOTES)}"  --J0ey1iu', style="dim italic")
        )
        lines.append(Text(""))
        if not self.ctx.config.get("api_key"):
            lines.append(
                Text(
                    "\u26a0  No API key configured — press Ctrl+O",
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
        if show:
            self._banner_widget.display = True
            self._chat.display = False

    def action_submit(self) -> None:
        d = self._chat_display
        if d is None:
            return
        text = self._input.text.strip()
        if not text or self._ctrl.streaming:
            return
        self._input.text = ""
        if self._first:
            self._first = False
            d.clear_chat()
            self._ctrl.buf.clear()
            self._banner_widget.display = False
            self._chat.display = True
        d.say(text, user=True)
        self._run(text)

    def _set_streaming(self, active: bool) -> None:
        self._ctrl.set_streaming(active)
        self._wrap.set_class(active, "streaming")
        self._input.disabled = active
        if not active:
            self._input.focus()

    @work(exclusive=True)
    async def _run(self, user_input: str) -> None:
        d = self._chat_display
        if d is None:
            return
        if self._ctrl.current_session_id is None:
            self._ctrl.start_with_default_agent()
        sid = self._ctrl.current_session_id
        sess = self._ctrl.current_session
        if sess is None:
            return
        self._ctrl.buf.clear()
        sess.reset()
        self._set_streaming(True)
        try:
            self._ctrl.inject_runtime_tools(sess.tools)
            stop_event, event_queue = self._ctrl.start_run(sess, user_input)
            while True:
                event = await event_queue.get()
                if event is None:
                    break
                if stop_event.is_set():
                    break
                d.handle_event(
                    to_client_event(event),
                    buf=self._ctrl.buf,
                    memory=sess.memory,
                )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            d.say(f"\nError: {e}", "bold bright_red")
        finally:
            if not self._ctrl.buf.flushed:
                d.flush(self._ctrl.buf)
            self._ctrl.buf.clear()
            self._set_streaming(False)
            if sid:
                self._ctrl.end_run(sid)

    def action_interrupt(self) -> None:
        if not self._ctrl.streaming:
            return
        d = self._chat_display
        if d is None:
            return
        self._ctrl.interrupt()
        d.say("  \u2717 interrupted", "bold bright_red")

    def _poll_handoff_events(self) -> None:
        d = self._chat_display
        sid = self._ctrl.current_session_id

        # Drain events for the currently-viewed session (background run)
        if sid:
            events, done = self._ctrl.drain_session_events(sid)
            if done:
                self._set_streaming(False)
                if d is not None:
                    if not self._ctrl.buf.flushed:
                        d.flush(self._ctrl.buf)
                    self._ctrl.buf.clear()
                    d.say("\u2713 Session ready", "bold bright_green")
            if events and d is not None:
                sess = self._ctrl.current_session
                for event in events:
                    d.handle_event(
                        to_client_event(event),
                        buf=self._ctrl.buf,
                        memory=sess.memory if sess else None,
                    )

        # Announce new handoff targets once
        for target_id in list(self._ctrl.handoff_target_ids):
            if target_id not in self._announced_delegates:
                self._announced_delegates.add(target_id)
                target = self._ctrl._sessions.get(target_id)
                name = target.name if target else "Agent"
                if d is not None:
                    d.say(f"\u2192 Delegated to {name}", "bold bright_blue")

        # Check for completed handoffs (not the currently-viewed session)
        if self._ctrl.poll_handoff_completion():
            if d is not None:
                d.say("\u2713 Handoff completed", "bold bright_green")

    def action_new(self) -> None:
        if self._ctrl.streaming:
            return

        from minimal_harness.client.built_in.config.agents import (
            SYSTEM_PROMPTS_DIR,
            load_agents_config,
            read_system_prompt,
        )

        agents = load_agents_config()

        def _pick_agent() -> None:
            def on_agent(agent: dict[str, Any] | None) -> None:
                if not agent:
                    return
                d = self._chat_display
                if d is None:
                    return
                prompt = read_system_prompt(
                    SYSTEM_PROMPTS_DIR / agent["system_prompt"]
                ) or agent.get("description", "")
                d.clear_chat()
                self._ctrl.buf.clear()
                self._first = True
                self._ctrl.create_session(
                    agent_name=agent["name"],
                    system_prompt=prompt,
                    default_tools=agent.get("default_tools"),
                )
                self._banner_widget.display = True
                self._chat.display = False
                self._banner()

            self.push_screen(AgentSelectScreen(agents), on_agent)

        if self._first:
            _pick_agent()
        else:
            self.push_screen(
                ConfirmScreen(
                    "Start new chat?",
                    "Session is saved.",
                    ok="New Chat",
                    variant="primary",
                ),
                lambda ok: _pick_agent() if ok else None,
            )

    def action_sessions(self) -> None:
        if self._ctrl.streaming:
            return
        sessions = self._ctrl.get_all_sessions_metadata()

        def done(session_id: str | None) -> None:
            if not session_id or self._session_manager is None:
                return
            d = self._chat_display
            if d is None:
                return
            self._first = True

            session = self._ctrl.load_session_from_disk(session_id)
            if session:
                self._ctrl.switch_session(session_id)
                success, inputs = self._session_manager.replay_session(
                    session,
                    clear_committed=self._clear_committed,
                    clear_buf=self._ctrl.buf.clear,
                )
                if success:
                    self._first = False
                    self._banner_widget.display = False
                    self._chat.display = True
                    self._input.input_history = inputs
                    self._input.reset_history_index()
                    if session_id in self._ctrl._active_runs:
                        self._set_streaming(True)
                        d.say(
                            "  \u23f3 Session is running — waiting for completion",
                            "bold bright_yellow",
                        )

        self.push_screen(SessionSelectScreen(sessions), done)

    def _clear_committed(self) -> None:
        if self._chat_display is not None:
            self._chat_display.clear_chat()

    def action_share(self) -> None:
        if self._ctrl.streaming:
            return
        d = self._chat_display
        e = self._exporter
        if d is None or e is None:
            return

        def done(path: str | None) -> None:
            if path:
                e.export_svg(
                    path,
                    export_history=d.export_history,
                    chat_width=self._chat_width,
                )

        self.push_screen(
            PromptScreen("\U0001f4f8  Export chat as SVG", "./chat-container.svg"), done
        )

    def action_config(self) -> None:
        if self._ctrl.streaming:
            return

        def done(result: dict | None) -> None:
            if result is None:
                return
            d = self._chat_display
            if d is None:
                return
            self.ctx.update_config(result)
            self.ctx.refresh_tools()
            if (t := result.get("theme")) in THEMES:
                self.theme = t
                d.theme = t
            self._ctrl.rebuild_current_session(
                llm_provider=self.ctx._create_llm_provider(self.ctx.config),
            )
            d.say("\u2713 Configuration saved", "bold bright_green")
            self._banner(show=self._first)

        self.push_screen(ConfigScreen(self.ctx.config), done)

    def action_tools(self) -> None:
        if self._ctrl.streaming or not self._all_tools:
            return
        selected = {t.name for t in self.active_tools}

        def done(chosen: list[str] | None) -> None:
            if chosen is None:
                return
            d = self._chat_display
            if d is None:
                return
            resolved = [
                self.ctx.all_tools[n] for n in chosen if n in self.ctx.all_tools
            ]
            self.ctx.select_tools(chosen)
            sess = self._ctrl.current_session
            if sess:
                self._ctrl.rebuild_current_session(
                    llm_provider=self.ctx._create_llm_provider(self.ctx.config),
                    tools=resolved,
                    agent_factory=self.ctx._agent_factory,
                )
                sess.memory.selected_tools = chosen  # type: ignore[reportAttributeAccessIssue]
            names = ", ".join(t.name for t in resolved) or "(none)"
            d.say(f"\u2713 Tools: {names}", "bold bright_green")
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
            d = self._chat_display
            if d is None:
                return
            try:
                p = Path(path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(
                    memory.dump_memory_json(indent=2),
                    encoding="utf-8",
                )
                d.say(f"\u2713 Memory dumped \u2192 {path}", "bold bright_green")
            except Exception as e:
                d.say(f"\u2717 {e}", "bold bright_red")

        self.push_screen(
            PromptScreen("\U0001f4be  Dump memory to file", "./memory_dump.json"), done
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
