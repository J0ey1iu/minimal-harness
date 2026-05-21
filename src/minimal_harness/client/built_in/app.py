"""Main TUI application."""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, ListView, Static

from minimal_harness.agent.registry import AgentRegistry
from minimal_harness.agent.runtime import AgentRuntime
from minimal_harness.client.built_in.actions.config import (
    action_config as _action_config,
)
from minimal_harness.client.built_in.actions.dump import action_dump as _action_dump
from minimal_harness.client.built_in.actions.interrupt import (
    action_interrupt as _action_interrupt,
)
from minimal_harness.client.built_in.actions.new import action_new as _action_new
from minimal_harness.client.built_in.actions.quit import (
    action_request_quit as _action_request_quit,
)
from minimal_harness.client.built_in.actions.reload import (
    action_reload as _action_reload,
)
from minimal_harness.client.built_in.actions.sessions import (
    action_sessions as _action_sessions,
)
from minimal_harness.client.built_in.actions.share import action_share as _action_share
from minimal_harness.client.built_in.actions.tools import action_tools as _action_tools
from minimal_harness.client.built_in.at_handler import AtCommandHandler
from minimal_harness.client.built_in.config import DEFAULT_CONFIG, load_config
from minimal_harness.client.built_in.constants import (
    FLUSH_INTERVAL,
    J0EY1IU_QUOTES,
    THEMES,
)
from minimal_harness.client.built_in.context import AppContext
from minimal_harness.client.built_in.display import ChatDisplay
from minimal_harness.tool.built_in.runtime_tools import register_runtime_tools
from minimal_harness.client.built_in.error_handler import CapturedError, ErrorHandler
from minimal_harness.client.built_in.export_presenter import ExportPresenter
from minimal_harness.client.built_in.messages import (
    AtCommandHide,
    AtCommandNavigateDown,
    AtCommandNavigateUp,
    AtCommandSelect,
    AtCommandShow,
    ChatInputDump,
    ChatInputSubmit,
    ErrorClicked,
    SlashCommandHide,
    SlashCommandNavigateDown,
    SlashCommandNavigateUp,
    SlashCommandSelect,
    SlashCommandShow,
)
from minimal_harness.client.built_in.modals import CopySelectScreen, ErrorScreen
from minimal_harness.client.built_in.session import SessionStatus
from minimal_harness.client.built_in.session_controller import SessionController
from minimal_harness.client.built_in.session_replayer import SessionReplayer
from minimal_harness.client.built_in.slash_handler import SlashCommandHandler
from minimal_harness.client.built_in.widgets import (
    Banner,
    ChatInput,
    ErrorNotification,
    SessionNotification,
    SessionNotificationClicked,
)
from minimal_harness.memory import Memory
from minimal_harness.tool.registry import ToolRegistry
from minimal_harness.types import AgentEnd, AgentEvent, ToolMetadata

_CSS_PATH = Path(__file__).parent / "app.tcss"


def _get_built_in_tool_names() -> set[str]:
    from minimal_harness.tool.registry import get_builtin_tool_names

    return get_builtin_tool_names()


def _resolve_dn(tool):
    resolve = getattr(tool, "resolve_display_name", None)
    return resolve() if resolve else (getattr(tool, "display_name", None) or tool.name)


class TUIApp(App):
    TITLE = "Minimal Harness"
    ENABLE_COMMAND_PALETTE = False

    CSS_PATH = _CSS_PATH

    BINDINGS = [
        Binding("ctrl+o", "config", "Config"),
        Binding("ctrl+t", "tools", "Tools"),
        Binding("escape", "interrupt", "Interrupt", show=False),
        Binding("ctrl+y", "copy_last_response", "Copy", priority=True),
        Binding("ctrl+c", "request_quit", "Quit"),
    ]

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        super().__init__()
        self.ctx = AppContext(config=config, registry=registry)
        self._agent_registry = AgentRegistry()
        self._runtime = AgentRuntime(
            agent_registry=self._agent_registry,
            session_store=self.ctx.session_store,
            tool_registry=self.ctx.registry,
            llm_provider_factory=lambda: self.ctx.create_llm_provider(),
        )
        self._ctrl = SessionController(self._runtime, self._agent_registry, self.ctx)
        self._first = True
        self._chat_display: ChatDisplay | None = None
        self._exporter: ExportPresenter | None = None
        self._slash_handler: SlashCommandHandler | None = None
        self._at_handler: AtCommandHandler | None = None
        self._session_manager: SessionReplayer | None = None

    @property
    def config(self) -> dict[str, Any]:
        return self.ctx.config

    async def get_memory(self) -> Memory | None:
        return await self._ctrl.get_memory()

    @property
    def active_tools(self) -> list[ToolMetadata]:
        return []

    async def get_active_tools(self) -> list[ToolMetadata]:
        return await self._ctrl.get_active_tools()

    @property
    def _all_tools(self) -> dict[str, ToolMetadata]:
        return self.ctx.all_tools

    def compose(self) -> ComposeResult:
        with Horizontal(id="top-bar"):
            yield Static("", id="top-bar-title")
            yield Static("", id="top-bar-version")
        yield SessionNotification("", "", id="session-notification")
        yield ErrorNotification(id="error-notification")
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

    async def on_mount(self) -> None:
        ErrorHandler().install()
        ErrorHandler().add_listener(self._on_error_captured)
        theme = self.ctx.config.get("theme", DEFAULT_CONFIG["theme"])
        if theme in THEMES:
            self.theme = theme
        await self.ctx.rebuild()
        await register_runtime_tools(
            agent_registry=self._agent_registry,
            session_store=self.ctx.session_store,
            tool_registry=self.ctx.registry,
            run_fn=self._runtime.run,
        )
        await self._ctrl.register_preset_agents()
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
        self._at_handler = AtCommandHandler(
            suggestion_list=self._suggestion_list,
            input_widget=self._input,
            get_input_text=lambda: self._input.text,
            set_input_text=lambda t: setattr(self._input, "text", t),
        )
        self._session_manager = SessionReplayer(
            ctx=self.ctx,
            display=d,
            clear_input=lambda: setattr(self._input, "text", ""),
            show_banner=lambda: self._banner(),
        )
        self.set_interval(FLUSH_INTERVAL, self._tick)
        self._ctrl.add_status_listener(self._on_session_status_changed)
        self._input.focus()
        self._chat.display = False
        await self._banner()
        self._top_bar_title = self.query_one("#top-bar-title", Static)
        self._top_bar_version = self.query_one("#top-bar-version", Static)
        await self._ctrl.start_with_default_agent()
        self._update_top_bar()

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

    def _handle_exception(self, error: Exception) -> None:
        err = CapturedError.from_exc_info(
            type(error), error, error.__traceback__, source="tui"
        )
        ErrorHandler().capture(err)

    def _handle_agent_error(self, error_text: str) -> None:
        err = CapturedError(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            formatted=error_text,
            brief=error_text,
            source="agent",
        )
        ErrorHandler().capture(err)

    def _on_error_captured(self, error) -> None:
        notification = self.query_one("#error-notification", ErrorNotification)
        notification.show_error(error.brief)

    def on_error_clicked(self, _event: ErrorClicked) -> None:
        handler = ErrorHandler()
        errors = [
            {
                "timestamp": e.timestamp,
                "brief": e.brief,
                "formatted": e.formatted,
                "source": e.source,
            }
            for e in handler.errors
        ]
        self.push_screen(ErrorScreen(errors))

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        return text[: max_len - 1] + "\u2026"

    @staticmethod
    def _app_version() -> str:
        try:
            from importlib.metadata import version

            return f"v{version('minimal-harness')}"
        except Exception:
            return "unknown version"

    def _update_top_bar(self) -> None:
        sess = self._ctrl.current_session
        name = sess.session.agent_name if sess else ""
        status_text = ""
        session_title = ""
        if sess:
            sid = sess.session.memory_id
            if self._ctrl.get_session_status(sid) == SessionStatus.RUNNING:
                status_text = "  ● Running"
            else:
                status_text = "  ○ Idle"
            raw_title = sess.session.title
            if raw_title:
                session_title = f"  \u201c{self._truncate(raw_title, 40)}\u201d"
        if name:
            self._top_bar_title.update(
                Text.assemble(
                    (f"  Minimal Harness — {name}", "bold"),
                    (session_title, "italic"),
                    (
                        status_text,
                        "bold bright_green" if "●" in status_text else "dim italic",
                    ),
                )
            )
        else:
            self._top_bar_title.update(Text("  Minimal Harness  ", style="bold"))
        self._top_bar_version.update(Text(self._app_version(), style="dim italic"))

    async def _on_session_status_changed(
        self, session_id: str, status: SessionStatus
    ) -> None:
        if session_id == self._ctrl.current_session_id:
            self._update_top_bar()

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

    def on_at_command_show(self, event: AtCommandShow) -> None:
        if self._at_handler:
            self._at_handler.on_at_command_show(event.text)

    def on_at_command_hide(self, event: AtCommandHide) -> None:
        if self._at_handler:
            self._at_handler.on_at_command_hide()

    def on_at_command_navigate_up(self, event: AtCommandNavigateUp) -> None:
        if self._at_handler:
            self._at_handler.on_at_command_navigate_up()

    def on_at_command_navigate_down(self, event: AtCommandNavigateDown) -> None:
        if self._at_handler:
            self._at_handler.on_at_command_navigate_down()

    def on_at_command_select(self, event: AtCommandSelect) -> None:
        if self._at_handler:
            self._at_handler.on_at_command_select()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if self._slash_handler:
            self._slash_handler.on_list_view_selected(event.list_view.index)
        if self._at_handler:
            self._at_handler.on_list_view_selected(event.list_view.index)

    def on_chat_input_submit(self, event: ChatInputSubmit) -> None:
        self.action_submit()

    def on_chat_input_dump(self, event: ChatInputDump) -> None:
        self.action_dump()

    async def _tick(self) -> None:
        try:
            await self._drain_session_events()
            await self._check_background_completions()
            if self._chat_display is not None:
                self._render_streaming()
        except Exception as e:
            from minimal_harness.client.built_in.error_handler import CapturedError

            err = CapturedError.from_exc_info(
                type(e), e, e.__traceback__, source="_tick"
            )
            ErrorHandler().capture(err)

    async def _banner(self, show: bool = True) -> None:
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
            ext_names = ", ".join(_resolve_dn(t) for t in ext)
            lines.append(
                Text(
                    f"Loaded {len(ext)} external tool(s): " + ext_names,
                    style="dim",
                )
            )
        active_tools = await self.get_active_tools()
        active = ", ".join(_resolve_dn(t) for t in active_tools) or "(none)"
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
        if not text:
            return
        sid = self._ctrl.current_session_id
        if sid and self._ctrl.get_session_status(sid) == SessionStatus.RUNNING:
            return
        self._input.text = ""
        if self._first:
            self._first = False
            d.clear_chat()
            if sid:
                self._ctrl.get_buf(sid).clear()
            self._banner_widget.display = False
            self._chat.display = True
        d.say(text, user=True)
        self._run(text)

    def _set_streaming(self, active: bool) -> None:
        self._ctrl.set_streaming(active)
        self._wrap.set_class(active, "streaming")
        top_bar = self.query_one("#top-bar")
        if active:
            top_bar.add_class("streaming-top-bar")
        else:
            top_bar.remove_class("streaming-top-bar")
        if not active:
            self._input.focus()

    def _render_streaming(self) -> None:
        d = self._chat_display
        if not d:
            return
        sid = self._ctrl.current_session_id
        if sid:
            buf = self._ctrl.get_buf(sid)
            streaming = self._ctrl.is_session_streaming(sid)
            d.tick(buf, streaming)

    @work(exclusive=False)
    async def _run(self, user_input: str) -> None:
        """Start an agent run for the current session. Events are drained by _tick."""
        try:
            if self._ctrl.current_session_id is None:
                await self._ctrl.start_with_default_agent()
                self._update_top_bar()
            sess = self._ctrl.current_session
            if sess is None:
                return
            sid = sess.session.memory_id
            result = await self._ctrl.start_run(sess, user_input)
            if result is None:
                return
            self._ctrl.get_buf(sid).clear()
            sess.reset()
            self._set_streaming(True)
            await self._tick()
        except Exception as e:
            from minimal_harness.client.built_in.error_handler import CapturedError

            err = CapturedError.from_exc_info(
                type(e), e, e.__traceback__, source="_run"
            )
            ErrorHandler().capture(err)

    def action_interrupt(self) -> None:
        _action_interrupt(self)

    def action_copy_last_response(self) -> None:
        d = self._chat_display
        if d is None:
            return
        texts = d.get_assistant_texts()
        if not texts:
            self.notify("No assistant messages to copy", severity="warning")
            return
        if len(texts) == 1:
            self.copy_to_clipboard(texts[0][1])
            self.notify("Copied to clipboard")
            return
        self.push_screen(CopySelectScreen(texts), self._on_copy_selected)

    def _on_copy_selected(self, text: str | None) -> None:
        if text:
            self.copy_to_clipboard(text)
            self.notify("Copied to clipboard")

    async def _drain_session_events(self) -> None:
        try:
            d = self._chat_display
            sid = self._ctrl.current_session_id

            if sid:
                events, done = await self._ctrl.drain_session_events(sid)
                if events and d is not None:
                    buf = self._ctrl.get_buf(sid)
                    agent_ends: list[AgentEvent] = []
                    for event in events:
                        if isinstance(event, AgentEnd):
                            agent_ends.append(event)
                            continue
                        d.handle_event(
                            event,
                            buf=buf,
                        )
                    if done and not buf.flushed:
                        d.flush(buf)
                    for event in agent_ends:
                        d.handle_event(
                            event,
                            buf=buf,
                        )
                if done:
                    self._set_streaming(False)
                    if d is not None:
                        buf = self._ctrl.get_buf(sid)
                        buf.clear()
                    if sid:
                        await self._ctrl.end_run(sid)
                    error = self._ctrl.pop_session_error(sid) if sid else None
                    if error:
                        self._handle_agent_error(error)
        except Exception as e:
            err = CapturedError.from_exc_info(
                type(e), e, e.__traceback__, source="_drain_session_events"
            )
            ErrorHandler().capture(err)

    async def _check_background_completions(self) -> None:
        sid = self._ctrl.current_session_id
        completed = await self._ctrl.poll_background_completions(sid)
        for session_id in completed:
            session = self._ctrl.get_all_sessions().get(session_id)
            if session:
                self._show_session_notification(
                    session_id, "", session.session.agent_name
                )
                error = self._ctrl.pop_session_error(session_id)
                if error:
                    self._handle_agent_error(error)

    def _show_session_notification(
        self, session_id: str, title: str, agent_name: str
    ) -> None:
        notification = self.query_one("#session-notification", SessionNotification)
        if notification._timer is not None:
            notification._timer.stop()
        notification._session_id = session_id
        parts: list[tuple[str, str]] = [("\u2713 ", "bold bright_green")]
        if title:
            parts.append((f'"{title}" ', "bold"))
        parts.append((f"{agent_name} finished", "bold"))
        parts.append(("  (click to switch)", "dim"))
        notification.update(Text.assemble(*parts))
        notification.add_class("visible")
        notification._timer = self.set_timer(10, self._dismiss_session_notification)

    def _dismiss_session_notification(self) -> None:
        notification = self.query_one("#session-notification", SessionNotification)
        notification.remove_class("visible")

    async def on_session_notification_clicked(
        self, event: SessionNotificationClicked
    ) -> None:
        await self._switch_to_session(event.session_id)

    async def _switch_to_session(self, session_id: str) -> None:
        if self._session_manager is None or self._chat_display is None:
            return
        d = self._chat_display
        self._dismiss_session_notification()
        session = await self._ctrl.load_session_from_disk(session_id)
        if session:
            self._ctrl.switch_session(session_id)
            self._update_top_bar()
            buf = self._ctrl.get_buf(session_id)
            success, inputs = await self._session_manager.replay_session(
                session,
                clear_committed=self._clear_committed,
                clear_buf=buf.clear,
            )
            if success:
                self._first = False
                self._banner_widget.display = False
                self._chat.display = True
                self._input.input_history = inputs
                self._input.reset_history_index()
                if self._ctrl.is_session_running(session_id):
                    events, finished = await self._ctrl.drain_session_events(session_id)
                    if events and d:
                        for event in events:
                            d.handle_event(
                                event,
                                buf=buf,
                            )
                    if not finished:
                        self._set_streaming(True)
                    else:
                        if not buf.flushed:
                            d.flush(buf)
                        buf.clear()
                        await self._ctrl.end_run(session_id)

    def action_new(self) -> None:
        _action_new(self)

    def action_reload(self) -> None:
        _action_reload(self)

    def action_sessions(self) -> None:
        _action_sessions(self)

    def _clear_committed(self) -> None:
        if self._chat_display is not None:
            self._chat_display.clear_chat()

    def action_share(self) -> None:
        _action_share(self)

    def action_config(self) -> None:
        _action_config(self)

    def action_tools(self) -> None:
        _action_tools(self)

    def action_dump(self) -> None:
        _action_dump(self)

    def action_request_quit(self) -> None:
        _action_request_quit(self)


def main() -> None:

    config = load_config()
    TUIApp(config=config).run()
