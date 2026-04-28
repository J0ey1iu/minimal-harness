"""Main TUI application."""

from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import Any

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Footer, ListView, Static

from minimal_harness.agent import (
    Agent,
    AgentRegistry,
    AgentRegistryProtocol,
    AgentRuntime,
    AgentRuntimeProtocol,
    Session,
)
from minimal_harness.client.built_in.buffer import StreamBuffer
from minimal_harness.client.built_in.config import DEFAULT_CONFIG
from minimal_harness.client.built_in.constants import (
    FLUSH_INTERVAL,
    J0EY1IU_QUOTES,
    THEMES,
)
from minimal_harness.client.built_in.context import AppContext
from minimal_harness.client.built_in.display import ChatDisplay
from minimal_harness.client.built_in.export_presenter import ExportPresenter
from minimal_harness.client.built_in.memory import PersistentMemory
from minimal_harness.client.built_in.modals import (
    AgentSelectScreen,
    ConfigScreen,
    ConfirmScreen,
    PromptScreen,
    SessionSelectScreen,
    ToolSelectScreen,
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
from minimal_harness.client.events import to_client_event
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
        self._agent_registry: AgentRegistryProtocol = AgentRegistry()
        self._runtime: AgentRuntimeProtocol = AgentRuntime(
            self._agent_registry, tool_registry=self.ctx.registry
        )
        self._current_session_id: str | None = None
        self.stop_event: asyncio.Event | None = None
        self.streaming = False
        self.buf = StreamBuffer()
        self._first = True
        self._chat_display: ChatDisplay | None = None
        self._exporter: ExportPresenter | None = None
        self._slash_handler: SlashCommandHandler | None = None
        self._session_manager: SessionManager | None = None
        self._runtime_session_ids: set[str] = set()
        self._watching_running: bool = False

    @property
    def config(self) -> dict[str, Any]:
        return self.ctx.config

    @property
    def memory(self) -> PersistentMemory | None:
        session = (
            self._runtime.get_session(self._current_session_id)
            if self._current_session_id
            else None
        )
        return session.memory if session else None

    @property
    def active_tools(self) -> list[StreamingTool]:
        session = (
            self._runtime.get_session(self._current_session_id)
            if self._current_session_id
            else None
        )
        if session:
            return session.tools
        from minimal_harness.client.built_in.config import load_agents_config

        agents = load_agents_config()
        default_name = self.ctx.config.get("default_agent", "general_assistant")
        for a in agents:
            if a.get("name") == default_name:
                tool_names = a.get("default_tools", [])
                return [
                    self.ctx.all_tools[n] for n in tool_names if n in self.ctx.all_tools
                ]
        return []

    @property
    def agent(self) -> Agent | None:
        session = (
            self._runtime.get_session(self._current_session_id)
            if self._current_session_id
            else None
        )
        return session.agent if session else None

    @property
    def _all_tools(self) -> dict[str, StreamingTool]:
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
        self._register_preset_agents()
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
        self._runtime.set_on_session_event(self._on_runtime_session_event)
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
            self._chat_display.tick(self.buf, self.streaming)
        self._poll_running_session()

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
        self._banner_widget.display = True
        self._chat.display = False

    def action_submit(self) -> None:
        d = self._chat_display
        if d is None:
            return
        text = self._input.text.strip()
        if not text or self.streaming:
            return
        self._input.text = ""
        if self._first:
            self._first = False
            d.clear_chat()
            self.buf.clear()
            self._banner_widget.display = False
            self._chat.display = True
        d.say(text, user=True)
        self._run(text)

    def _set_streaming(self, active: bool) -> None:
        self.streaming = active
        self._wrap.set_class(active, "streaming")
        self._input.disabled = active
        if not active:
            self._input.focus()

    @work(exclusive=True)
    async def _run(self, user_input: str) -> None:
        d = self._chat_display
        if d is None:
            return
        if self._current_session_id is None:
            self._start_with_default_agent()
        self.buf.clear()
        self.stop_event = asyncio.Event()
        self._set_streaming(True)
        try:
            async for event in self._runtime.run(
                user_input=[{"type": "text", "text": user_input}],
                stop_event=self.stop_event,
                session_id=self._current_session_id,
            ):
                if self.stop_event.is_set():
                    break
                d.handle_event(
                    to_client_event(event),
                    buf=self.buf,
                    memory=self.memory,
                )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            d.say(f"\nError: {e}", "bold bright_red")
        finally:
            if not self.buf._flushed:
                d.flush(self.buf)
            self.buf.clear()
            self.stop_event = None
            self._set_streaming(False)

    def action_interrupt(self) -> None:
        if self.streaming and self.stop_event is not None:
            d = self._chat_display
            if d is not None:
                self.stop_event.set()
                d.say("  \u2717 interrupted", "bold bright_red")

    def _create_session(
        self,
        agent_name: str = "general_assistant",
        system_prompt: str | None = None,
        default_tools: list[str] | None = None,
    ) -> Session:
        from minimal_harness.agent.simple import SimpleAgent

        self.ctx.memory = None
        self.ctx.rebuild(system_prompt=system_prompt)
        assert self.ctx.memory is not None

        base_tools = self.ctx.all_tools
        if default_tools is not None:
            tools = [base_tools[n] for n in default_tools if n in base_tools]
        else:
            tools = self.ctx.active_tools

        session = self._runtime.create_session(
            config=self.ctx.config,
            tools=tools,
            memory=self.ctx.memory,
            agent_factory=SimpleAgent,
            agent_name=agent_name,
            default_tools=default_tools,
        )
        if default_tools is not None:
            session.memory.selected_tools = default_tools
        self._current_session_id = session.session_id
        return session

    def _register_preset_agents(self) -> None:
        from minimal_harness.client.built_in.config import (
            SYSTEM_PROMPTS_DIR,
            load_agents_config,
            read_system_prompt,
        )

        agents = load_agents_config()
        if not agents:
            return
        llm = self.ctx._create_llm_provider(self.ctx.config)
        for a in agents:
            prompt_path = SYSTEM_PROMPTS_DIR / a["system_prompt"]
            system_prompt = read_system_prompt(prompt_path) or a.get("description", "")
            self._runtime.register_agent(
                name=a["name"],
                description=a["description"],
                system_prompt=system_prompt,
                llm_provider=llm,
                tools=self.ctx.active_tools,
                agent_factory=self.ctx._agent_factory,
                default_tools=a.get("default_tools") or [],
            )

    def _start_with_default_agent(self) -> None:
        from minimal_harness.client.built_in.config import (
            SYSTEM_PROMPTS_DIR,
            load_agents_config,
            read_system_prompt,
        )

        agents = load_agents_config()
        default_name = self.ctx.config.get("default_agent", "general_assistant")
        agent = self._get_default_agent(agents, default_name)
        if agent:
            prompt = read_system_prompt(
                SYSTEM_PROMPTS_DIR / agent["system_prompt"]
            ) or agent.get("description", "")
            self._create_session(
                agent_name=agent["name"],
                system_prompt=prompt,
                default_tools=agent.get("default_tools"),
            )
        else:
            self._create_session()

    @staticmethod
    def _get_default_agent(
        agents: list[dict[str, Any]],
        default_name: str = "general_assistant",
    ) -> dict[str, Any] | None:
        for a in agents:
            if a.get("name") == default_name:
                return a
        for a in agents:
            if a.get("name") == "general_assistant":
                return a
        return agents[0] if agents else None

    def _poll_running_session(self) -> None:
        sid = self._current_session_id
        if not sid:
            return
        is_running = self._runtime.is_session_running(sid)
        if not is_running and self._watching_running:
            self._watching_running = False
            self._set_streaming(False)
            d = self._chat_display
            if d is not None:
                if not self.buf._flushed:
                    d.flush(self.buf)
                self.buf.clear()
                d.say("\u2713 Session ready", "bold bright_green")
            return
        if is_running:
            self._watching_running = True
            session = self._runtime.get_session(sid)
            if session is not None and session.event_queue is not None:
                q = session.event_queue
                events: list[Any] = []
                while not q.empty():
                    try:
                        events.append(q.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                if events and self._chat_display is not None:
                    for event in events:
                        self._chat_display.handle_event(
                            to_client_event(event), buf=self.buf, memory=session.memory
                        )

    def _on_runtime_session_event(self, session_id: str) -> None:
        self._runtime_session_ids.add(session_id)

    def action_new(self) -> None:
        if self.streaming:
            return

        from minimal_harness.client.built_in.config import (
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
                self.buf.clear()
                self._first = True
                self._create_session(
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
        if self.streaming:
            return
        disk_sessions = PersistentMemory.list_sessions()
        disk_ids = {s["session_id"] for s in disk_sessions}

        runtime_sessions = []
        for sid in self._runtime_session_ids:
            if sid in disk_ids:
                continue
            s = self._runtime.get_session(sid)
            if s is not None:
                runtime_sessions.append(
                    {
                        "session_id": s.session_id,
                        "title": s.name or "Delegated Task",
                        "created_at": "",
                        "path": "",
                        "message_count": len(s.memory.get_all_messages()),
                        "agent_name": s.memory._agent_name,
                    }
                )

        sessions = runtime_sessions + disk_sessions

        def done(session_id: str | None) -> None:
            if not session_id or self._session_manager is None:
                return
            d = self._chat_display
            if d is None:
                return
            self._first = True

            session = self._runtime.get_session(session_id)
            if session is None:
                session = self._runtime.load_session(
                    session_id=session_id,
                    config=self.ctx.config,
                    tools=self.ctx.active_tools,
                    agent_factory=self.ctx._agent_factory,
                )
                if session and session.memory.selected_tools:
                    restored = [
                        self.ctx.all_tools[n]
                        for n in session.memory.selected_tools
                        if n in self.ctx.all_tools
                    ]
                    if restored:
                        session.rebuild(
                            self.ctx.config,
                            tools=restored,
                            agent_factory=self.ctx._agent_factory,
                        )

            if session:
                self._current_session_id = session_id
                success, inputs = self._session_manager.replay_session(
                    session,
                    clear_committed=self._clear_committed,
                    clear_buf=self.buf.clear,
                )
                if success:
                    self._first = False
                    self._banner_widget.display = False
                    self._chat.display = True
                    self._input._input_history = inputs  # type: ignore[attr-defined]
                    self._input.reset_history_index()  # type: ignore[attr-defined]
                    if self._runtime.is_session_running(session_id):
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
        if self.streaming:
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
        if self.streaming:
            return

        def done(result: dict | None) -> None:
            if result is None:
                return
            d = self._chat_display
            if d is None:
                return
            self.ctx.update_config(result)
            if (t := result.get("theme")) in THEMES:
                self.theme = t
                d.theme = t
            session = (
                self._runtime.get_session(self._current_session_id)
                if self._current_session_id
                else None
            )
            if session:
                session.rebuild(self.ctx.config, agent_factory=self.ctx._agent_factory)
            d.say("\u2713 Configuration saved", "bold bright_green")
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
            d = self._chat_display
            if d is None:
                return
            resolved = [
                self.ctx.all_tools[n] for n in chosen if n in self.ctx.all_tools
            ]
            self.ctx.select_tools(chosen)
            session = (
                self._runtime.get_session(self._current_session_id)
                if self._current_session_id
                else None
            )
            if session:
                session.rebuild(
                    self.ctx.config,
                    tools=resolved,
                    agent_factory=self.ctx._agent_factory,
                )
                session.memory.selected_tools = chosen
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
