"""Session lifecycle management — extracted from TUIApp."""

from __future__ import annotations

from typing import Any

from minimal_harness.agent.simple import SimpleAgent
from minimal_harness.client.built_in.buffer import StreamBuffer
from minimal_harness.client.built_in.config.agents import (
    SYSTEM_PROMPTS_DIR,
    load_agents_config,
    read_system_prompt,
)
from minimal_harness.client.built_in.context import AppContext
from minimal_harness.client.built_in.memory import PersistentMemory
from minimal_harness.client.built_in.session import TUISession
from minimal_harness.tool.base import Tool


class SessionController:
    """Owns session state, creation, handoff tracking, and metadata listing."""

    def __init__(
        self,
        runtime: Any,
        ctx: AppContext,
    ) -> None:
        self._runtime = runtime
        self._ctx = ctx
        self._current_session_id: str | None = None
        self._sessions: dict[str, TUISession] = {}
        self._handoff_target_ids: set[str] = set()
        self._watching_running: bool = False
        self.streaming = False
        self.buf = StreamBuffer()

    @property
    def current_session(self) -> TUISession | None:
        if self._current_session_id:
            return self._sessions.get(self._current_session_id)
        return None

    @property
    def current_session_id(self) -> str | None:
        return self._current_session_id

    @property
    def memory(self) -> PersistentMemory | None:
        session = self.current_session
        return session.memory if session else None

    @property
    def active_tools(self) -> list[Tool]:
        session = self.current_session
        if session:
            return session.tools
        agents = load_agents_config()
        default_name = self._ctx.config.get("default_agent", "general_assistant")
        for a in agents:
            if a.get("name") == default_name:
                tool_names = a.get("default_tools", [])
                return [
                    self._ctx.all_tools[n]
                    for n in tool_names
                    if n in self._ctx.all_tools
                ]
        return []

    @current_session_id.setter
    def current_session_id(self, value: str | None) -> None:
        self._current_session_id = value

    @property
    def handoff_target_ids(self) -> set[str]:
        return self._handoff_target_ids

    def set_streaming(self, active: bool) -> None:
        self.streaming = active

    def register_handoff(self, session_id: str) -> None:
        self._handoff_target_ids.add(session_id)

    def create_session(
        self,
        agent_name: str = "general_assistant",
        system_prompt: str | None = None,
        default_tools: list[str] | None = None,
    ) -> TUISession:
        self._ctx.memory = None
        self._ctx.rebuild(system_prompt=system_prompt)
        assert self._ctx.memory is not None

        base_tools = self._ctx.all_tools
        if default_tools is not None:
            tools = [base_tools[n] for n in default_tools if n in base_tools]
        else:
            tools = self._ctx.active_tools

        llm = self._ctx._create_llm_provider(self._ctx.config)
        agent = SimpleAgent(
            llm_provider=llm, tools=list(tools), memory=self._ctx.memory
        )

        session = TUISession(
            session_id=self._ctx.memory.session_id,
            name=self._ctx.memory.title or "New Chat",
            agent=agent,
            memory=self._ctx.memory,
            tools=list(tools),
        )
        if default_tools is not None:
            session.memory.selected_tools = default_tools
        self._sessions[session.session_id] = session
        self._current_session_id = session.session_id
        return session

    def interrupt(self) -> None:
        session = self.current_session
        if session is not None:
            session.interrupt()

    def get_session(self, session_id: str) -> TUISession | None:
        return self._sessions.get(session_id)

    def rebuild_current_session(
        self,
        llm_provider: Any,
        tools: list[Tool] | None = None,
        agent_factory: Any = None,
    ) -> None:
        session = self.current_session
        if session is not None:
            session.rebuild(
                llm_provider=llm_provider,
                tools=tools,
                agent_factory=agent_factory,
            )

    def register_preset_agents(self) -> None:
        agents = load_agents_config()
        if not agents:
            return
        llm = self._ctx._create_llm_provider(self._ctx.config)
        for a in agents:
            prompt_path = SYSTEM_PROMPTS_DIR / a["system_prompt"]
            system_prompt = read_system_prompt(prompt_path) or a.get("description", "")
            self._runtime.register_agent(
                name=a["name"],
                description=a["description"],
                system_prompt=system_prompt,
                llm_provider=llm,
                tools=self._ctx.active_tools,
                agent_factory=self._ctx._agent_factory,
                default_tools=a.get("default_tools") or [],
            )

    def start_with_default_agent(self) -> None:
        agents = load_agents_config()
        default_name = self._ctx.config.get("default_agent", "general_assistant")
        agent = self._get_default_agent(agents, default_name)
        if agent:
            prompt = read_system_prompt(
                SYSTEM_PROMPTS_DIR / agent["system_prompt"]
            ) or agent.get("description", "")
            self.create_session(
                agent_name=agent["name"],
                system_prompt=prompt,
                default_tools=agent.get("default_tools"),
            )
        else:
            self.create_session()

    def poll_handoff(self) -> tuple[bool, list[Any], Any]:
        """Returns (is_completed, events, target).

        When completed, the caller should update the display and reset streaming.
        When events are non-empty and target is set, the caller should dispatch them.
        """
        sid = self._current_session_id
        if not sid:
            return False, [], None

        is_running = self._runtime.is_background_task_running(sid)
        if not is_running and self._watching_running:
            self._watching_running = False
            return True, [], None

        if is_running:
            self._watching_running = True
            target = self._runtime.get_handoff_target(sid)
            if target is None:
                return False, [], None
            events = self._runtime.drain_handoff_events(sid)
            return False, events, target

        return False, [], None

    def load_session_from_disk(self, session_id: str) -> TUISession | None:
        session = self._sessions.get(session_id)
        if session is not None:
            return session

        target = self._runtime.get_handoff_target(session_id)
        if target is not None:
            session = TUISession(
                session_id=target.session_id,
                name=target.name,
                agent=target.agent,
                memory=target.memory,
                tools=list(target.tools),
            )
            self._sessions[session_id] = session
            return session

        memory = PersistentMemory.from_session(session_id)
        llm = self._ctx._create_llm_provider(self._ctx.config)
        tools = self._ctx.active_tools
        if memory.selected_tools:
            restored = [
                self._ctx.all_tools[n]
                for n in memory.selected_tools
                if n in self._ctx.all_tools
            ]
            if restored:
                tools = restored
        agent = SimpleAgent(llm_provider=llm, tools=tools, memory=memory)
        session = TUISession(
            session_id=session_id,
            name=memory.title or "Untitled",
            agent=agent,
            memory=memory,
            tools=list(tools),
        )
        self._sessions[session_id] = session
        return session

    def get_all_sessions_metadata(self) -> list[dict[str, Any]]:
        disk_sessions = PersistentMemory.list_sessions()
        disk_ids = {s["session_id"] for s in disk_sessions}

        handoff_sessions = []
        for sid in self._handoff_target_ids:
            if sid in disk_ids:
                continue
            target = self._runtime.get_handoff_target(sid)
            if target is not None:
                handoff_sessions.append(
                    {
                        "session_id": target.session_id,
                        "title": target.name or "Delegated Task",
                        "created_at": "",
                        "path": "",
                        "message_count": len(target.memory.get_all_messages()),
                        "agent_name": target.memory.agent_name,
                    }
                )

        tui_sessions = []
        for sid, s in self._sessions.items():
            if sid in disk_ids:
                continue
            tui_sessions.append(
                {
                    "session_id": s.session_id,
                    "title": s.name or "Chat",
                    "created_at": "",
                    "path": "",
                    "message_count": len(s.memory.get_all_messages()),
                    "agent_name": s.memory.agent_name,
                }
            )

        return handoff_sessions + tui_sessions + disk_sessions

    def switch_session(self, session_id: str) -> None:
        self._current_session_id = session_id

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
