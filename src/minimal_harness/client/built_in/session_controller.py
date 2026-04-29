"""Session lifecycle management — coordinates AgentManager and RunManager."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any

from minimal_harness.agent.registry import AgentRegistryProtocol
from minimal_harness.agent.runtime import AgentRuntimeProtocol
from minimal_harness.agent.simple import SimpleAgent
from minimal_harness.client.built_in.agent_manager import AgentManager
from minimal_harness.client.built_in.buffer import StreamBuffer
from minimal_harness.client.built_in.config.agents import load_agents_config
from minimal_harness.client.built_in.context import AppContext
from minimal_harness.client.built_in.memory import PersistentMemory
from minimal_harness.client.built_in.run_manager import RunManager
from minimal_harness.client.built_in.session import ConversationSession
from minimal_harness.tool.base import Tool

if TYPE_CHECKING:
    from minimal_harness.memory import Memory
    from minimal_harness.types import AgentEvent


class SessionController:
    """Owns session state, creation, handoff tracking, and metadata listing."""

    def __init__(
        self,
        runtime: AgentRuntimeProtocol,
        agent_registry: AgentRegistryProtocol,
        ctx: AppContext,
    ) -> None:
        self._ctx = ctx
        self._agents = AgentManager(ctx, agent_registry)
        self._runs = RunManager(runtime)
        self._current_session_id: str | None = None
        self.streaming = False
        self.buf = StreamBuffer()

    @property
    def _sessions(self) -> dict[str, ConversationSession]:
        return self._agents.sessions

    @property
    def _preset_session_ids(self) -> set[str]:
        return self._agents.preset_session_ids

    @property
    def _active_runs(
        self,
    ) -> dict[
        str, tuple[asyncio.Task, asyncio.Event, asyncio.Queue[AgentEvent | None]]
    ]:
        return self._runs.active_runs

    @property
    def _foreground_session_id(self) -> str | None:
        return self._runs.foreground_session_id

    @_foreground_session_id.setter
    def _foreground_session_id(self, value: str | None) -> None:
        self._runs.foreground_session_id = value

    @property
    def current_session(self) -> ConversationSession | None:
        if self._current_session_id:
            return self._sessions.get(self._current_session_id)
        return None

    @property
    def current_session_id(self) -> str | None:
        return self._current_session_id

    @current_session_id.setter
    def current_session_id(self, value: str | None) -> None:
        self._current_session_id = value

    @property
    def memory(self) -> Memory | None:
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

    @property
    def handoff_target_ids(self) -> set[str]:
        return self._runs.handoff_target_ids(self._current_session_id)

    def register_handoff_run(
        self,
        agent_name: str,
        task: asyncio.Task,
        stop_event: asyncio.Event,
        queue: asyncio.Queue[AgentEvent | None],
    ) -> None:
        for sid, s in self._sessions.items():
            if s.name == agent_name:
                self._active_runs[sid] = (task, stop_event, queue)
                setattr(s.memory, "created_at", datetime.now().isoformat())
                return
        session = self.create_session(agent_name=agent_name)
        self._active_runs[session.session_id] = (task, stop_event, queue)

    def set_streaming(self, active: bool) -> None:
        self.streaming = active

    def create_session(
        self,
        agent_name: str = "general_assistant",
        system_prompt: str | None = None,
        default_tools: list[str] | None = None,
    ) -> ConversationSession:
        self._ctx.rebuild()
        memory = PersistentMemory(system_prompt=system_prompt or "")
        memory.agent_name = agent_name

        base_tools = self._ctx.all_tools
        if default_tools is not None:
            tools = [base_tools[n] for n in default_tools if n in base_tools]
        else:
            tools = self._ctx.active_tools

        llm = self._ctx._create_llm_provider(self._ctx.config)
        agent = SimpleAgent(llm_provider=llm, tools=list(tools), memory=memory)

        session = ConversationSession(
            session_id=memory.session_id,
            agent=agent,
            memory=memory,
            tools=list(tools),
            name=agent_name,
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
        if self._current_session_id and self._current_session_id in self._active_runs:
            _, stop_event, _ = self._active_runs[self._current_session_id]
            stop_event.set()

    def rebuild_current_session(
        self,
        llm_provider: Any,
        tools: list[Tool] | None = None,
        agent_factory: Any = None,
    ) -> None:
        session = self.current_session
        if session is not None:
            if tools is not None:
                session.tools = list(tools)
            factory = agent_factory or SimpleAgent
            session.agent = factory(
                llm_provider=llm_provider, tools=session.tools, memory=session.memory
            )

    def register_preset_agents(self) -> None:
        self._agents.register_preset_agents()

    def start_with_default_agent(self) -> None:
        self._agents.start_with_default_agent(self.create_session)

    def drain_session_events(self, session_id: str) -> tuple[list[AgentEvent], bool]:
        return self._runs.drain_session_events(session_id)

    def poll_handoff_completion(self) -> bool:
        return self._runs.poll_handoff_completion(
            self.handoff_target_ids, self._current_session_id
        )

    def start_run(
        self, session: ConversationSession, user_input: str
    ) -> tuple[asyncio.Event, asyncio.Queue[AgentEvent | None]]:
        return self._runs.start_run(session, user_input)

    def end_run(self, session_id: str) -> None:
        self._runs.end_run(session_id)

    def load_session_from_disk(self, session_id: str) -> ConversationSession | None:
        session = self._sessions.get(session_id)
        if session is not None:
            return session

        try:
            memory = PersistentMemory.from_session(session_id)
        except (FileNotFoundError, ValueError):
            return None

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
        session = ConversationSession(
            session_id=session_id,
            agent=agent,
            memory=memory,
            tools=list(tools),
            name=memory.agent_name or memory.title or "Untitled",
        )
        self._sessions[session_id] = session
        return session

    def get_all_sessions_metadata(self) -> list[dict[str, Any]]:
        disk_sessions = PersistentMemory.list_sessions()
        disk_ids = {s["session_id"] for s in disk_sessions}

        memory_sessions = []
        for sid, s in self._sessions.items():
            if sid in disk_ids:
                continue
            if sid in self._preset_session_ids and sid not in self.handoff_target_ids:
                continue
            memory_sessions.append(
                {
                    "session_id": s.session_id,
                    "title": s.name or "Chat",
                    "created_at": getattr(s.memory, "created_at", ""),
                    "path": "",
                    "message_count": len(s.memory.get_all_messages()),
                    "agent_name": getattr(s.memory, "agent_name", ""),
                }
            )

        combined = memory_sessions + disk_sessions
        combined.sort(key=lambda s: s.get("created_at") or "", reverse=True)
        return combined

    def switch_session(self, session_id: str) -> None:
        self._current_session_id = session_id
