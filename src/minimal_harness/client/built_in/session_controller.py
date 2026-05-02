"""Session lifecycle management — coordinates SessionFactory and AgentManager."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable, cast

from minimal_harness.agent.registry import AgentRegistryProtocol
from minimal_harness.agent.runtime import AgentRuntimeProtocol
from minimal_harness.client.built_in.agent_manager import AgentManager
from minimal_harness.client.built_in.buffer import StreamBuffer
from minimal_harness.client.built_in.config.agents import load_agents_config
from minimal_harness.client.built_in.context import AppContext
from minimal_harness.client.built_in.memory import PersistentMemory
from minimal_harness.client.built_in.session import ConversationSession, SessionStatus
from minimal_harness.client.built_in.session_factory import SessionFactory
from minimal_harness.tool.base import Tool

if TYPE_CHECKING:
    from minimal_harness.types import AgentEvent


class SessionController:
    """Coordinates session lifecycle: creation and run management."""

    def __init__(
        self,
        runtime: AgentRuntimeProtocol,
        agent_registry: AgentRegistryProtocol,
        ctx: AppContext,
    ) -> None:
        self._runtime = runtime
        self._ctx = ctx
        self._agent_registry = agent_registry
        self._factory = SessionFactory(ctx)
        self._agents = AgentManager(ctx, agent_registry)
        self._active_runs: dict[
            str, tuple[asyncio.Task, asyncio.Event, asyncio.Queue[AgentEvent | None]]
        ] = {}
        self._current_session_id: str | None = None
        self.streaming = False
        self._per_session_buf: dict[str, StreamBuffer] = {}
        self._per_session_streaming: dict[str, bool] = {}
        self._status_listeners: list[Callable[[str, SessionStatus], None]] = []

    @property
    def _sessions(self) -> dict[str, ConversationSession]:
        return self._agents.sessions

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
    def memory(self) -> PersistentMemory | None:
        session = self.current_session
        if session is None:
            return None
        return cast("PersistentMemory", session.memory)

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

    def create_session(
        self,
        agent_name: str = "general_assistant",
        system_prompt: str | None = None,
        default_tools: list[str] | None = None,
    ) -> ConversationSession:
        session = self._factory.create_session(
            agent_name=agent_name,
            system_prompt=system_prompt,
            default_tools=default_tools,
        )
        self._sessions[session.session_id] = session
        self._current_session_id = session.session_id
        return session

    def load_session_from_disk(self, session_id: str) -> ConversationSession | None:
        existing = self._sessions.get(session_id)
        if existing is not None:
            return existing
        session = self._factory.load_session_from_disk(session_id)
        if session is None:
            return None
        self._sessions[session_id] = session
        return session

    def rebuild_current_session(
        self,
        llm_provider: Any,
        tools: list[Tool] | None = None,
        agent_factory: Any = None,
    ) -> None:
        session = self.current_session
        if session is not None:
            self._factory.rebuild_current_session(
                session, llm_provider, tools, agent_factory
            )

    def register_preset_agents(self) -> None:
        self._agents.register_preset_agents()

    def start_with_default_agent(self) -> None:
        self._agents.start_with_default_agent(self.create_session)

    def interrupt(self) -> None:
        session = self.current_session
        if session is not None:
            session.interrupt()
        if self._current_session_id and self._current_session_id in self._active_runs:
            _, stop_event, _ = self._active_runs[self._current_session_id]
            stop_event.set()

    def set_streaming(self, active: bool) -> None:
        self.streaming = active
        sid = self._current_session_id
        if sid:
            self._per_session_streaming[sid] = active

    def is_session_streaming(self, session_id: str) -> bool:
        return self._per_session_streaming.get(session_id, False)

    def get_buf(self, session_id: str) -> StreamBuffer:
        if session_id not in self._per_session_buf:
            self._per_session_buf[session_id] = StreamBuffer()
        return self._per_session_buf[session_id]

    def start_run(
        self, session: ConversationSession, user_input: str
    ) -> tuple[asyncio.Event, asyncio.Queue[AgentEvent | None]] | None:
        if session.session_id in self._active_runs:
            return None
        task, stop_event, event_queue = self._runtime.run(
            agent=session.agent,
            memory=session.memory,
            tools=session.tools,
            user_input=[{"type": "text", "text": user_input}],
            agent_name=session.name,
        )
        self._active_runs[session.session_id] = (task, stop_event, event_queue)
        self._per_session_streaming[session.session_id] = True
        self._notify_status_changed(session.session_id, SessionStatus.RUNNING)
        return stop_event, event_queue

    def add_status_listener(
        self, listener: Callable[[str, SessionStatus], None]
    ) -> None:
        self._status_listeners.append(listener)

    def remove_status_listener(
        self, listener: Callable[[str, SessionStatus], None]
    ) -> None:
        self._status_listeners.remove(listener)

    def _notify_status_changed(self, session_id: str, status: SessionStatus) -> None:
        for listener in list(self._status_listeners):
            listener(session_id, status)

    def get_session_status(self, session_id: str) -> SessionStatus:
        return (
            SessionStatus.RUNNING
            if session_id in self._active_runs
            else SessionStatus.IDLE
        )

    def end_run(self, session_id: str) -> None:
        self._active_runs.pop(session_id, None)
        self._per_session_streaming.pop(session_id, None)
        self._notify_status_changed(session_id, SessionStatus.IDLE)

    def drain_session_events(self, session_id: str) -> tuple[list[AgentEvent], bool]:
        if session_id not in self._active_runs:
            return [], False

        _, _, event_queue = self._active_runs[session_id]
        events: list[AgentEvent] = []
        done = False
        while True:
            try:
                event = event_queue.get_nowait()
                if event is None:
                    done = True
                    break
                events.append(event)
            except asyncio.QueueEmpty:
                break

        if done:
            self._active_runs.pop(session_id, None)
            self._per_session_streaming.pop(session_id, None)
            self._notify_status_changed(session_id, SessionStatus.IDLE)

        return events, done

    def get_all_sessions_metadata(self) -> list[dict[str, Any]]:
        disk_sessions = PersistentMemory.list_sessions()
        disk_ids = {s["session_id"] for s in disk_sessions}

        memory_sessions = []
        for sid, s in self._sessions.items():
            if sid in disk_ids:
                continue
            mem = cast("PersistentMemory", s.memory)
            memory_sessions.append(
                {
                    "session_id": s.session_id,
                    "title": s.name or "Chat",
                    "created_at": mem.created_at,
                    "path": "",
                    "message_count": len(mem.get_all_messages()),
                    "agent_name": mem.agent_name,
                    "status": self.get_session_status(sid).name.lower(),
                }
            )

        for ds in disk_sessions:
            ds.setdefault(
                "status", self.get_session_status(ds["session_id"]).name.lower()
            )

        combined = memory_sessions + disk_sessions
        combined.sort(key=lambda s: s.get("created_at") or "", reverse=True)
        return combined

    def switch_session(self, session_id: str) -> None:
        self._current_session_id = session_id
