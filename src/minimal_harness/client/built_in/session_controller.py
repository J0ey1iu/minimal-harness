"""Session lifecycle management — coordinates SessionFactory and AgentManager."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from minimal_harness.agent.registry import AgentRegistryProtocol
from minimal_harness.agent.runtime import AgentRuntimeProtocol
from minimal_harness.client.built_in.agent_manager import AgentManager
from minimal_harness.client.built_in.buffer import StreamBuffer
from minimal_harness.client.built_in.context import AppContext
from minimal_harness.client.built_in.session import ConversationSession, SessionStatus
from minimal_harness.client.built_in.session_factory import SessionFactory

if TYPE_CHECKING:
    from minimal_harness.memory import Memory
    from minimal_harness.tool.base import Tool
    from minimal_harness.types import AgentEvent


class SessionController:
    """Coordinates session lifecycle: creation and run management.

    Uses Layer 2 abstractions (AgentRegistry, DiskMemoryStore, ToolRegistry)
    exclusively. Never directly instantiates or uses Layer 1 types.
    """

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
        self._status_listeners: list[
            Callable[[str, SessionStatus], Awaitable[None]]
        ] = []

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

    async def get_memory(self, session_id: str | None = None) -> Memory | None:
        sid = session_id or self._current_session_id
        if sid is None:
            return None
        session = self._sessions.get(sid)
        if session is None:
            return None
        return await self._ctx.memory_store.get_memory(session.memory_id)

    async def get_active_tools(self) -> list[Tool]:
        session = self.current_session
        if session and session.tool_names:
            return [
                t
                for n in session.tool_names
                if (t := self._ctx.all_tools.get(n)) is not None
            ]
        default_name = self._ctx.config.get("default_agent", "general_assistant")
        metadata = await self._agent_registry.get(default_name)
        if metadata:
            return [
                self._ctx.all_tools[n]
                for n in metadata.tool_names
                if n in self._ctx.all_tools
            ]
        return []

    async def create_session(
        self,
        agent_name: str = "general_assistant",
        default_tools: list[str] | None = None,
    ) -> ConversationSession:
        session = await self._factory.create_session(
            agent_name=agent_name,
            default_tools=default_tools,
        )
        self._sessions[session.session_id] = session
        self._current_session_id = session.session_id
        return session

    async def load_session_from_disk(
        self, session_id: str
    ) -> ConversationSession | None:
        existing = self._sessions.get(session_id)
        if existing is not None:
            return existing
        session = await self._factory.load_session_from_disk(session_id)
        if session is None:
            return None
        self._sessions[session_id] = session
        return session

    def rebuild_current_session(
        self,
        tools: list[Tool] | None = None,
    ) -> None:
        session = self.current_session
        if session is not None:
            self._factory.rebuild_current_session(session, tools)

    async def register_preset_agents(self) -> None:
        await self._agents.register_preset_agents()

    async def start_with_default_agent(self) -> None:
        await self._agents.start_with_default_agent(self.create_session)

    def interrupt(self) -> None:
        session = self.current_session
        if session is not None:
            session.interrupt()
        if self._current_session_id and self._current_session_id in self._active_runs:
            task, stop_event, _ = self._active_runs[self._current_session_id]
            task.cancel()
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

    async def start_run(
        self, session: ConversationSession, user_input: str
    ) -> tuple[asyncio.Event, asyncio.Queue[AgentEvent | None]] | None:
        if session.session_id in self._active_runs:
            return None
        task, stop_event, event_queue = await self._runtime.run(
            user_input=[{"type": "text", "text": user_input}],
            agent_metadata_id=session.agent_metadata_id,
            memory_id=session.memory_id,
            tool_names=session.tool_names if session.tool_names else None,
            context={"agent_name": session.name},
        )
        self._active_runs[session.session_id] = (task, stop_event, event_queue)
        self._per_session_streaming[session.session_id] = True
        await self._notify_status_changed(session.session_id, SessionStatus.RUNNING)
        return stop_event, event_queue

    def add_status_listener(
        self, listener: Callable[[str, SessionStatus], Awaitable[None]]
    ) -> None:
        self._status_listeners.append(listener)

    def remove_status_listener(
        self, listener: Callable[[str, SessionStatus], Awaitable[None]]
    ) -> None:
        self._status_listeners.remove(listener)

    async def _notify_status_changed(
        self, session_id: str, status: SessionStatus
    ) -> None:
        for listener in list(self._status_listeners):
            await listener(session_id, status)

    def get_session_status(self, session_id: str) -> SessionStatus:
        return (
            SessionStatus.RUNNING
            if session_id in self._active_runs
            else SessionStatus.IDLE
        )

    def is_session_running(self, session_id: str) -> bool:
        return session_id in self._active_runs

    def get_all_sessions(self) -> dict[str, ConversationSession]:
        return dict(self._sessions)

    async def end_run(self, session_id: str) -> None:
        self._active_runs.pop(session_id, None)
        self._per_session_streaming.pop(session_id, None)
        await self._notify_status_changed(session_id, SessionStatus.IDLE)

    async def poll_background_completions(
        self, current_session_id: str | None
    ) -> list[str]:
        completed: list[str] = []
        for sid in list(self._active_runs.keys()):
            if sid == current_session_id:
                continue
            _, _, event_queue = self._active_runs[sid]
            done = False
            while True:
                try:
                    event = event_queue.get_nowait()
                    if event is None:
                        done = True
                        break
                except asyncio.QueueEmpty:
                    break
            if done:
                self._active_runs.pop(sid, None)
                self._per_session_streaming.pop(sid, None)
                await self._notify_status_changed(sid, SessionStatus.IDLE)
                completed.append(sid)
        return completed

    async def drain_session_events(
        self, session_id: str
    ) -> tuple[list[AgentEvent], bool]:
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
            await self._notify_status_changed(session_id, SessionStatus.IDLE)

        return events, done

    async def get_all_sessions_metadata(self) -> list[dict[str, Any]]:
        store = self._ctx.memory_store
        disk_sessions = await store.list_sessions()
        disk_ids = {s["memory_id"] for s in disk_sessions}

        memory_sessions = []
        for sid, s in self._sessions.items():
            if s.memory_id in disk_ids:
                continue
            mem = await self._ctx.memory_store.get_memory(s.memory_id)
            title = getattr(mem, "title", None) if mem else None
            created_at = getattr(mem, "created_at", "") if mem else ""
            msg_count = len(mem.get_all_messages()) if mem else 0
            if msg_count == 0:
                continue
            memory_sessions.append(
                {
                    "session_id": s.session_id,
                    "title": title or s.name or "Chat",
                    "created_at": created_at,
                    "path": "",
                    "message_count": msg_count,
                    "agent_name": s.name or "",
                    "status": self.get_session_status(sid).name.lower(),
                }
            )

        for ds in disk_sessions:
            ds["session_id"] = ds.get("memory_id", ds.get("session_id", ""))
            ds.setdefault(
                "status", self.get_session_status(ds["session_id"]).name.lower()
            )

        combined = memory_sessions + disk_sessions
        combined.sort(key=lambda s: s.get("created_at") or "", reverse=True)
        return combined

    def switch_session(self, session_id: str) -> None:
        self._current_session_id = session_id
