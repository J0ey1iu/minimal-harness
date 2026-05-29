"""Session lifecycle management — coordinates SessionFactory and AgentManager."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from minimal_harness.agent.registry import AgentRegistryProtocol
from minimal_harness.agent.runtime import AgentRuntimeProtocol
from minimal_harness.client.built_in.agent_manager import AgentManager
from minimal_harness.client.built_in.buffer import StreamBuffer
from minimal_harness.client.built_in.context import AppContext
from minimal_harness.client.built_in.session import ConversationSession, SessionStatus
from minimal_harness.client.built_in.session_factory import SessionFactory
from minimal_harness.types import AgentEnd

if TYPE_CHECKING:
    from minimal_harness.types import AgentEvent, ToolMetadata

logger = logging.getLogger(__name__)


class SessionController:
    """Coordinates session lifecycle: creation and run management.

    Uses Layer 2 abstractions (AgentRegistry, JsonlSessionStore, ToolRegistry)
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
        self._session_errors: dict[str, str] = {}
        self._lock = asyncio.Lock()

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

    async def get_memory(self, session_id: str | None = None) -> Any | None:
        sid = session_id or self._current_session_id
        if sid is None:
            return None
        session = self._sessions.get(sid)
        if session is None:
            return None
        return await self._ctx.session_store.get_session(session.session.memory_id)

    async def get_active_tools(self) -> list[ToolMetadata]:
        session = self.current_session
        if session and session.tool_names:
            return [
                t
                for n in session.tool_names
                if (t := self._ctx.all_tools.get(n)) is not None
            ]
        default_name = self._ctx.config.get("default_agent", "")
        if not default_name:
            return []
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
        agent_name: str,
        default_tools: list[str] | None = None,
    ) -> ConversationSession:
        session = await self._factory.create_session(
            agent_name=agent_name,
            default_tools=default_tools,
        )
        self._sessions[session.session.memory_id] = session
        self._current_session_id = session.session.memory_id
        logger.info(
            "session.create id=%s agent=%s", session.session.memory_id, agent_name
        )
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
        tools: list[ToolMetadata] | None = None,
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
            stop_event.set()
            if not task.done():
                task.cancel()

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
        async with self._lock:
            if session.session.memory_id in self._active_runs:
                return None
            llm_kwargs: dict[str, Any] = {}
            reasoning_effort = self._ctx.config.get("reasoning_effort")
            if reasoning_effort in ("low", "medium", "high"):
                llm_kwargs["reasoning_effort"] = reasoning_effort
            llm_kwargs["extra_body"] = {
                "enable_thinking": reasoning_effort != "off",
            }
            llm_kwargs["timeout"] = None
            logger.info(
                "session.run.start id=%s agent=%s input=%.80s",
                session.session.memory_id,
                session.agent_metadata_id,
                user_input,
            )
            task, stop_event, event_queue = await self._runtime.run(
                user_input=[{"type": "text", "text": user_input}],
                agent_metadata_id=session.agent_metadata_id,
                memory_id=session.session.memory_id,
                tool_names=session.tool_names if session.tool_names else None,
                context={"agent_name": session.session.agent_name},
                llm_kwargs=llm_kwargs or None,
            )
            self._active_runs[session.session.memory_id] = (
                task,
                stop_event,
                event_queue,
            )
            self._per_session_streaming[session.session.memory_id] = True
        await self._notify_status_changed(
            session.session.memory_id, SessionStatus.RUNNING
        )
        return stop_event, event_queue

    def add_status_listener(
        self, listener: Callable[[str, SessionStatus], Awaitable[None]]
    ) -> None:
        self._status_listeners.append(listener)

    def remove_status_listener(
        self, listener: Callable[[str, SessionStatus], Awaitable[None]]
    ) -> None:
        try:
            self._status_listeners.remove(listener)
        except ValueError:
            pass

    async def _notify_status_changed(
        self, session_id: str, status: SessionStatus
    ) -> None:
        for listener in list(self._status_listeners):
            try:
                await listener(session_id, status)
            except Exception:
                pass

    def get_session_status(self, session_id: str) -> SessionStatus:
        return (
            SessionStatus.RUNNING
            if session_id in self._active_runs
            else SessionStatus.IDLE
        )

    def is_session_running(self, session_id: str) -> bool:
        return session_id in self._active_runs

    def is_any_session_running(self) -> bool:
        return bool(self._active_runs)

    def get_all_sessions(self) -> dict[str, ConversationSession]:
        return dict(self._sessions)

    async def end_run(self, session_id: str) -> None:
        async with self._lock:
            self._active_runs.pop(session_id, None)
            self._per_session_streaming.pop(session_id, None)
            logger.info("session.run.end id=%s", session_id)
        await self._notify_status_changed(session_id, SessionStatus.IDLE)
        session = self._sessions.get(session_id)
        if session is not None:
            try:
                logger.debug("session.end-run.save id=%s", session_id)
                await self._ctx.session_store.save_memory(
                    memory=session.session.memory,
                    session_id=session_id,
                    extra={
                        "memory_id": session_id,
                        "title": session.session.title,
                        "created_at": session.session.created_at,
                        "agent_name": session.session.agent_name,
                        "user_id": session.session.user_id,
                        "scenario_id": session.session.scenario_id,
                    },
                )
            except Exception:
                logger.exception("session.end-run.save.error id=%s", session_id)

    async def poll_background_completions(
        self, current_session_id: str | None
    ) -> list[str]:
        if not current_session_id:
            return []
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
                    if isinstance(event, AgentEnd) and event.error:
                        self._session_errors[sid] = event.error
                except asyncio.QueueEmpty:
                    break
            if done:
                async with self._lock:
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
                if isinstance(event, AgentEnd) and event.error:
                    self._session_errors[session_id] = event.error
            except asyncio.QueueEmpty:
                break

        if done:
            async with self._lock:
                self._active_runs.pop(session_id, None)
                self._per_session_streaming.pop(session_id, None)
            await self._notify_status_changed(session_id, SessionStatus.IDLE)

        return events, done

    async def get_all_sessions_metadata(self) -> list[dict[str, Any]]:
        store = self._ctx.session_store
        disk_sessions = await store.list_sessions()
        disk_ids = {s["session_id"] for s in disk_sessions}

        memory_sessions = []
        for sid, s in self._sessions.items():
            if s.session.memory_id in disk_ids:
                continue
            session_obj = await self._ctx.session_store.get_session(s.session.memory_id)
            title = session_obj.title if session_obj else None
            created_at = session_obj.created_at if session_obj else ""
            msg_count = len(session_obj.get_all_messages()) if session_obj else 0
            if msg_count == 0:
                continue
            memory_sessions.append(
                {
                    "session_id": s.session.memory_id,
                    "title": title or s.session.agent_name or "Chat",
                    "created_at": created_at,
                    "path": "",
                    "message_count": msg_count,
                    "agent_name": s.session.agent_name or "",
                    "status": self.get_session_status(sid).name.lower(),
                }
            )

        for ds in disk_sessions:
            ds["status"] = self.get_session_status(ds["session_id"]).name.lower()

        combined = memory_sessions + disk_sessions
        combined.sort(key=lambda s: s.get("created_at") or "", reverse=True)
        return combined

    def switch_session(self, session_id: str) -> None:
        self._current_session_id = session_id

    async def delete_session(self, session_id: str) -> bool:
        if session_id in self._active_runs:
            task, stop_event, _ = self._active_runs[session_id]
            stop_event.set()
            if not task.done():
                task.cancel()
            async with self._lock:
                self._active_runs.pop(session_id, None)
                self._per_session_streaming.pop(session_id, None)

        self._sessions.pop(session_id, None)
        self._per_session_buf.pop(session_id, None)
        self._session_errors.pop(session_id, None)

        if self._current_session_id == session_id:
            self._current_session_id = None

        result = await self._ctx.session_store.delete_session(session_id)

        await self._notify_status_changed(session_id, SessionStatus.IDLE)

        logger.info("session.delete id=%s existed=%s", session_id, result)
        return result

    async def rename_session(self, session_id: str, new_title: str) -> bool:
        result = await self._ctx.session_store.rename_session(session_id, new_title)
        if result:
            conv = self._sessions.get(session_id)
            if conv is not None:
                conv.session.title = new_title  # type: ignore[misc]
            logger.info("session.rename id=%s title=%s", session_id, new_title)
        return result

    def pop_session_error(self, session_id: str) -> str | None:
        return self._session_errors.pop(session_id, None)
