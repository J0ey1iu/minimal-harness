"""Session lifecycle management — coordinates SessionFactory, AgentManager, RunManager, and HandoffCoordinator."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from minimal_harness.agent.registry import AgentRegistryProtocol
from minimal_harness.agent.runtime import AgentRuntimeProtocol
from minimal_harness.client.built_in.agent_manager import AgentManager
from minimal_harness.client.built_in.buffer import StreamBuffer
from minimal_harness.client.built_in.config.agents import load_agents_config
from minimal_harness.client.built_in.context import AppContext
from minimal_harness.client.built_in.handoff_coordinator import HandoffCoordinator
from minimal_harness.client.built_in.memory import PersistentMemory
from minimal_harness.client.built_in.run_manager import RunManager
from minimal_harness.client.built_in.session import ConversationSession
from minimal_harness.client.built_in.session_factory import SessionFactory
from minimal_harness.tool.base import Tool

if TYPE_CHECKING:
    from minimal_harness.memory import Memory
    from minimal_harness.types import AgentEvent


class SessionController:
    """Coordinates session lifecycle: creation, run management, handoff tracking."""

    def __init__(
        self,
        runtime: AgentRuntimeProtocol,
        agent_registry: AgentRegistryProtocol,
        ctx: AppContext,
    ) -> None:
        self._ctx = ctx
        self._agent_registry = agent_registry
        self._factory = SessionFactory(ctx)
        self._agents = AgentManager(ctx, agent_registry)
        self._runs = RunManager(runtime)
        self._handoff = HandoffCoordinator(
            run_manager=self._runs,
            sessions=self._agents.sessions,
            create_session_fn=lambda agent_name: self.create_session(
                agent_name=agent_name
            ),
        )
        self._current_session_id: str | None = None
        self.streaming = False
        self.buf = StreamBuffer()

    @property
    def _sessions(self) -> dict[str, ConversationSession]:
        return self._agents.sessions

    @property
    def _active_runs(
        self,
    ) -> dict[
        str, tuple[asyncio.Task, asyncio.Event, asyncio.Queue[AgentEvent | None]]
    ]:
        return self._runs.active_runs

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
        return self._handoff.handoff_target_ids

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

    def make_handoff_memory(self, agent_name: str) -> Memory:
        metadata = self._agent_registry.get(agent_name)
        system_prompt = ""
        if metadata is not None:
            msgs = metadata.agent.memory.get_all_messages()
            if msgs and msgs[0].get("role") == "system":
                content = msgs[0].get("content", "")
                system_prompt = content if isinstance(content, str) else ""
        prev = self._current_session_id
        session = self.create_session(
            agent_name=agent_name, system_prompt=system_prompt
        )
        self._current_session_id = prev
        self._handoff.register_handoff_session(session.session_id)
        return session.memory

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

    def register_handoff_run(
        self,
        agent_name: str,
        task: asyncio.Task,
        stop_event: asyncio.Event,
        queue: asyncio.Queue[AgentEvent | None],
    ) -> None:
        self._handoff.register_handoff_run(agent_name, task, stop_event, queue)

    def interrupt(self) -> None:
        session = self.current_session
        if session is not None:
            session.interrupt()
        if self._current_session_id and self._current_session_id in self._active_runs:
            _, stop_event, _ = self._active_runs[self._current_session_id]
            stop_event.set()

    def set_streaming(self, active: bool) -> None:
        self.streaming = active

    def start_run(
        self, session: ConversationSession, user_input: str
    ) -> tuple[asyncio.Event, asyncio.Queue[AgentEvent | None]]:
        return self._runs.start_run(session, user_input)

    def end_run(self, session_id: str) -> None:
        self._runs.end_run(session_id)

    def drain_session_events(self, session_id: str) -> tuple[list[AgentEvent], bool]:
        return self._runs.drain_session_events(session_id)

    def poll_handoff_completion(self) -> bool:
        return self._handoff.poll_handoff_completion(self._current_session_id)

    def get_all_sessions_metadata(self) -> list[dict[str, Any]]:
        disk_sessions = PersistentMemory.list_sessions()
        disk_ids = {s["session_id"] for s in disk_sessions}

        memory_sessions = []
        for sid, s in self._sessions.items():
            if sid in disk_ids:
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
