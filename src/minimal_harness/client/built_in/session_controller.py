"""Session lifecycle management — extracted from TUIApp."""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

from minimal_harness.agent.registry import AgentRegistryProtocol
from minimal_harness.agent.runtime import AgentRuntimeProtocol
from minimal_harness.agent.simple import SimpleAgent
from minimal_harness.client.built_in.buffer import StreamBuffer
from minimal_harness.client.built_in.config.agents import (
    SYSTEM_PROMPTS_DIR,
    load_agents_config,
    read_system_prompt,
)
from minimal_harness.client.built_in.context import AppContext
from minimal_harness.client.built_in.memory import PersistentMemory
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
        self._runtime = runtime
        self._agent_registry = agent_registry
        self._ctx = ctx
        self._current_session_id: str | None = None
        self._sessions: dict[str, ConversationSession] = {}
        self._preset_session_ids: set[str] = set()
        self._active_runs: dict[
            str, tuple[asyncio.Task, asyncio.Event, asyncio.Queue[AgentEvent | None]]
        ] = {}
        self._foreground_session_id: str | None = None
        self.streaming = False
        self.buf = StreamBuffer()

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
        return {sid for sid in self._active_runs if sid != self._foreground_session_id}

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
                return

    def set_streaming(self, active: bool) -> None:
        self.streaming = active

    def create_session(
        self,
        agent_name: str = "general_assistant",
        system_prompt: str | None = None,
        default_tools: list[str] | None = None,
    ) -> ConversationSession:
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

        session = ConversationSession(
            session_id=self._ctx.memory.session_id,
            agent=agent,
            memory=self._ctx.memory,
            tools=list(tools),
            name=self._ctx.memory.title or "New Chat",
        )
        if default_tools is not None:
            session.memory.selected_tools = default_tools  # type: ignore[reportAttributeAccessIssue]
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
        agents = load_agents_config()
        if not agents:
            return
        llm = self._ctx._create_llm_provider(self._ctx.config)
        for a in agents:
            prompt_path = SYSTEM_PROMPTS_DIR / a["system_prompt"]
            system_prompt = read_system_prompt(prompt_path) or a.get("description", "")
            default_tools = a.get("default_tools") or []

            resolved_tools = [
                self._ctx.all_tools[n]
                for n in default_tools
                if n in self._ctx.all_tools
            ] or self._ctx.active_tools

            memory = PersistentMemory(
                system_prompt=system_prompt,
                agent_name=a["name"],
                session_id=uuid.uuid4().hex,
            )
            agent = SimpleAgent(
                llm_provider=llm, tools=list(resolved_tools), memory=memory
            )
            session = ConversationSession(
                session_id=memory.session_id,
                agent=agent,
                memory=memory,
                tools=list(resolved_tools),
                name=a["name"],
            )
            self._sessions[session.session_id] = session
            self._preset_session_ids.add(session.session_id)
            self._agent_registry.register(
                agent=agent, name=a["name"], description=a.get("description", "")
            )

    def start_with_default_agent(self) -> None:
        agents = load_agents_config()
        default_name = self._ctx.config.get("default_agent", "general_assistant")
        agent_cfg = self._get_default_agent(agents, default_name)
        if agent_cfg:
            prompt = read_system_prompt(
                SYSTEM_PROMPTS_DIR / agent_cfg["system_prompt"]
            ) or agent_cfg.get("description", "")
            self.create_session(
                agent_name=agent_cfg["name"],
                system_prompt=prompt,
                default_tools=agent_cfg.get("default_tools"),
            )
        else:
            self.create_session()

    def drain_session_events(self, session_id: str) -> tuple[list[AgentEvent], bool]:
        """Drain available events from a session's active run.

        Returns (events, completed) where completed means the
        None sentinel was found (run finished).
        """
        if session_id == self._foreground_session_id:
            return [], False
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

        return events, done

    def poll_handoff_completion(self) -> bool:
        """Check if any handoff target (other than foreground) completed.

        Uses task.done() instead of destructively peeking at the queue,
        so queued events are not consumed/discarded prematurely.
        Skips the currently-viewed session — drain_session_events handles it.
        """
        for sid in list(self.handoff_target_ids):
            if sid == self._current_session_id:
                continue
            if sid not in self._active_runs:
                continue
            task, _, event_queue = self._active_runs[sid]
            if task.done():
                # Drain remaining events (silently — user is not viewing this session)
                while True:
                    try:
                        event_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                self._active_runs.pop(sid, None)
                return True
        return False

    def start_run(
        self, session: ConversationSession, user_input: str
    ) -> tuple[asyncio.Event, asyncio.Queue[AgentEvent | None]]:
        task, stop_event, event_queue = self._runtime.run(
            agent=session.agent,
            memory=session.memory,
            tools=session.tools,
            user_input=[{"type": "text", "text": user_input}],
        )
        self._active_runs[session.session_id] = (task, stop_event, event_queue)
        self._foreground_session_id = session.session_id
        return stop_event, event_queue

    def end_run(self, session_id: str) -> None:
        self._active_runs.pop(session_id, None)
        if self._foreground_session_id == session_id:
            self._foreground_session_id = None

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
            name=memory.title or "Untitled",
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
                    "created_at": "",
                    "path": "",
                    "message_count": len(s.memory.get_all_messages()),
                    "agent_name": getattr(s.memory, "agent_name", ""),
                }
            )

        return memory_sessions + disk_sessions

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
