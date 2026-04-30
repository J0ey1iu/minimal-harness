"""Run lifecycle management — start, end, drain agent event queues."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minimal_harness.agent.runtime import AgentRuntimeProtocol
    from minimal_harness.client.built_in.session import ConversationSession
    from minimal_harness.types import AgentEvent


class RunManager:
    def __init__(
        self,
        runtime: AgentRuntimeProtocol,
    ) -> None:
        self._runtime = runtime
        self._active_runs: dict[
            str, tuple[asyncio.Task, asyncio.Event, asyncio.Queue[AgentEvent | None]]
        ] = {}
        self._foreground_session_id: str | None = None

    @property
    def active_runs(
        self,
    ) -> dict[
        str, tuple[asyncio.Task, asyncio.Event, asyncio.Queue[AgentEvent | None]]
    ]:
        return self._active_runs

    @property
    def foreground_session_id(self) -> str | None:
        return self._foreground_session_id

    @foreground_session_id.setter
    def foreground_session_id(self, value: str | None) -> None:
        self._foreground_session_id = value

    def start_run(
        self, session: ConversationSession, user_input: str
    ) -> tuple[asyncio.Event, asyncio.Queue[AgentEvent | None]]:
        task, stop_event, event_queue = self._runtime.run(
            agent=session.agent,
            memory=session.memory,
            tools=session.tools,
            user_input=[{"type": "text", "text": user_input}],
            agent_name=session.name,
        )
        self._active_runs[session.session_id] = (task, stop_event, event_queue)
        self._foreground_session_id = session.session_id
        return stop_event, event_queue

    def end_run(self, session_id: str) -> None:
        self._active_runs.pop(session_id, None)
        if self._foreground_session_id == session_id:
            self._foreground_session_id = None

    def drain_session_events(self, session_id: str) -> tuple[list[AgentEvent], bool]:
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
