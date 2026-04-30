"""Tracks handoff state and coordinates handoff runs."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from minimal_harness.client.built_in.run_manager import RunManager
    from minimal_harness.client.built_in.session import ConversationSession
    from minimal_harness.types import AgentEvent


class HandoffCoordinator:
    def __init__(
        self,
        run_manager: RunManager,
        sessions: dict[str, ConversationSession],
        create_session_fn: Callable[[str], ConversationSession],
    ) -> None:
        self._runs = run_manager
        self._sessions = sessions
        self._create_session_fn = create_session_fn
        self._last_handoff_session_id: str | None = None

    def register_handoff_session(self, session_id: str) -> None:
        self._last_handoff_session_id = session_id

    def register_handoff_run(
        self,
        agent_name: str,
        task: asyncio.Task,
        stop_event: asyncio.Event,
        queue: asyncio.Queue[AgentEvent | None],
    ) -> None:
        sid = self._last_handoff_session_id
        if sid is not None and sid in self._sessions:
            self._runs.active_runs[sid] = (task, stop_event, queue)
        else:
            session = self._create_session_fn(agent_name)
            self._runs.active_runs[session.session_id] = (task, stop_event, queue)

    @property
    def handoff_target_ids(self) -> set[str]:
        return {
            sid
            for sid in self._runs.active_runs
            if sid != self._runs.foreground_session_id
        }

    def poll_handoff_completion(self, current_session_id: str | None) -> bool:
        for sid in list(self.handoff_target_ids):
            if sid == current_session_id:
                continue
            if sid not in self._runs.active_runs:
                continue
            task, _, _ = self._runs.active_runs[sid]
            if task.done():
                return True
        return False
