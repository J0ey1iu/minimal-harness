"""Tracks handoff state and coordinates handoff runs."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from minimal_harness.client.built_in.session import ConversationSession
    from minimal_harness.types import AgentEvent


class HandoffCoordinator:
    def __init__(
        self,
        active_runs: dict[
            str, tuple[asyncio.Task, asyncio.Event, asyncio.Queue[AgentEvent | None]]
        ],
        foreground_session_id: Callable[[], str | None],
        sessions: dict[str, ConversationSession],
        create_session_fn: Callable[[str], ConversationSession],
    ) -> None:
        self._active_runs = active_runs
        self._foreground_session_id = foreground_session_id
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
    ) -> str:
        sid = self._last_handoff_session_id
        if sid is not None and sid in self._sessions:
            self._active_runs[sid] = (task, stop_event, queue)
        else:
            session = self._create_session_fn(agent_name)
            sid = session.session_id
            self._active_runs[sid] = (task, stop_event, queue)
        return sid

    @property
    def handoff_target_ids(self) -> set[str]:
        foreground_id = self._foreground_session_id()
        return {sid for sid in self._active_runs if sid != foreground_id}

    def poll_handoff_completion(self, current_session_id: str | None) -> bool:
        for sid in list(self.handoff_target_ids):
            if sid == current_session_id:
                continue
            if sid not in self._active_runs:
                continue
            task, _, _ = self._active_runs[sid]
            if task.done():
                return True
        return False
