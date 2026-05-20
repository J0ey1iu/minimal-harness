from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from minimal_harness.memory import Memory
from minimal_harness.session import Session, SessionSummary

MemoryFactory = Callable[[], Memory]


@runtime_checkable
class SessionStoreProtocol(Protocol):
    """Protocol for persistent session (memory) storage.

    All methods operate on ``Session`` instances, which carry identity
    information (``user_id``, ``scenario_id``) alongside message data.
    """

    async def create_session(
        self,
        session_id: str | None = None,
        agent_name: str = "",
        user_id: str = "",
        scenario_id: str | None = None,
    ) -> Session: ...

    async def get_session(self, session_id: str) -> Session | None: ...

    async def save_memory(
        self, memory: Memory, session_id: str, extra: dict[str, Any] | None = None
    ) -> None: ...

    async def delete_session(self, session_id: str) -> bool: ...

    async def list_sessions(self) -> list[SessionSummary]: ...

    async def list_user_sessions(
        self, user_id: str, scenario_id: str | None = None
    ) -> list[SessionSummary]: ...

    async def get_session_messages(self, session_id: str) -> list[dict]: ...

    def get_messages_as_items(self, session: Session) -> list[dict]: ...
