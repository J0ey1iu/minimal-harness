from __future__ import annotations

from typing import Any, Protocol, TypedDict
from uuid import uuid4

from minimal_harness.memory import (
    ConversationMemory,
    Memory,
    MemoryData,
    Message,
    TokenUsage,
)


class SessionSummary(TypedDict):
    session_id: str
    agent_name: str
    user_id: str
    scenario_id: str | None
    title: str | None
    created_at: str
    message_count: int
    status: str  # "running" | "idle", filled by the TUI layer
    display_name_locale: (
        str | None
    )  # JSON-encoded i18n dict, e.g. {"zh":"通用助手","en":"General Assistant"}


class Session(Protocol):
    """An identity-enriched Memory.

    Carries user/scenario context and delegates all message operations
    to an underlying Memory instance.  Stores that work with ``Session``
    can expose user- and scenario-level information without relying on
    the ``MemoryData.extra`` dict.
    """

    @property
    def session_id(self) -> str: ...
    @property
    def memory_id(self) -> str: ...
    @property
    def agent_name(self) -> str: ...
    @property
    def display_name_locale(self) -> str | None: ...
    @property
    def user_id(self) -> str: ...
    @property
    def scenario_id(self) -> str | None: ...
    @property
    def title(self) -> str | None: ...
    @property
    def created_at(self) -> str: ...
    @property
    def memory(self) -> Memory: ...

    async def add_message(self, message: Message) -> None: ...
    def get_all_messages(self) -> list[Message]: ...
    def get_forward_messages(self) -> list[Message]: ...
    def clear_messages(self) -> None: ...
    def set_message_usage(self, usage: TokenUsage) -> None: ...
    def get_message_usage(self) -> TokenUsage: ...
    def dump_memory(self) -> MemoryData: ...
    def load_memory(self, data: MemoryData) -> None: ...


class SimpleSession:
    """A basic Session implementation backed by ConversationMemory."""

    def __init__(
        self,
        session_id: str = "",
        agent_name: str = "",
        user_id: str = "",
        scenario_id: str | None = None,
        display_name_locale: str | None = None,
    ) -> None:
        from minimal_harness.database import generate_bigint_id

        self._memory = ConversationMemory()
        self.db_id: int = generate_bigint_id()
        self.session_id = session_id or f"sess_{uuid4().hex[:12]}"
        self.memory_id = self.session_id
        self.agent_name = agent_name
        self.user_id = user_id
        self.scenario_id = scenario_id
        self.title: str | None = None
        self.display_name_locale = display_name_locale
        self._created_at = ""

    @property
    def memory(self) -> Memory:
        return self._memory

    @property
    def created_at(self) -> str:
        return self._created_at

    @created_at.setter
    def created_at(self, value: str) -> None:
        self._created_at = value

    async def add_message(self, message: Message) -> None:
        await self._memory.add_message(message)

    def get_all_messages(self) -> list[Message]:
        return self._memory.get_all_messages()

    def get_forward_messages(self) -> list[Message]:
        return self._memory.get_forward_messages()

    def clear_messages(self) -> None:
        self._memory.clear_messages()

    def set_message_usage(self, usage: Any) -> None:
        self._memory.set_message_usage(usage)

    def get_message_usage(self) -> Any:
        return self._memory.get_message_usage()

    def dump_memory(self) -> Any:
        return self._memory.dump_memory()

    def load_memory(self, data: Any) -> None:
        self._memory.load_memory(data)
