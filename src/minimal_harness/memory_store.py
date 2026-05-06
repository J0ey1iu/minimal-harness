from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from minimal_harness.memory import Memory

MemoryFactory = Callable[[], Memory]


@runtime_checkable
class MemoryStoreProtocol(Protocol):
    """Protocol for persistent memory storage."""

    def create_memory(
        self,
        memory_id: str | None = None,
        agent_name: str = "",
    ) -> Memory: ...

    def get_memory(self, memory_id: str) -> Memory | None: ...

    def save_memory(
        self, memory: Memory, memory_id: str, extra: dict[str, Any] | None = None
    ) -> None: ...

    def delete_memory(self, memory_id: str) -> bool: ...

    def list_sessions(self) -> list[dict[str, Any]]: ...
