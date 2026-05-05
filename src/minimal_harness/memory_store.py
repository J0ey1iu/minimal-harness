from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from minimal_harness.memory import (
    ConversationMemory,
    Memory,
    MemoryData,
    Message,
    TokenUsage,
)

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


class MemoryStore:
    """Persistence layer for Memory instances — stores and retrieves memories.

    This is a Layer 2 service abstraction. It manages the lifecycle
    of Memory instances and provides CRUD operations with file-based
    persistence.
    """

    def __init__(
        self,
        storage_dir: Path | None = None,
        memory_factory: MemoryFactory | None = None,
    ) -> None:
        self._storage_dir = storage_dir or Path.home() / ".minimal_harness" / "memories"
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, _ManagedMemory] = {}
        self._memory_factory = memory_factory or (lambda: ConversationMemory())

    @property
    def storage_dir(self) -> Path:
        return self._storage_dir

    def create_memory(
        self,
        memory_id: str | None = None,
        agent_name: str = "",
    ) -> Memory:
        mid = memory_id or uuid.uuid4().hex
        inner = self._memory_factory()
        managed = _ManagedMemory(
            store=self,
            memory_id=mid,
            inner=inner,
            agent_name=agent_name,
        )
        self._cache[mid] = managed
        return managed

    def get_memory(self, memory_id: str) -> Memory | None:
        cached = self._cache.get(memory_id)
        if cached is not None:
            return cached
        path = self._path_for(memory_id)
        if not path.exists():
            return None
        try:
            data: MemoryData = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        inner = self._memory_factory()
        inner.load_memory(data)
        extra = data.get("extra", {})
        managed = _ManagedMemory(
            store=self,
            memory_id=memory_id,
            inner=inner,
            agent_name=extra.get("agent_name", ""),
        )
        managed._title = extra.get("title")
        managed._created_at = extra.get("created_at", "")
        managed._first_user_message = False
        self._cache[memory_id] = managed
        return managed

    def save_memory(
        self, memory: Memory, memory_id: str, extra: dict[str, Any] | None = None
    ) -> None:
        data = memory.dump_memory()
        existing = data.get("extra", {})
        merged_extra = {**existing, **(extra or {})}
        data["extra"] = merged_extra
        self._path_for(memory_id).write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    def delete_memory(self, memory_id: str) -> bool:
        self._cache.pop(memory_id, None)
        path = self._path_for(memory_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        for path in sorted(
            self._storage_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                extra = data.get("extra", {})
                sessions.append(
                    {
                        "memory_id": extra.get("memory_id", path.stem),
                        "title": extra.get("title", "Untitled"),
                        "created_at": extra.get("created_at", ""),
                        "path": str(path),
                        "message_count": len(data.get("messages", [])),
                        "agent_name": extra.get("agent_name", ""),
                    }
                )
            except Exception:
                continue
        return sessions

    def _path_for(self, memory_id: str) -> Path:
        safe = memory_id.replace("/", "_").replace("\\", "_").replace("..", "_")
        return self._storage_dir / f"{safe}.json"

    def export_memory_json(self, memory_id: str, indent: int | None = 2) -> str:
        memory = self.get_memory(memory_id)
        if memory is None:
            raise ValueError(f"Memory '{memory_id}' not found")
        data = memory.dump_memory()
        return json.dumps(data, indent=indent, ensure_ascii=False, default=str)

    def _persist(self, managed: _ManagedMemory) -> None:
        self.save_memory(
            memory=managed._inner,
            memory_id=managed.memory_id,
            extra={
                "memory_id": managed.memory_id,
                "title": managed.title,
                "created_at": managed.created_at,
                "agent_name": managed.agent_name,
            },
        )


class _ManagedMemory:
    """Memory wrapper that auto-persists via MemoryStore.

    Delegates Memory protocol methods to an inner Memory
    instance and automatically persists after each mutating operation.
    """

    def __init__(
        self,
        store: MemoryStore,
        memory_id: str,
        inner: Memory,
        agent_name: str = "",
    ) -> None:
        self._store = store
        self._memory_id = memory_id
        self._inner = inner
        self.agent_name = agent_name
        self._title: str | None = None
        self._created_at = datetime.now().isoformat()
        self._first_user_message = True

    @property
    def memory_id(self) -> str:
        return self._memory_id

    @property
    def title(self) -> str | None:
        return self._title

    @property
    def created_at(self) -> str:
        return self._created_at

    @created_at.setter
    def created_at(self, value: str) -> None:
        self._created_at = value

    # -- Memory protocol methods (delegated to inner) -------------------

    def add_message(self, message: Message) -> None:
        if self._first_user_message and message.get("role") == "user":
            content = message.get("content", [])
            if content and isinstance(content[0], dict) and "text" in content[0]:
                self._title = content[0]["text"][:100]
            self._first_user_message = False
        self._inner.add_message(message)
        self._auto_save()

    def get_all_messages(self) -> list[Message]:
        return self._inner.get_all_messages()

    def get_forward_messages(self) -> list[Message]:
        return self._inner.get_forward_messages()

    def clear_messages(self) -> None:
        self._inner.clear_messages()
        self._auto_save()

    def set_message_usage(self, usage: TokenUsage) -> None:
        self._inner.set_message_usage(usage)
        self._auto_save()

    def get_message_usage(self) -> TokenUsage:
        return self._inner.get_message_usage()

    def dump_memory(self) -> MemoryData:
        return self._inner.dump_memory()

    def load_memory(self, data: MemoryData) -> None:
        self._inner.load_memory(data)

    # -- internal --------------------------------------------------------

    def _auto_save(self) -> None:
        self._store._persist(self)
