from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from minimal_harness.memory import (
    ConversationMemory,
    Memory,
    MemoryData,
    Message,
    TokenUsage,
)
from minimal_harness.memory_store import MemoryFactory
from minimal_harness.session import Session, SessionSummary


class DiskSessionStore:
    """Persistence layer for Session instances — stores and retrieves sessions.

    This is a Layer 2 service abstraction. It manages the lifecycle
    of Session instances and provides CRUD operations with file-based
    persistence.
    """

    def __init__(
        self,
        storage_dir: Path | None = None,
        memory_factory: MemoryFactory | None = None,
    ) -> None:
        self._storage_dir = storage_dir or Path.home() / ".minimal_harness" / "sessions"
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, ManagedSession] = {}
        self._memory_factory = memory_factory or (lambda: ConversationMemory())
        self._locks: dict[str, asyncio.Lock] = {}
        self._pending_saves: dict[str, asyncio.Task] = {}
        self._transient: set[str] = set()
        self._list_cache: list[SessionSummary] | None = None
        self._list_cache_mtime: float = 0.0

    @property
    def storage_dir(self) -> Path:
        return self._storage_dir

    # -- lock helpers -------------------------------------------------------

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    def _cleanup_lock(self, session_id: str) -> None:
        self._locks.pop(session_id, None)

    # -- scheduled persistance (debounced) ----------------------------------

    def _schedule_persist(self, managed: ManagedSession) -> None:
        mid = managed.session_id
        prev = self._pending_saves.get(mid)
        if prev is not None:
            prev.cancel()

        async def _task() -> None:
            try:
                await self._persist(managed)
            except asyncio.CancelledError:
                pass
            finally:
                if self._pending_saves.get(mid) is asyncio.current_task():
                    self._pending_saves.pop(mid, None)

        self._pending_saves[mid] = asyncio.create_task(_task())

    # -- CRUD ---------------------------------------------------------------

    async def create_session(
        self,
        session_id: str | None = None,
        agent_name: str = "",
        user_id: str = "",
        scenario_id: str | None = None,
        transient: bool = False,
    ) -> Session:
        mid = session_id or uuid.uuid4().hex
        inner = self._memory_factory()
        managed = ManagedSession(
            store=self,
            session_id=mid,
            inner=inner,
            agent_name=agent_name,
            user_id=user_id,
            scenario_id=scenario_id,
            transient=transient,
        )
        self._cache[mid] = managed
        if transient:
            self._transient.add(mid)
        return managed

    async def get_session(self, session_id: str) -> Session | None:
        cached = self._cache.get(session_id)
        if cached is not None:
            return cached
        path = self._path_for(session_id)

        def _read() -> MemoryData | None:
            if not path.exists():
                return None
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return None

        data = await asyncio.to_thread(_read)
        if data is None:
            return None
        inner = self._memory_factory()
        inner.load_memory(data)
        extra = data.get("extra", {})
        is_transient = extra.get("transient", False) or session_id in self._transient
        managed = ManagedSession(
            store=self,
            session_id=session_id,
            inner=inner,
            agent_name=extra.get("agent_name", ""),
            user_id=extra.get("user_id", ""),
            scenario_id=extra.get("scenario_id", None),
            transient=is_transient,
        )
        managed._title = extra.get("title")
        managed._created_at = extra.get("created_at", "")
        managed._first_user_message = False
        if is_transient:
            self._transient.add(session_id)
        self._cache[session_id] = managed
        return managed

    async def save_memory(
        self, memory: Memory, session_id: str, extra: dict[str, Any] | None = None
    ) -> None:
        async with self._get_lock(session_id):
            if session_id not in self._cache:
                return
            data = memory.dump_memory()
            existing = data.get("extra", {})
            merged_extra = {**existing, **(extra or {})}
            data["extra"] = merged_extra
            path = self._path_for(session_id)
            content = json.dumps(data, indent=2, ensure_ascii=False, default=str)

            def _write() -> None:
                tmp = path.with_suffix(".tmp")
                tmp.write_text(content, encoding="utf-8")
                os.replace(tmp, path)

            await asyncio.to_thread(_write)
            self._invalidate_list_cache()

    async def delete_session(self, session_id: str) -> bool:
        prev = self._pending_saves.pop(session_id, None)
        if prev is not None:
            prev.cancel()
            try:
                await prev
            except (asyncio.CancelledError, Exception):
                pass

        self._cache.pop(session_id, None)
        self._transient.discard(session_id)

        async with self._get_lock(session_id):
            path = self._path_for(session_id)

            def _unlink() -> bool:
                if path.exists():
                    path.unlink()
                    return True
                return False

            result = await asyncio.to_thread(_unlink)

        self._invalidate_list_cache()
        self._cleanup_lock(session_id)
        return result

    async def list_sessions(self) -> list[SessionSummary]:
        def _list() -> list[SessionSummary]:
            sessions: list[SessionSummary] = []
            try:
                paths = sorted(
                    self._storage_dir.glob("*.json"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
            except OSError:
                return sessions
            for path in paths:
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    extra = data.get("extra", {})
                    if extra.get("transient"):
                        continue
                    sessions.append(
                        {
                            "session_id": extra.get("memory_id", path.stem),
                            "title": extra.get("title", "Untitled"),
                            "created_at": extra.get("created_at", ""),
                            "message_count": len(data.get("messages", [])),
                            "agent_name": extra.get("agent_name", ""),
                            "user_id": extra.get("user_id", ""),
                            "scenario_id": extra.get("scenario_id", None),
                            "status": "idle",
                        }
                    )
                except Exception:
                    continue
            return sessions

        mtime = self._storage_dir.stat().st_mtime if self._storage_dir.exists() else 0.0
        if self._list_cache is not None and mtime == self._list_cache_mtime:
            return self._list_cache

        result = await asyncio.to_thread(_list)
        self._list_cache = result
        self._list_cache_mtime = mtime
        return result

    def _invalidate_list_cache(self) -> None:
        self._list_cache = None
        self._list_cache_mtime = 0.0

    def _path_for(self, session_id: str) -> Path:
        safe = session_id.replace("/", "_").replace("\\", "_").replace("..", "_")
        return self._storage_dir / f"{safe}.json"

    async def export_memory_json(self, session_id: str, indent: int | None = 2) -> str:
        session = await self.get_session(session_id)
        if session is None:
            raise ValueError(f"Session '{session_id}' not found")
        data = session.dump_memory()
        return json.dumps(data, indent=indent, ensure_ascii=False, default=str)

    async def _persist(self, managed: ManagedSession) -> None:
        await self.save_memory(
            memory=managed,
            session_id=managed.session_id,
            extra={
                "memory_id": managed.session_id,
                "title": managed.title,
                "created_at": managed.created_at,
                "agent_name": managed.agent_name,
                "user_id": managed.user_id,
                "scenario_id": managed.scenario_id,
                "transient": managed._transient,
            },
        )


class ManagedSession:
    """Session wrapper that auto-persists via DiskSessionStore.

    Implements both the ``Memory`` and ``Session`` protocols.
    Delegates message operations to an inner ``Memory`` instance
    and automatically persists after each mutating operation.
    """

    def __init__(
        self,
        store: DiskSessionStore,
        session_id: str,
        inner: Memory,
        agent_name: str = "",
        user_id: str = "",
        scenario_id: str | None = None,
        transient: bool = False,
    ) -> None:
        self._store = store
        self._session_id = session_id
        self._inner = inner
        self.agent_name = agent_name
        self._user_id = user_id
        self._scenario_id = scenario_id
        self._title: str | None = None
        self._created_at = datetime.now().isoformat()
        self._first_user_message = True
        self._transient = transient

    # -- Session protocol properties ---------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def memory_id(self) -> str:
        return self._session_id

    @property
    def user_id(self) -> str:
        return self._user_id

    @property
    def scenario_id(self) -> str | None:
        return self._scenario_id

    @property
    def title(self) -> str | None:
        return self._title

    @property
    def created_at(self) -> str:
        return self._created_at

    @created_at.setter
    def created_at(self, value: str) -> None:
        self._created_at = value

    @property
    def memory(self) -> Memory:
        return self._inner

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
        data = self._inner.dump_memory()
        data["extra"] = {
            **data.get("extra", {}),
            "memory_id": self._session_id,
            "title": self._title,
            "created_at": self._created_at,
            "agent_name": self.agent_name,
            "user_id": self._user_id,
            "scenario_id": self._scenario_id,
            "transient": self._transient,
        }
        return data

    def load_memory(self, data: MemoryData) -> None:
        self._inner.load_memory(data)

    # -- internal -------------------------------------------------------

    def _auto_save(self) -> None:
        self._store._schedule_persist(self)
