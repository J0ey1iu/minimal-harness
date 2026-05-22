"""SQLite-backed session store — replaces DiskSessionStore.

One table, one row per session.  The entire session state (messages +
metadata) is stored as a JSON blob in the ``data`` column, exactly
analogous to the previous JSON-per-file format.  SQLite handles atomicity
and write isolation, eliminating the need for atomic writes, temp files,
per-session locks, and debounced save tasks.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

from minimal_harness.memory import (
    ConversationMemory,
    Memory,
    MemoryData,
    Message,
    TokenUsage,
)
from minimal_harness.memory_store import MemoryFactory
from minimal_harness.session import Session, SessionSummary


class SqliteSessionStore:
    """Persistence layer backed by a local SQLite database.

    Each session is stored as one row in a single ``sessions`` table.
    The ``data`` column holds the full serialised session (messages +
    metadata) as JSON — the same content that was previously written
    to ``*.json`` files.

    Auto-save uses an ``asyncio.Event``-driven loop per session instead
    of the old debounce/cancel pattern, eliminating the race condition
    where a cancelled task's background thread continued writing while
    a new save was already in progress.
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        memory_factory: MemoryFactory | None = None,
    ) -> None:
        self._db_path = (
            Path(db_path)
            if db_path
            else Path.home() / ".minimal_harness" / "sessions.db"
        )
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: aiosqlite.Connection | None = None
        self._cache: dict[str, SqliteManagedSession] = {}
        self._memory_factory = memory_factory or (lambda: ConversationMemory())
        self._save_events: dict[str, asyncio.Event] = {}
        self._save_tasks: dict[str, asyncio.Task] = {}
        self._transient: set[str] = set()

    # ── connection ────────────────────────────────────────────────────────

    async def _ensure_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            self._conn = await aiosqlite.connect(str(self._db_path))
            await self._conn.execute("PRAGMA journal_mode=WAL")
            await self._conn.execute("PRAGMA busy_timeout=5000")
            await self._conn.execute(
                """CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    data       TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            await self._conn.commit()
        return self._conn

    # ── auto-save (event-driven, no cancellation) ────────────────────────

    def _schedule_save(self, managed: SqliteManagedSession) -> None:
        mid = managed.session_id
        if mid not in self._save_events:
            event = asyncio.Event()
            self._save_events[mid] = event
            task = asyncio.create_task(self._save_loop(mid, event))
            self._save_tasks[mid] = task
        self._save_events[mid].set()

    async def _save_loop(self, mid: str, event: asyncio.Event) -> None:
        consecutive_failures = 0
        try:
            while True:
                await event.wait()
                event.clear()
                managed = self._cache.get(mid)
                if managed is None:
                    break
                try:
                    await self.save_memory(
                        memory=managed,
                        session_id=mid,
                        extra={
                            "memory_id": mid,
                            "title": managed.title,
                            "created_at": managed.created_at,
                            "agent_name": managed.agent_name,
                            "user_id": managed.user_id,
                            "scenario_id": managed.scenario_id,
                            "transient": managed._transient,
                        },
                    )
                    consecutive_failures = 0
                except Exception:
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        break
                await asyncio.sleep(0)
                if not event.is_set():
                    break
        finally:
            self._save_events.pop(mid, None)
            self._save_tasks.pop(mid, None)

    async def flush(self) -> None:
        """Ensure all pending saves complete before shutdown."""
        if not self._save_events:
            return
        for event in self._save_events.values():
            event.set()
        for _ in range(100):
            if not self._save_events:
                return
            await asyncio.sleep(0.05)

    async def close(self) -> None:
        """Close the database connection.

        Must be called before shutdown to avoid background threads
        being interrupted by ``KeyboardInterrupt`` during interpreter exit.
        """
        if self._conn is not None:
            try:
                await self._conn.close()
            except Exception:
                pass
            self._conn = None

    # ── CRUD ──────────────────────────────────────────────────────────────

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
        managed = SqliteManagedSession(
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

        conn = await self._ensure_conn()
        cursor = await conn.execute(
            "SELECT data, created_at FROM sessions WHERE session_id = ?",
            [session_id],
        )
        row = await cursor.fetchone()
        if row is None:
            return None

        raw_data, created_at = row
        data: MemoryData = json.loads(raw_data)
        inner = self._memory_factory()
        inner.load_memory(data)
        extra = data.get("extra", {})
        is_transient = extra.get("transient", False) or session_id in self._transient

        managed = SqliteManagedSession(
            store=self,
            session_id=session_id,
            inner=inner,
            agent_name=extra.get("agent_name", ""),
            user_id=extra.get("user_id", ""),
            scenario_id=extra.get("scenario_id", None),
            transient=is_transient,
        )
        managed._title = extra.get("title")
        managed._created_at = created_at
        managed._first_user_message = False
        if is_transient:
            self._transient.add(session_id)
        self._cache[session_id] = managed
        return managed

    async def save_memory(
        self, memory: Memory, session_id: str, extra: dict[str, Any] | None = None
    ) -> None:
        if session_id not in self._cache:
            return
        data = memory.dump_memory()
        existing = data.get("extra", {})
        merged_extra = {**existing, **(extra or {})}
        data["extra"] = merged_extra
        content = json.dumps(data, ensure_ascii=False, default=str)
        now = datetime.now().isoformat()
        created_at = data.get("extra", {}).get("created_at", now)

        conn = await self._ensure_conn()
        await conn.execute(
            "INSERT OR REPLACE INTO sessions (session_id, data, created_at, updated_at) VALUES (?, ?, ?, ?)",
            [session_id, content, created_at, now],
        )
        await conn.commit()

    async def delete_session(self, session_id: str) -> bool:
        self._cache.pop(session_id, None)
        event = self._save_events.pop(session_id, None)
        task = self._save_tasks.pop(session_id, None)
        if event is not None:
            event.set()
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._transient.discard(session_id)

        conn = await self._ensure_conn()
        cursor = await conn.execute(
            "DELETE FROM sessions WHERE session_id = ?",
            [session_id],
        )
        await conn.commit()
        return cursor.rowcount > 0

    async def list_sessions(self) -> list[SessionSummary]:
        conn = await self._ensure_conn()
        cursor = await conn.execute(
            "SELECT session_id, data, created_at FROM sessions ORDER BY created_at DESC"
        )
        rows = await cursor.fetchall()
        result: list[SessionSummary] = []
        for session_id, raw_data, created_at in rows:
            try:
                data = json.loads(raw_data)
                extra = data.get("extra", {})
                if extra.get("transient"):
                    continue
                result.append(
                    {
                        "session_id": extra.get("memory_id", session_id),
                        "title": extra.get("title", "Untitled"),
                        "created_at": created_at,
                        "message_count": len(data.get("messages", [])),
                        "agent_name": extra.get("agent_name", ""),
                        "user_id": extra.get("user_id", ""),
                        "scenario_id": extra.get("scenario_id", None),
                        "status": "idle",
                    }
                )
            except Exception:
                continue
        return result

    async def list_user_sessions(
        self, user_id: str, scenario_id: str | None = None
    ) -> list[SessionSummary]:
        sessions = await self.list_sessions()
        return [s for s in sessions if s.get("user_id") == user_id]

    async def get_session_messages(self, session_id: str) -> list[dict]:
        session = await self.get_session(session_id)
        if session is None:
            return []
        return [dict(m) for m in session.get_all_messages()]

    @staticmethod
    def get_messages_as_items(session: Session) -> list[dict]:
        items: list[dict] = []
        for i, msg in enumerate(session.get_all_messages()):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                texts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                content = "\n".join(texts)
            elif not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            items.append(
                {
                    "id": f"msg-{i}",
                    "role": role,
                    "content": content,
                    "tool_calls": msg.get("tool_calls"),
                    "tool_call_id": msg.get("tool_call_id"),
                    "progress": msg.get("progress"),
                }
            )
        return items

    async def export_memory_json(self, session_id: str, indent: int | None = 2) -> str:
        session = await self.get_session(session_id)
        if session is None:
            raise ValueError(f"Session '{session_id}' not found")
        data = session.dump_memory()
        return json.dumps(data, indent=indent, ensure_ascii=False, default=str)


class SqliteManagedSession:
    """Session wrapper that auto-persists via SqliteSessionStore.

    Implements both the ``Memory`` and ``Session`` protocols.
    Delegates message operations to an inner ``Memory`` instance
    and automatically persists after each mutating operation.
    """

    def __init__(
        self,
        store: SqliteSessionStore,
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

    def get_persisted_count(self) -> int:
        return self._inner.get_persisted_count()

    def get_new_messages(self) -> list[Message]:
        return self._inner.get_new_messages()

    def mark_all_persisted(self) -> None:
        self._inner.mark_all_persisted()

    def set_persisted_count(self, count: int) -> None:
        self._inner.set_persisted_count(count)

    # -- internal -------------------------------------------------------

    def _auto_save(self) -> None:
        self._store._schedule_save(self)
