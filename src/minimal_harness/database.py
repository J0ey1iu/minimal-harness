from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone
from typing import Any, Protocol, cast, runtime_checkable
from uuid import uuid4

from minimal_harness.memory import Memory, Message
from minimal_harness.memory_store import SessionStoreProtocol
from minimal_harness.session import Session, SessionSummary, SimpleSession


def generate_bigint_id() -> int:
    """Generate a monotonically increasing positive BIGINT ID.

    Uses a snowflake-like scheme:
      - 41 bits: milliseconds since 2025-01-01
      - 22 bits: random jitter for collision resistance
    Result fits in 63 bits (positive BIGINT).
    """
    EPOCH = 1735689600000  # 2025-01-01T00:00:00Z in ms
    elapsed = int(time.time() * 1000) - EPOCH
    jitter = random.getrandbits(22)
    return (elapsed << 22) | jitter


def _ts_ms() -> str:
    """Return ISO 8601 timestamp with millisecond precision."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@runtime_checkable
class DatabaseProtocol(Protocol):
    """Abstract database interface for all services.

    Implement this protocol to support different database backends
    (SQLite, PostgreSQL, etc.) for customer deployment.
    """

    async def init(self, dsn: str) -> None: ...
    async def close(self) -> None: ...
    async def init_schema(self) -> None: ...
    async def create_session_store(self) -> SessionStoreProtocol: ...
    async def execute(self, sql: str, params: list | None = None) -> Any: ...
    async def execute_write(self, sql: str, params: list | None = None) -> int: ...
    async def execute_many_write(self, sql: str, params_list: list[list]) -> None: ...
    async def fetch_one(self, sql: str, params: list | None = None) -> dict | None: ...
    async def fetch_all(self, sql: str, params: list | None = None) -> list[dict]: ...

    # ── Transaction support ──
    async def begin(self) -> None: ...
    async def commit(self) -> None: ...
    async def rollback(self) -> None: ...
    async def executemany(self, sql: str, params_list: list[list]) -> None: ...


class _CursorWrapper:
    """Wraps execute result to provide .rowcount for compatibility."""

    def __init__(self, rowcount: int = 0) -> None:
        self.rowcount = rowcount


class DatabaseBackend:
    """Registry for database backend drivers.

    Built-in backends (``sqlite``, ``opengauss``) are pre-registered.

    Customers can register their own backends for custom deployment::

        from minimal_harness.database import DatabaseBackend

        class MyMySQLDatabase:
            ...

        DatabaseBackend.register("mysql", MyMySQLDatabase)
    """

    _drivers: dict[str, type] = {}

    @classmethod
    def register(cls, name: str, driver_cls: type) -> None:
        cls._drivers[name] = driver_cls

    @classmethod
    def get(cls, name: str) -> type:
        if name not in cls._drivers:
            raise ValueError(
                f"Unknown database backend: {name!r}. "
                f"Available backends: {list(cls._drivers)}"
            )
        return cls._drivers[name]


class SqliteDatabase:
    """Default SQLite-backed implementation of DatabaseProtocol."""

    def __init__(self) -> None:
        self._conn: Any = None

    async def init(self, dsn: str) -> None:
        import aiosqlite

        self._conn = await aiosqlite.connect(dsn)
        self._conn.row_factory = aiosqlite.Row

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()

    async def execute(self, sql: str, params: list | None = None) -> Any:
        assert self._conn is not None
        return await self._conn.execute(sql, params or [])

    async def execute_write(self, sql: str, params: list | None = None) -> int:
        cursor = await self.execute(sql, params)
        assert self._conn is not None
        await self._conn.commit()
        return cursor.lastrowid or 0

    async def execute_many_write(self, sql: str, params_list: list[list]) -> None:
        assert self._conn is not None
        await self._conn.executemany(sql, params_list)
        await self._conn.commit()

    async def fetch_one(self, sql: str, params: list | None = None) -> dict | None:
        cursor = await self.execute(sql, params)
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def fetch_all(self, sql: str, params: list | None = None) -> list[dict]:
        cursor = await self.execute(sql, params)
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ── Transaction support ──

    async def begin(self) -> None:
        assert self._conn is not None
        await self._conn.execute("BEGIN IMMEDIATE")

    async def commit(self) -> None:
        assert self._conn is not None
        await self._conn.commit()

    async def rollback(self) -> None:
        assert self._conn is not None
        await self._conn.rollback()

    async def executemany(self, sql: str, params_list: list[list]) -> None:
        assert self._conn is not None
        await self._conn.executemany(sql, params_list)

    # ── Schema initialisation ──

    async def init_schema(self) -> None:
        try:
            await self.fetch_one("SELECT creation_date FROM sessions LIMIT 1")
        except Exception:
            await self.execute("DROP TABLE IF EXISTS session_messages")
            await self.execute("DROP TABLE IF EXISTS sessions")

        await self.execute(
            """CREATE TABLE IF NOT EXISTS sessions (
                id BIGINT PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                scenario_id TEXT NOT NULL,
                title TEXT DEFAULT '',
                status TEXT DEFAULT 'idle',
                created_by TEXT NOT NULL,
                last_updated_by TEXT NOT NULL,
                creation_date TEXT NOT NULL,
                last_update_date TEXT NOT NULL,
                delete_flag TEXT DEFAULT 'N',
                last_update_trace_id TEXT NOT NULL,
                transient TEXT DEFAULT 'N'
            )"""
        )
        await self.execute(
            """CREATE TABLE IF NOT EXISTS session_messages (
                id BIGINT PRIMARY KEY,
                session_id TEXT NOT NULL,
                data TEXT NOT NULL,
                sort_order INTEGER NOT NULL DEFAULT 0,
                created_by TEXT NOT NULL,
                last_updated_by TEXT NOT NULL,
                creation_date TEXT NOT NULL,
                last_update_date TEXT NOT NULL,
                delete_flag TEXT DEFAULT 'N',
                last_update_trace_id TEXT NOT NULL
            )"""
        )
        await self.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON sessions(session_id)"
        )
        await self.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_messages_session_id ON session_messages(session_id)"
        )

        # Migrate existing tables that lack the transient column
        try:
            await self.fetch_one("SELECT transient FROM sessions LIMIT 1")
        except Exception:
            await self.execute(
                "ALTER TABLE sessions ADD COLUMN transient TEXT DEFAULT 'N'"
            )

        # Migrate existing session_messages that lack the sort_order column
        try:
            await self.fetch_one("SELECT sort_order FROM session_messages LIMIT 1")
        except Exception:
            await self.execute(
                "ALTER TABLE session_messages ADD COLUMN sort_order INTEGER NOT NULL DEFAULT 0"
            )

        await self.execute_write("SELECT 1")

    # ── Session store ──

    async def create_session_store(self) -> SessionStoreProtocol:
        return _SqliteSessionStore(self)


class _SqliteSessionStore:
    """SQLite-backed SessionStoreProtocol implementation.

    Writes SQL with SQLite-native ``?`` placeholders.
    """

    def __init__(self, db: DatabaseProtocol) -> None:
        self._db = db
        self._cache: dict[str, SimpleSession] = {}

    async def create_session(
        self,
        session_id: str | None = None,
        agent_name: str = "",
        user_id: str = "",
        scenario_id: str | None = None,
        transient: bool = False,
    ) -> SimpleSession:
        sid = session_id or f"mem_{uuid4().hex[:12]}"
        now = _ts_ms()
        trace_id = uuid4().hex

        session = SimpleSession(
            session_id=sid,
            agent_name=agent_name,
            user_id=user_id,
            scenario_id=scenario_id,
        )

        await self._db.execute_write(
            """INSERT INTO sessions
               (id, session_id, user_id, agent_name, scenario_id, status,
                created_by, last_updated_by, creation_date, last_update_date,
                delete_flag, last_update_trace_id, transient)
               VALUES (?, ?, ?, ?, ?, 'idle',
                       ?, ?, ?, ?,
                       'N', ?, ?)""",
            [
                session.db_id,
                sid,
                user_id,
                agent_name,
                scenario_id or "",
                user_id,
                user_id,
                now,
                now,
                trace_id,
                "Y" if transient else "N",
            ],
        )

        session.created_at = now
        self._cache[sid] = session
        return session

    async def get_session(self, session_id: str) -> SimpleSession | None:
        if session_id in self._cache:
            return self._cache[session_id]

        row = await self._db.fetch_one(
            "SELECT * FROM sessions WHERE session_id = ? AND delete_flag = 'N'",
            [session_id],
        )
        if row is None:
            return None

        session = SimpleSession(
            session_id=row["session_id"],
            agent_name=row["agent_name"],
            user_id=row["user_id"],
            scenario_id=row["scenario_id"],
        )
        session.db_id = row["id"]
        session.created_at = row["creation_date"]
        session.title = row.get("title")

        msg_rows = await self._db.fetch_all(
            "SELECT data FROM session_messages WHERE session_id = ? AND delete_flag = 'N' ORDER BY sort_order, id",
            [session_id],
        )
        for m in msg_rows:
            msg_data = json.loads(m["data"])
            if isinstance(msg_data, dict):
                await session.add_message(cast("Message", msg_data))

        session.memory.set_persisted_count(len(msg_rows))

        self._cache[session_id] = session
        return session

    async def save_memory(
        self, memory: Memory, session_id: str, extra: dict[str, Any] | None = None
    ) -> None:
        now = _ts_ms()
        trace_id = uuid4().hex
        new_msgs = memory.get_new_messages()
        title = (extra or {}).get("title")

        if not new_msgs and not title:
            return

        session_row = await self._db.fetch_one(
            "SELECT user_id FROM sessions WHERE session_id = ? AND delete_flag = 'N'",
            [session_id],
        )
        owner_id = session_row["user_id"] if session_row else "unknown"

        base_order = memory.get_persisted_count()

        await self._db.begin()
        try:
            if new_msgs:
                rows = []
                for idx, m in enumerate(new_msgs):
                    mid = generate_bigint_id()
                    rows.append(
                        [
                            mid,
                            session_id,
                            json.dumps(m, ensure_ascii=False),
                            base_order + idx,
                            owner_id,
                            owner_id,
                            now,
                            now,
                            "N",
                            trace_id,
                        ]
                    )
                await self._db.executemany(
                    """INSERT INTO session_messages
                       (id, session_id, data, sort_order,
                        created_by, last_updated_by, creation_date, last_update_date,
                        delete_flag, last_update_trace_id)
                       VALUES (?, ?, ?, ?,
                               ?, ?, ?, ?,
                               ?, ?)""",
                    rows,
                )

            if title:
                await self._db.execute(
                    "UPDATE sessions SET title = ?, last_updated_by = ?, last_update_date = ?, status = 'idle', last_update_trace_id = ? WHERE session_id = ?",
                    [title, owner_id, now, trace_id, session_id],
                )
            else:
                await self._db.execute(
                    "UPDATE sessions SET last_updated_by = ?, last_update_date = ?, status = 'idle', last_update_trace_id = ? WHERE session_id = ?",
                    [owner_id, now, trace_id, session_id],
                )

            await self._db.commit()
        except Exception:
            await self._db.rollback()
            raise

        if new_msgs:
            memory.mark_all_persisted()

    async def delete_session(self, session_id: str) -> bool:
        self._cache.pop(session_id, None)
        now = _ts_ms()
        trace_id = uuid4().hex

        await self._db.execute_write(
            "UPDATE session_messages SET delete_flag = 'Y', last_updated_by = ?, last_update_date = ?, last_update_trace_id = ? WHERE session_id = ?",
            ["system", now, trace_id, session_id],
        )
        cur = await self._db.execute(
            "UPDATE sessions SET delete_flag = 'Y', last_updated_by = ?, last_update_date = ?, last_update_trace_id = ?, status = 'deleted' WHERE session_id = ? AND delete_flag = 'N'",
            ["system", now, trace_id, session_id],
        )
        return cur.rowcount > 0

    async def list_sessions(self) -> list[SessionSummary]:
        rows = await self._db.fetch_all(
            """SELECT s.session_id, s.agent_name, s.user_id, s.scenario_id, s.title, s.creation_date, s.status,
                      (SELECT COUNT(*) FROM session_messages m WHERE m.session_id = s.session_id AND m.delete_flag = 'N') AS message_count
               FROM sessions s
               WHERE s.delete_flag = 'N' AND s.transient = 'N'
               ORDER BY s.creation_date DESC"""
        )
        result: list[SessionSummary] = []
        for r in rows:
            result.append(
                SessionSummary(
                    session_id=r["session_id"],
                    agent_name=r["agent_name"],
                    user_id=r["user_id"],
                    scenario_id=r["scenario_id"],
                    title=r.get("title"),
                    created_at=r["creation_date"],
                    message_count=r["message_count"],
                    status=r["status"],
                )
            )
        return result

    async def list_user_sessions(
        self, user_id: str, scenario_id: str | None = None
    ) -> list[SessionSummary]:
        if scenario_id:
            rows = await self._db.fetch_all(
                """SELECT s.session_id, s.agent_name, s.user_id, s.scenario_id, s.title, s.creation_date, s.status,
                          (SELECT COUNT(*) FROM session_messages m WHERE m.session_id = s.session_id AND m.delete_flag = 'N') AS message_count
                   FROM sessions s
                   WHERE s.user_id = ? AND s.scenario_id = ? AND s.delete_flag = 'N' AND s.transient = 'N'
                   ORDER BY s.creation_date DESC""",
                [user_id, scenario_id],
            )
        else:
            rows = await self._db.fetch_all(
                """SELECT s.session_id, s.agent_name, s.user_id, s.scenario_id, s.title, s.creation_date, s.status,
                          (SELECT COUNT(*) FROM session_messages m WHERE m.session_id = s.session_id AND m.delete_flag = 'N') AS message_count
                   FROM sessions s
                   WHERE s.user_id = ? AND s.delete_flag = 'N' AND s.transient = 'N'
                   ORDER BY s.creation_date DESC""",
                [user_id],
            )
        result: list[SessionSummary] = []
        for r in rows:
            result.append(
                SessionSummary(
                    session_id=r["session_id"],
                    agent_name=r["agent_name"],
                    user_id=r["user_id"],
                    scenario_id=r["scenario_id"],
                    title=r.get("title"),
                    created_at=r["creation_date"],
                    message_count=r["message_count"],
                    status=r["status"],
                )
            )
        return result

    async def get_session_messages(self, session_id: str) -> list[dict]:
        rows = await self._db.fetch_all(
            "SELECT data FROM session_messages WHERE session_id = ? AND delete_flag = 'N' ORDER BY sort_order, id",
            [session_id],
        )
        return [json.loads(r["data"]) for r in rows]

    def get_messages_as_items(self, session: Session) -> list[dict]:
        items: list[dict] = []
        for i, msg in enumerate(session.get_all_messages()):
            role = msg.get("role", "")
            content = msg.get("content")
            if content is None:
                content = None
            elif isinstance(content, list):
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
                    "meta": msg.get("meta"),
                }
            )
        return items


class OpenGaussDatabase:
    """openGauss-backed implementation of DatabaseProtocol.

    Uses async_gaussdb under the hood.  Placeholder style ``?`` is automatically
    converted to ``$1, $2, …`` for async_gaussdb compatibility.
    """

    def __init__(self) -> None:
        self._conn: Any = None

    async def init(self, dsn: str) -> None:
        import async_gaussdb

        self._conn = await async_gaussdb.connect(dsn)

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()

    @staticmethod
    def _convert(sql: str) -> str:
        """Convert ``?`` placeholders to ``$1, $2, …``."""
        i = 0
        buf: list[str] = []
        for ch in sql:
            if ch == "?":
                i += 1
                buf.append(f"${i}")
            else:
                buf.append(ch)
        return "".join(buf)

    async def execute(self, sql: str, params: list | None = None) -> Any:
        assert self._conn is not None
        converted = self._convert(sql)
        status = await self._conn.execute(converted, *(params or []))
        return _CursorWrapper(self._parse_rowcount(status, converted))

    async def execute_write(self, sql: str, params: list | None = None) -> int:
        assert self._conn is not None
        converted = self._convert(sql)
        status = await self._conn.execute(converted, *(params or []))
        return self._parse_rowcount(status, converted)

    async def execute_many_write(self, sql: str, params_list: list[list]) -> None:
        assert self._conn is not None
        converted = self._convert(sql)
        await self._conn.executemany(converted, params_list)

    async def fetch_one(self, sql: str, params: list | None = None) -> dict | None:
        assert self._conn is not None
        converted = self._convert(sql)
        row = await self._conn.fetchrow(converted, *(params or []))
        return dict(row) if row else None

    async def fetch_all(self, sql: str, params: list | None = None) -> list[dict]:
        assert self._conn is not None
        converted = self._convert(sql)
        rows = await self._conn.fetch(converted, *(params or []))
        return [dict(r) for r in rows]

    # ── Transaction support ──

    async def begin(self) -> None:
        assert self._conn is not None
        await self._conn.execute("BEGIN")

    async def commit(self) -> None:
        assert self._conn is not None
        await self._conn.commit()

    async def rollback(self) -> None:
        assert self._conn is not None
        await self._conn.rollback()

    async def executemany(self, sql: str, params_list: list[list]) -> None:
        assert self._conn is not None
        await self._conn.executemany(self._convert(sql), params_list)

    @staticmethod
    def _parse_rowcount(status: str, sql: str) -> int:
        """Parse command tag string like ``INSERT 0 1`` -> 1."""
        parts = status.split()
        if len(parts) >= 2:
            try:
                return int(parts[-1])
            except ValueError:
                pass
        return 0

    # ── Schema initialisation ──

    async def init_schema(self) -> None:
        try:
            await self.fetch_one("SELECT creation_date FROM sessions LIMIT 1")
        except Exception:
            await self.execute("DROP TABLE IF EXISTS session_messages")
            await self.execute("DROP TABLE IF EXISTS sessions")

        await self.execute(
            """CREATE TABLE IF NOT EXISTS sessions (
                id BIGINT PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                scenario_id TEXT NOT NULL,
                title TEXT DEFAULT '',
                status TEXT DEFAULT 'idle',
                created_by TEXT NOT NULL,
                last_updated_by TEXT NOT NULL,
                creation_date TIMESTAMP(3) WITH TIME ZONE NOT NULL,
                last_update_date TIMESTAMP(3) WITH TIME ZONE NOT NULL,
                delete_flag CHAR(1) DEFAULT 'N',
                last_update_trace_id TEXT NOT NULL,
                transient CHAR(1) DEFAULT 'N'
            )"""
        )
        await self.execute(
            """CREATE TABLE IF NOT EXISTS session_messages (
                id BIGINT PRIMARY KEY,
                session_id TEXT NOT NULL,
                data TEXT NOT NULL,
                sort_order INTEGER NOT NULL DEFAULT 0,
                created_by TEXT NOT NULL,
                last_updated_by TEXT NOT NULL,
                creation_date TIMESTAMP(3) WITH TIME ZONE NOT NULL,
                last_update_date TIMESTAMP(3) WITH TIME ZONE NOT NULL,
                delete_flag CHAR(1) DEFAULT 'N',
                last_update_trace_id TEXT NOT NULL
            )"""
        )
        await self.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON sessions(session_id)"
        )
        await self.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_messages_session_id ON session_messages(session_id)"
        )

        # Migrate existing tables that lack the transient column
        try:
            await self.fetch_one("SELECT transient FROM sessions LIMIT 1")
        except Exception:
            await self.execute(
                "ALTER TABLE sessions ADD COLUMN transient CHAR(1) DEFAULT 'N'"
            )

        # Migrate existing session_messages that lack the sort_order column
        try:
            await self.fetch_one("SELECT sort_order FROM session_messages LIMIT 1")
        except Exception:
            await self.execute(
                "ALTER TABLE session_messages ADD COLUMN sort_order INTEGER NOT NULL DEFAULT 0"
            )

        await self.execute_write("SELECT 1")

    # ── Session store ──

    async def create_session_store(self) -> SessionStoreProtocol:
        return _OpenGaussSessionStore(self)


class _OpenGaussSessionStore:
    """openGauss-backed SessionStoreProtocol implementation.

    Writes SQL with native ``$1, $2, …`` placeholders (no placeholder
    conversion needed).
    """

    def __init__(self, db: DatabaseProtocol) -> None:
        self._db = db
        self._cache: dict[str, SimpleSession] = {}

    async def create_session(
        self,
        session_id: str | None = None,
        agent_name: str = "",
        user_id: str = "",
        scenario_id: str | None = None,
        transient: bool = False,
    ) -> SimpleSession:
        sid = session_id or f"mem_{uuid4().hex[:12]}"
        now = _ts_ms()
        trace_id = uuid4().hex

        session = SimpleSession(
            session_id=sid,
            agent_name=agent_name,
            user_id=user_id,
            scenario_id=scenario_id,
        )

        await self._db.execute_write(
            """INSERT INTO sessions
               (id, session_id, user_id, agent_name, scenario_id, status,
                created_by, last_updated_by, creation_date, last_update_date,
                delete_flag, last_update_trace_id, transient)
               VALUES ($1, $2, $3, $4, $5, 'idle',
                       $6, $7, $8, $9,
                       'N', $10, $11)""",
            [
                session.db_id,
                sid,
                user_id,
                agent_name,
                scenario_id or "",
                user_id,
                user_id,
                now,
                now,
                trace_id,
                "Y" if transient else "N",
            ],
        )

        session.created_at = now
        self._cache[sid] = session
        return session

    async def get_session(self, session_id: str) -> SimpleSession | None:
        if session_id in self._cache:
            return self._cache[session_id]

        row = await self._db.fetch_one(
            "SELECT * FROM sessions WHERE session_id = $1 AND delete_flag = 'N'",
            [session_id],
        )
        if row is None:
            return None

        session = SimpleSession(
            session_id=row["session_id"],
            agent_name=row["agent_name"],
            user_id=row["user_id"],
            scenario_id=row["scenario_id"],
        )
        session.db_id = row["id"]
        session.created_at = row["creation_date"]
        session.title = row.get("title")

        msg_rows = await self._db.fetch_all(
            "SELECT data FROM session_messages WHERE session_id = $1 AND delete_flag = 'N' ORDER BY sort_order, id",
            [session_id],
        )
        for m in msg_rows:
            msg_data = json.loads(m["data"])
            if isinstance(msg_data, dict):
                await session.add_message(cast("Message", msg_data))

        session.memory.set_persisted_count(len(msg_rows))

        self._cache[session_id] = session
        return session

    async def save_memory(
        self, memory: Memory, session_id: str, extra: dict[str, Any] | None = None
    ) -> None:
        now = _ts_ms()
        trace_id = uuid4().hex
        new_msgs = memory.get_new_messages()
        title = (extra or {}).get("title")

        if not new_msgs and not title:
            return

        session_row = await self._db.fetch_one(
            "SELECT user_id FROM sessions WHERE session_id = $1 AND delete_flag = 'N'",
            [session_id],
        )
        owner_id = session_row["user_id"] if session_row else "unknown"

        base_order = memory.get_persisted_count()

        await self._db.begin()
        try:
            if new_msgs:
                rows = []
                for idx, m in enumerate(new_msgs):
                    mid = generate_bigint_id()
                    rows.append(
                        [
                            mid,
                            session_id,
                            json.dumps(m, ensure_ascii=False),
                            base_order + idx,
                            owner_id,
                            owner_id,
                            now,
                            now,
                            "N",
                            trace_id,
                        ]
                    )
                await self._db.executemany(
                    """INSERT INTO session_messages
                       (id, session_id, data, sort_order,
                        created_by, last_updated_by, creation_date, last_update_date,
                        delete_flag, last_update_trace_id)
                       VALUES ($1, $2, $3, $4,
                               $5, $6, $7, $8,
                               $9, $10)""",
                    rows,
                )

            if title:
                await self._db.execute(
                    "UPDATE sessions SET title = $1, last_updated_by = $2, last_update_date = $3, status = 'idle', last_update_trace_id = $4 WHERE session_id = $5",
                    [title, owner_id, now, trace_id, session_id],
                )
            else:
                await self._db.execute(
                    "UPDATE sessions SET last_updated_by = $1, last_update_date = $2, status = 'idle', last_update_trace_id = $3 WHERE session_id = $4",
                    [owner_id, now, trace_id, session_id],
                )

            await self._db.commit()
        except Exception:
            await self._db.rollback()
            raise

        if new_msgs:
            memory.mark_all_persisted()

    async def delete_session(self, session_id: str) -> bool:
        self._cache.pop(session_id, None)
        now = _ts_ms()
        trace_id = uuid4().hex

        await self._db.execute_write(
            "UPDATE session_messages SET delete_flag = 'Y', last_updated_by = $1, last_update_date = $2, last_update_trace_id = $3 WHERE session_id = $4",
            ["system", now, trace_id, session_id],
        )
        cur = await self._db.execute(
            "UPDATE sessions SET delete_flag = 'Y', last_updated_by = $1, last_update_date = $2, last_update_trace_id = $3, status = 'deleted' WHERE session_id = $4 AND delete_flag = 'N'",
            ["system", now, trace_id, session_id],
        )
        return cur.rowcount > 0

    async def list_sessions(self) -> list[SessionSummary]:
        rows = await self._db.fetch_all(
            """SELECT s.session_id, s.agent_name, s.user_id, s.scenario_id, s.title, s.creation_date, s.status,
                      (SELECT COUNT(*) FROM session_messages m WHERE m.session_id = s.session_id AND m.delete_flag = 'N') AS message_count
               FROM sessions s
               WHERE s.delete_flag = 'N' AND s.transient = 'N'
               ORDER BY s.creation_date DESC"""
        )
        result: list[SessionSummary] = []
        for r in rows:
            result.append(
                SessionSummary(
                    session_id=r["session_id"],
                    agent_name=r["agent_name"],
                    user_id=r["user_id"],
                    scenario_id=r["scenario_id"],
                    title=r.get("title"),
                    created_at=r["creation_date"],
                    message_count=r["message_count"],
                    status=r["status"],
                )
            )
        return result

    async def list_user_sessions(
        self, user_id: str, scenario_id: str | None = None
    ) -> list[SessionSummary]:
        if scenario_id:
            rows = await self._db.fetch_all(
                """SELECT s.session_id, s.agent_name, s.user_id, s.scenario_id, s.title, s.creation_date, s.status,
                          (SELECT COUNT(*) FROM session_messages m WHERE m.session_id = s.session_id AND m.delete_flag = 'N') AS message_count
                   FROM sessions s
                   WHERE s.user_id = $1 AND s.scenario_id = $2 AND s.delete_flag = 'N' AND s.transient = 'N'
                   ORDER BY s.creation_date DESC""",
                [user_id, scenario_id],
            )
        else:
            rows = await self._db.fetch_all(
                """SELECT s.session_id, s.agent_name, s.user_id, s.scenario_id, s.title, s.creation_date, s.status,
                          (SELECT COUNT(*) FROM session_messages m WHERE m.session_id = s.session_id AND m.delete_flag = 'N') AS message_count
                   FROM sessions s
                   WHERE s.user_id = $1 AND s.delete_flag = 'N' AND s.transient = 'N'
                   ORDER BY s.creation_date DESC""",
                [user_id],
            )
        result: list[SessionSummary] = []
        for r in rows:
            result.append(
                SessionSummary(
                    session_id=r["session_id"],
                    agent_name=r["agent_name"],
                    user_id=r["user_id"],
                    scenario_id=r["scenario_id"],
                    title=r.get("title"),
                    created_at=r["creation_date"],
                    message_count=r["message_count"],
                    status=r["status"],
                )
            )
        return result

    async def get_session_messages(self, session_id: str) -> list[dict]:
        rows = await self._db.fetch_all(
            "SELECT data FROM session_messages WHERE session_id = $1 AND delete_flag = 'N' ORDER BY sort_order, id",
            [session_id],
        )
        return [json.loads(r["data"]) for r in rows]

    def get_messages_as_items(self, session: Session) -> list[dict]:
        items: list[dict] = []
        for i, msg in enumerate(session.get_all_messages()):
            role = msg.get("role", "")
            content = msg.get("content")
            if content is None:
                content = None
            elif isinstance(content, list):
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
                    "meta": msg.get("meta"),
                }
            )
        return items


# ── Register built-in backends ──────────────────────────────────────────────

DatabaseBackend.register("sqlite", SqliteDatabase)
DatabaseBackend.register("opengauss", OpenGaussDatabase)
