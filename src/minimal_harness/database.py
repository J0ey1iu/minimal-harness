from __future__ import annotations

import random
import time
from typing import Any, Protocol, runtime_checkable


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


@runtime_checkable
class DatabaseProtocol(Protocol):
    """Abstract database interface for all services.

    Implement this protocol to support different database backends
    (SQLite, PostgreSQL, etc.) for customer deployment.
    """

    async def init(self, dsn: str) -> None: ...
    async def close(self) -> None: ...
    async def execute(self, sql: str, params: list | None = None) -> Any: ...
    async def execute_write(self, sql: str, params: list | None = None) -> int: ...
    async def execute_many_write(self, sql: str, params_list: list[list]) -> None: ...
    async def fetch_one(self, sql: str, params: list | None = None) -> dict | None: ...
    async def fetch_all(self, sql: str, params: list | None = None) -> list[dict]: ...


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


# ── Register built-in backends ──────────────────────────────────────────────

DatabaseBackend.register("sqlite", SqliteDatabase)
DatabaseBackend.register("opengauss", OpenGaussDatabase)
