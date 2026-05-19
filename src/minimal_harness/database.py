from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DatabaseProtocol(Protocol):
    """Abstract database interface for all services.

    Implement this protocol to support different database backends
    (SQLite, PostgreSQL, etc.) for customer deployment.
    """

    async def init(self, path: str) -> None: ...
    async def close(self) -> None: ...
    async def execute(self, sql: str, params: list | None = None) -> Any: ...
    async def execute_write(self, sql: str, params: list | None = None) -> int: ...
    async def execute_many_write(self, sql: str, params_list: list[list]) -> None: ...
    async def fetch_one(self, sql: str, params: list | None = None) -> dict | None: ...
    async def fetch_all(self, sql: str, params: list | None = None) -> list[dict]: ...


class SqliteDatabase:
    """Default SQLite-backed implementation of DatabaseProtocol."""

    def __init__(self) -> None:
        self._conn: Any = None

    async def init(self, path: str) -> None:
        import aiosqlite

        self._conn = await aiosqlite.connect(path)
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
