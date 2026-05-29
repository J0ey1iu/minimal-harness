"""JSON file-backed session store — one JSON file per session.

Replaces ``SqliteSessionStore`` with simple atomic JSON writes.
Each session is a single ``.json`` file under ``{config_dir}/sessions/``.
A lightweight ``_index.json`` keeps metadata for fast listing without
scanning all session files.

On ``add_message`` every role is persisted immediately to prevent the
debounce-race where a session reload happens before the scheduled save.

``set_message_usage`` uses a debounced fire-and-forget task (no guard)
to avoid write thrash during streaming token-usage updates.
"""

from __future__ import annotations

import asyncio
import json
import logging
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
)
from minimal_harness.memory_store import MemoryFactory
from minimal_harness.session import Session, SessionSummary
from minimal_harness.types import TokenUsage

logger = logging.getLogger(__name__)


class JsonlSessionStore:
    """Persistence layer backed by local JSON files.

    Each session is stored as ``{sessions_dir}/{session_id}.json``.
    Atomic writes use temp files + ``os.replace()``.
    """

    def __init__(
        self,
        sessions_dir: Path | str | None = None,
        memory_factory: MemoryFactory | None = None,
    ) -> None:
        if sessions_dir:
            self._dir = Path(sessions_dir)
        else:
            from minimal_harness.client.built_in.config.paths import get_config_dir

            self._dir = get_config_dir() / "sessions"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, JsonlManagedSession] = {}
        self._memory_factory = memory_factory or (lambda: ConversationMemory())
        self._transient: set[str] = set()
        self._write_locks: dict[str, asyncio.Lock] = {}
        self._index: dict[str, dict] = {}
        self._index_loaded = False
        logger.debug("store.init dir=%s cached=%d", self._dir, len(self._cache))

    # ── index ──────────────────────────────────────────────────────────

    @property
    def _index_path(self) -> Path:
        return self._dir / "_index.json"

    def _load_index(self) -> dict[str, dict]:
        if self._index_loaded:
            return self._index
        if self._index_path.exists():
            try:
                self._index = json.loads(self._index_path.read_text("utf-8"))
                logger.debug(
                    "index.loaded path=%s entries=%d",
                    self._index_path,
                    len(self._index),
                )
            except Exception as exc:
                logger.debug("index.load.error path=%s err=%s", self._index_path, exc)
                self._index = {}
        else:
            logger.debug("index.missing path=%s", self._index_path)
        self._index_loaded = True
        return self._index

    async def _save_index(self) -> None:
        tmp = self._index_path.with_suffix(".json.tmp")
        content = json.dumps(self._index, ensure_ascii=False, default=str)
        tmp.write_text(content, "utf-8")
        os.replace(str(tmp), str(self._index_path))
        logger.debug(
            "index.saved path=%s entries=%d bytes=%d",
            self._index_path,
            len(self._index),
            len(content),
        )

    # ── CRUD ────────────────────────────────────────────────────────────

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
        managed = JsonlManagedSession(
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
        logger.debug(
            "session.create id=%s agent=%s transient=%s cached=%d",
            mid,
            agent_name,
            transient,
            len(self._cache),
        )
        return managed

    async def get_session(self, session_id: str) -> Session | None:
        cached = self._cache.get(session_id)
        if cached is not None:
            logger.debug(
                "session.get.cache-hit id=%s msgs=%d",
                session_id,
                len(cached.get_all_messages()),
            )
            return cached

        path = self._session_path(session_id)
        if not path.exists():
            logger.debug("session.get.missing-file id=%s path=%s", session_id, path)
            return None

        try:
            data: MemoryData = json.loads(path.read_text("utf-8"))
        except Exception as exc:
            logger.debug("session.get.parse-error id=%s err=%s", session_id, exc)
            return None

        inner = self._memory_factory()
        inner.load_memory(data)
        extra = data.get("extra", {})
        is_transient = extra.get("transient", False) or session_id in self._transient

        managed = JsonlManagedSession(
            store=self,
            session_id=session_id,
            inner=inner,
            agent_name=extra.get("agent_name", ""),
            user_id=extra.get("user_id", ""),
            scenario_id=extra.get("scenario_id", None),
            transient=is_transient,
        )
        managed._title = extra.get("title")
        managed._created_at = extra.get("created_at", datetime.now().isoformat())
        managed._first_user_message = False
        if is_transient:
            self._transient.add(session_id)
        self._cache[session_id] = managed
        logger.debug(
            "session.get.disk id=%s msgs=%d title=%s cached=%d",
            session_id,
            len(managed.get_all_messages()),
            managed._title,
            len(self._cache),
        )
        return managed

    async def save_memory(
        self, memory: Memory, session_id: str, extra: dict[str, Any] | None = None
    ) -> None:
        lock = self._write_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            if session_id not in self._cache:
                logger.debug(
                    "save.skip-not-cached id=%s cached=%s",
                    session_id,
                    list(self._cache.keys()),
                )
                return
            data = memory.dump_memory()
            existing = data.get("extra", {})
            merged_extra = {**existing, **(extra or {})}
            data["extra"] = merged_extra
            msg_count = len(data.get("messages", []))

            path = self._session_path(session_id)
            tmp = path.with_suffix(".json.tmp")
            content = json.dumps(data, ensure_ascii=False, default=str)
            tmp.write_text(content, "utf-8")
            os.replace(str(tmp), str(path))

            index = self._load_index()
            index[session_id] = {
                "session_id": session_id,
                "title": merged_extra.get("title"),
                "agent_name": merged_extra.get("agent_name", ""),
                "user_id": merged_extra.get("user_id", ""),
                "scenario_id": merged_extra.get("scenario_id", None),
                "created_at": merged_extra.get("created_at", ""),
                "message_count": msg_count,
                "transient": merged_extra.get("transient", False),
            }
            await self._save_index()

            last_role = (
                data["messages"][-1].get("role", "?")
                if data.get("messages")
                else "empty"
            )
            logger.debug(
                "session.saved id=%s msgs=%d bytes=%d last_role=%s",
                session_id,
                msg_count,
                len(content),
                last_role,
            )

    async def delete_session(self, session_id: str) -> bool:
        self._cache.pop(session_id, None)
        self._transient.discard(session_id)

        path = self._session_path(session_id)
        existed = path.exists()
        if existed:
            path.unlink()
            logger.debug("session.deleted id=%s", session_id)

        if existed:
            index = self._load_index()
            index.pop(session_id, None)
            await self._save_index()

        return existed

    async def list_sessions(self) -> list[SessionSummary]:
        index = self._load_index()
        result: list[SessionSummary] = []
        for sid, entry in index.items():
            if entry.get("transient"):
                continue
            result.append(
                {
                    "session_id": sid,
                    "title": entry.get("title", "Untitled"),
                    "created_at": entry.get("created_at", ""),
                    "message_count": entry.get("message_count", 0),
                    "agent_name": entry.get("agent_name", ""),
                    "user_id": entry.get("user_id", ""),
                    "scenario_id": entry.get("scenario_id", None),
                    "status": "idle",
                }
            )
        result.sort(key=lambda s: s.get("created_at") or "", reverse=True)
        logger.debug("session.list total=%d", len(result))
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

    async def flush(self) -> None:
        pending = []
        for session in self._cache.values():
            task = session._pending_save
            if task is not None and not task.done():
                pending.append(task)
        if pending:
            for task in pending:
                try:
                    await task
                except Exception:
                    pass
        logger.debug("store.flush cached=%d pending=%d", len(self._cache), len(pending))

    async def close(self) -> None:
        logger.debug("store.close noop cached=%d", len(self._cache))

    # ── helpers ───────────────────────────────────────────────────────

    def _session_path(self, session_id: str) -> Path:
        return self._dir / f"{session_id}.json"


class JsonlManagedSession:
    """Session wrapper that auto-persists via ``JsonlSessionStore``.

    Implements both the ``Memory`` and ``Session`` protocols.
    Delegates message operations to an inner ``Memory`` instance.

    ``add_message()`` persists **immediately** for every role (user, tool,
    assistant, reasoning) so the on-disk state never lags behind the
    in-memory buffer.  ``set_message_usage()`` uses a debounced
    fire-and-forget task (no dedup guard) to avoid write thrash during
    streaming token updates.
    """

    _SAVE_DEBOUNCE = 0.05  # seconds

    def __init__(
        self,
        store: JsonlSessionStore,
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
        self._pending_save: asyncio.Task | None = None

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

    async def add_message(self, message: Message) -> None:
        role = message.get("role", "?")
        content_preview: str = ""
        if isinstance(message.get("content"), str):
            content_preview = message["content"][:80]  # type: ignore[arg-type]
        elif isinstance(message.get("content"), list):
            teasers = [
                p.get("text", "")[:40]
                for p in message["content"]  # type: ignore[union-attr]
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            content_preview = " | ".join(teasers)[:80]

        if self._first_user_message and role == "user":
            self._title = content_preview[:100] if content_preview else None
            self._first_user_message = False

        await self._inner.add_message(message)
        logger.debug(
            "managed.add_message id=%s role=%s msgs=%d preview=%.80s",
            self._session_id,
            role,
            len(self._inner.get_all_messages()),
            content_preview,
        )

        await self._store.save_memory(
            memory=self,
            session_id=self._session_id,
            extra=self._make_extra(),
        )

    def get_all_messages(self) -> list[Message]:
        return self._inner.get_all_messages()

    def get_forward_messages(self) -> list[Message]:
        return self._inner.get_forward_messages()

    def clear_messages(self) -> None:
        logger.debug("managed.clear_messages id=%s", self._session_id)
        self._inner.clear_messages()
        self._schedule_save()

    def set_message_usage(self, usage: TokenUsage) -> None:
        self._inner.set_message_usage(usage)
        self._schedule_save()

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

    def _make_extra(self) -> dict[str, Any]:
        return {
            "memory_id": self._session_id,
            "title": self._title,
            "created_at": self._created_at,
            "agent_name": self.agent_name,
            "user_id": self._user_id,
            "scenario_id": self._scenario_id,
            "transient": self._transient,
        }

    def _schedule_save(self) -> None:
        extra = self._make_extra()

        async def _do_save() -> None:
            await asyncio.sleep(self._SAVE_DEBOUNCE)
            await self._store.save_memory(
                memory=self,
                session_id=self._session_id,
                extra=extra,
            )

        try:
            loop = asyncio.get_running_loop()
            self._pending_save = loop.create_task(_do_save())
        except RuntimeError:
            logger.debug("managed.schedule-save.skip-no-loop id=%s", self._session_id)
