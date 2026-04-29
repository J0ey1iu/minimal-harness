"""Persistent memory implementation for the TUI."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from minimal_harness.memory import ConversationMemory, MemoryData, Message
from minimal_harness.types import TokenUsage


class PersistentMemory:
    """Memory that auto-saves to disk and tracks session metadata."""

    def __init__(
        self,
        memory_dir: Path | None = None,
        session_id: str | None = None,
        system_prompt: str = "You are a helpful assistant.",
        agent_name: str = "",
        selected_tools: list[str] | None = None,
    ) -> None:
        self._inner = ConversationMemory(system_prompt=system_prompt)
        self._memory_dir = memory_dir or Path.home() / ".minimal_harness" / "memories"
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        self._session_id = session_id or uuid.uuid4().hex
        self._title: str | None = None
        self._created_at = datetime.now().isoformat()
        self._first_user_message = True
        self._agent_name = agent_name
        self.selected_tools: list[str] = selected_tools or []

    @property
    def title(self) -> str | None:
        return self._title

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def agent_name(self) -> str:
        return self._agent_name

    @agent_name.setter
    def agent_name(self, value: str) -> None:
        self._agent_name = value

    @property
    def created_at(self) -> str:
        return self._created_at

    def add_message(self, message: Message) -> None:
        if self._first_user_message and message.get("role") == "user":
            content = message.get("content", [])
            if content and isinstance(content[0], dict) and "text" in content[0]:
                self._title = content[0]["text"][:100]
            self._first_user_message = False
        self._inner.add_message(message)
        self._flush()

    def get_all_messages(self) -> list[Message]:
        return self._inner.get_all_messages()

    def get_forward_messages(self) -> list[Message]:
        return self._inner.get_forward_messages()

    def clear_messages(self) -> None:
        self._inner.clear_messages()
        self._flush()

    def set_message_usage(self, usage: TokenUsage) -> None:
        self._inner.set_message_usage(usage)
        self._flush()

    def get_message_usage(self) -> TokenUsage:
        return self._inner.get_message_usage()

    def dump_memory(self) -> MemoryData:
        data = self._inner.dump_memory()
        data["extra"]["session_id"] = self._session_id
        data["extra"]["title"] = self._title or "Untitled"
        data["extra"]["created_at"] = self._created_at
        data["extra"]["agent_name"] = self._agent_name
        data["extra"]["selected_tools"] = self.selected_tools
        return data

    def dump_memory_json(self, indent: int | None = 2) -> str:
        return json.dumps(
            self.dump_memory(), indent=indent, ensure_ascii=False, default=str
        )

    def load_memory(self, data: MemoryData) -> None:
        self._inner.load_memory(data)
        extra = data.get("extra", {})
        self._session_id = extra.get("session_id", self._session_id)
        self._title = extra.get("title", self._title)
        self._created_at = extra.get("created_at", self._created_at)
        self._agent_name = extra.get("agent_name", self._agent_name)
        self.selected_tools = list(extra.get("selected_tools", []))
        self._first_user_message = False

    def load_memory_json(self, data: str) -> None:
        parsed: MemoryData = json.loads(data)
        self.load_memory(parsed)

    def update_system_prompt(self, prompt: str) -> None:
        self._inner.update_system_prompt(prompt)
        self._flush()

    def flush(self) -> None:
        self._flush()

    # -- Persistence ---------------------------------------------------------

    def _flush(self) -> None:
        path = self._memory_dir / f"{self._session_id}.json"
        path.write_text(self.dump_memory_json(indent=2), encoding="utf-8")

    @classmethod
    def list_sessions(cls, memory_dir: Path | None = None) -> list[dict[str, Any]]:
        directory = memory_dir or Path.home() / ".minimal_harness" / "memories"
        if not directory.exists():
            return []
        sessions: list[dict[str, Any]] = []
        for path in directory.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                extra = data.get("extra", {})
                sessions.append(
                    {
                        "session_id": extra.get("session_id", path.stem),
                        "title": extra.get("title", "Untitled"),
                        "created_at": extra.get("created_at", ""),
                        "path": str(path),
                        "message_count": len(data.get("messages", [])),
                        "agent_name": extra.get("agent_name", ""),
                    }
                )
            except Exception:
                continue
        sessions.sort(key=lambda s: s.get("created_at") or "", reverse=True)
        return sessions

    @classmethod
    def from_session(
        cls, session_id: str, memory_dir: Path | None = None
    ) -> PersistentMemory:
        directory = memory_dir or Path.home() / ".minimal_harness" / "memories"
        if not re.fullmatch(r"[a-f0-9]{32}", session_id):
            raise ValueError(f"Invalid session_id: {session_id!r}")
        path = (directory / session_id).with_suffix(".json")
        if not path.exists():
            raise FileNotFoundError(f"Session {session_id} not found")
        data: MemoryData = json.loads(path.read_text(encoding="utf-8"))
        memory = cls(memory_dir=directory, session_id=session_id)
        memory.load_memory(data)
        return memory
