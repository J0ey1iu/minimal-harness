from __future__ import annotations

import json
from pathlib import Path

import pytest

from minimal_harness.client.built_in import memory as mem_mod
from minimal_harness.client.built_in.memory import PersistentMemory
from minimal_harness.types import TokenUsage


class TestPersistentMemoryInit:
    def test_creates_memory_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        mem_dir = tmp_path / "memories"
        monkeypatch.setattr(mem_mod, "Path", lambda: tmp_path)
        pm = PersistentMemory(memory_dir=mem_dir, session_id="test-session")
        assert mem_dir.exists()
        assert pm._session_id == "test-session"
        assert pm.title is None

    def test_generates_session_id(self, tmp_path: Path):
        pm = PersistentMemory(memory_dir=tmp_path)
        assert pm._session_id is not None
        assert len(pm._session_id) > 0

    def test_system_prompt_set(self, tmp_path: Path):
        pm = PersistentMemory(memory_dir=tmp_path, system_prompt="You are a bot.")
        msgs = pm.get_all_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are a bot."


class TestPersistentMemoryMessages:
    def test_add_user_message_sets_title(self, tmp_path: Path):
        pm = PersistentMemory(memory_dir=tmp_path, system_prompt="system")
        pm.add_message(
            {"role": "user", "content": [{"type": "text", "text": "Hello world"}]}
        )
        assert pm.title == "Hello world"

    def test_title_truncated_to_50_chars(self, tmp_path: Path):
        pm = PersistentMemory(memory_dir=tmp_path, system_prompt="system")
        long_text = "a" * 100
        pm.add_message(
            {"role": "user", "content": [{"type": "text", "text": long_text}]}
        )
        assert pm.title is not None
        assert len(pm.title) == 50

    def test_first_user_message_only_sets_title_once(self, tmp_path: Path):
        pm = PersistentMemory(memory_dir=tmp_path, system_prompt="system")
        pm.add_message({"role": "user", "content": [{"type": "text", "text": "first"}]})
        assert pm.title == "first"
        pm.add_message(
            {"role": "user", "content": [{"type": "text", "text": "second"}]}
        )
        assert pm.title == "first"

    def test_get_all_messages_returns_system_and_messages(self, tmp_path: Path):
        pm = PersistentMemory(memory_dir=tmp_path, system_prompt="You are helpful.")
        pm.add_message({"role": "user", "content": [{"type": "text", "text": "hi"}]})
        msgs = pm.get_all_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are helpful."

    def test_clear_messages(self, tmp_path: Path):
        pm = PersistentMemory(memory_dir=tmp_path, system_prompt="system")
        pm.add_message({"role": "user", "content": [{"type": "text", "text": "hi"}]})
        pm.clear_messages()
        msgs = pm.get_all_messages()
        assert all(m["role"] != "user" for m in msgs)

    def test_get_forward_messages(self, tmp_path: Path):
        pm = PersistentMemory(memory_dir=tmp_path, system_prompt="system")
        pm.add_message({"role": "user", "content": [{"type": "text", "text": "hi"}]})
        fwd = pm.get_forward_messages()
        assert len(fwd) > 0

    def test_message_usage(self, tmp_path: Path):
        pm = PersistentMemory(memory_dir=tmp_path, system_prompt="system")
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        pm.set_message_usage(usage)
        assert pm.get_message_usage() == usage


class TestPersistentMemoryDumpLoad:
    def test_dump_includes_metadata(self, tmp_path: Path):
        pm = PersistentMemory(
            memory_dir=tmp_path, session_id="sid-1", system_prompt="system"
        )
        pm.add_message({"role": "user", "content": [{"type": "text", "text": "hi"}]})
        data = pm.dump_memory()
        assert data["extra"]["session_id"] == "sid-1"
        assert "title" in data["extra"]
        assert "created_at" in data["extra"]

    def test_dump_memory_json(self, tmp_path: Path):
        pm = PersistentMemory(
            memory_dir=tmp_path, session_id="sid-1", system_prompt="system"
        )
        s = pm.dump_memory_json()
        data = json.loads(s)
        assert data["extra"]["session_id"] == "sid-1"

    def test_load_memory_restores_state(self, tmp_path: Path):
        pm = PersistentMemory(
            memory_dir=tmp_path, session_id="sid-1", system_prompt="system"
        )
        pm.add_message({"role": "user", "content": [{"type": "text", "text": "hello"}]})
        data = pm.dump_memory()

        pm2 = PersistentMemory(
            memory_dir=tmp_path, session_id="sid-2", system_prompt="system"
        )
        pm2.load_memory(data)
        assert pm2._session_id == "sid-1"
        assert len(pm2.get_all_messages()) == 2

    def test_load_memory_json(self, tmp_path: Path):
        pm = PersistentMemory(
            memory_dir=tmp_path, session_id="sid-1", system_prompt="system"
        )
        pm.add_message({"role": "user", "content": [{"type": "text", "text": "hi"}]})
        data_str = pm.dump_memory_json()

        pm2 = PersistentMemory(
            memory_dir=tmp_path, session_id="sid-2", system_prompt="system"
        )
        pm2.load_memory_json(data_str)
        assert len(pm2.get_all_messages()) == 2


class TestUpdateSystemPrompt:
    def test_updates_system_prompt(self, tmp_path: Path):
        pm = PersistentMemory(memory_dir=tmp_path, system_prompt="Old prompt")
        pm.update_system_prompt("New prompt")
        msgs = pm.get_all_messages()
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "New prompt"


class TestFromSession:
    def test_invalid_session_id_path_traversal(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Invalid session_id"):
            PersistentMemory.from_session("../malicious", memory_dir=tmp_path)

    def test_invalid_session_id_with_slash(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Invalid session_id"):
            PersistentMemory.from_session("a/b", memory_dir=tmp_path)

    def test_invalid_session_id_with_backslash(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Invalid session_id"):
            PersistentMemory.from_session("a\\b", memory_dir=tmp_path)

    def test_session_not_found(self, tmp_path: Path):
        hex_id = "a" * 32
        with pytest.raises(FileNotFoundError, match=f"Session {hex_id} not found"):
            PersistentMemory.from_session(hex_id, memory_dir=tmp_path)

    def test_loads_session_successfully(self, tmp_path: Path):
        hex_id = "a" * 32
        pm = PersistentMemory(
            memory_dir=tmp_path, session_id=hex_id, system_prompt="system"
        )
        pm.add_message({"role": "user", "content": [{"type": "text", "text": "hello"}]})

        loaded = PersistentMemory.from_session(hex_id, memory_dir=tmp_path)
        assert loaded.session_id == hex_id
        assert len(loaded.get_all_messages()) == 2


class TestListSessions:
    def test_empty_directory(self, tmp_path: Path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir(parents=True)
        sessions = PersistentMemory.list_sessions(memory_dir=empty_dir)
        assert sessions == []

    def test_nonexistent_directory(self, tmp_path: Path):
        sessions = PersistentMemory.list_sessions(memory_dir=tmp_path / "nope")
        assert sessions == []

    def test_lists_sessions(self, tmp_path: Path):
        for sid in ["session-a", "session-b"]:
            pm = PersistentMemory(
                memory_dir=tmp_path, session_id=sid, system_prompt="sys"
            )
            pm.add_message(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"hello from {sid}"}],
                }
            )

        sessions = PersistentMemory.list_sessions(memory_dir=tmp_path)
        assert len(sessions) == 2
        titles = {s["title"] for s in sessions}
        assert "hello from session-a" in titles
        assert "hello from session-b" in titles
        assert all(s["message_count"] >= 2 for s in sessions)

    def test_skips_corrupted_files(self, tmp_path: Path):
        (tmp_path / "good.json").write_text(
            json.dumps(
                {"messages": [], "extra": {"session_id": "good", "title": "Good"}}
            ),
            encoding="utf-8",
        )
        (tmp_path / "bad.json").write_text("not json", encoding="utf-8")
        sessions = PersistentMemory.list_sessions(memory_dir=tmp_path)
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "good"


class TestFlush:
    def test_flush_writes_to_disk(self, tmp_path: Path):
        pm = PersistentMemory(
            memory_dir=tmp_path, session_id="flush-test", system_prompt="sys"
        )
        pm.add_message({"role": "user", "content": [{"type": "text", "text": "hi"}]})
        session_file = tmp_path / "flush-test.json"
        assert session_file.exists()
        data = json.loads(session_file.read_text(encoding="utf-8"))
        assert len(data["messages"]) == 2

    def test_flush_after_clear(self, tmp_path: Path):
        pm = PersistentMemory(
            memory_dir=tmp_path, session_id="clear-test", system_prompt="sys"
        )
        pm.add_message({"role": "user", "content": [{"type": "text", "text": "hi"}]})
        pm.clear_messages()
        session_file = tmp_path / "clear-test.json"
        data = json.loads(session_file.read_text(encoding="utf-8"))
        assert all(m["role"] != "user" for m in data["messages"])
