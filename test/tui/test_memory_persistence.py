from __future__ import annotations

import json

import pytest

from minimal_harness.client.built_in.memory import PersistentMemory
from minimal_harness.types import TokenUsage


class TestPersistentMemoryInit:
    def test_creates_memory_dir(self, tmp_path):
        mem_dir = tmp_path / "memories"
        pm = PersistentMemory(memory_dir=mem_dir, session_id="test-session")
        assert mem_dir.exists()
        assert pm.session_id == "test-session"
        assert pm.title is None

    def test_generates_session_id_when_not_provided(self, tmp_path):
        pm = PersistentMemory(memory_dir=tmp_path)
        assert pm.session_id is not None
        assert len(pm.session_id) > 0

    def test_system_prompt_stored(self, tmp_path):
        pm = PersistentMemory(memory_dir=tmp_path, system_prompt="You are a bot.")
        msgs = pm.get_all_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are a bot."

    def test_created_at_set_on_construction(self, tmp_path):
        pm = PersistentMemory(memory_dir=tmp_path)
        assert pm.created_at is not None

    def test_consecutive_instances_different_created_at(self, tmp_path):
        pm1 = PersistentMemory(memory_dir=tmp_path)
        pm2 = PersistentMemory(memory_dir=tmp_path)
        assert pm1.created_at != pm2.created_at

    def test_default_agent_name(self, tmp_path):
        pm = PersistentMemory(memory_dir=tmp_path)
        assert pm.agent_name == ""

    def test_custom_agent_name(self, tmp_path):
        pm = PersistentMemory(memory_dir=tmp_path, agent_name="my_agent")
        assert pm.agent_name == "my_agent"

    def test_agent_name_setter(self, tmp_path):
        pm = PersistentMemory(memory_dir=tmp_path)
        pm.agent_name = "renamed"
        assert pm.agent_name == "renamed"

    def test_selected_tools_default(self, tmp_path):
        pm = PersistentMemory(memory_dir=tmp_path)
        assert pm.selected_tools == []

    def test_selected_tools_custom(self, tmp_path):
        pm = PersistentMemory(memory_dir=tmp_path, selected_tools=["bash", "read"])
        assert pm.selected_tools == ["bash", "read"]


class TestPersistentMemoryMessages:
    def test_add_user_message_sets_title(self, tmp_path):
        pm = PersistentMemory(memory_dir=tmp_path, system_prompt="system")
        pm.add_message(
            {"role": "user", "content": [{"type": "text", "text": "Hello world"}]}
        )
        assert pm.title == "Hello world"

    def test_title_truncated_to_100_chars(self, tmp_path):
        pm = PersistentMemory(memory_dir=tmp_path, system_prompt="system")
        long_text = "a" * 200
        pm.add_message(
            {"role": "user", "content": [{"type": "text", "text": long_text}]}
        )
        assert pm.title is not None
        assert len(pm.title) == 100

    def test_first_user_message_only_sets_title_once(self, tmp_path):
        pm = PersistentMemory(memory_dir=tmp_path, system_prompt="system")
        pm.add_message({"role": "user", "content": [{"type": "text", "text": "first"}]})
        assert pm.title == "first"
        pm.add_message(
            {"role": "user", "content": [{"type": "text", "text": "second"}]}
        )
        assert pm.title == "first"

    def test_title_not_set_by_assistant_message(self, tmp_path):
        pm = PersistentMemory(memory_dir=tmp_path, system_prompt="system")
        pm.add_message({"role": "assistant", "content": "I am an AI."})
        assert pm.title is None

    def test_get_all_messages_returns_system_and_messages(self, tmp_path):
        pm = PersistentMemory(memory_dir=tmp_path, system_prompt="You are helpful.")
        pm.add_message({"role": "user", "content": [{"type": "text", "text": "hi"}]})
        msgs = pm.get_all_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are helpful."

    def test_clear_messages_keeps_system(self, tmp_path):
        pm = PersistentMemory(memory_dir=tmp_path, system_prompt="system")
        pm.add_message({"role": "user", "content": [{"type": "text", "text": "hi"}]})
        pm.clear_messages()
        msgs = pm.get_all_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"

    def test_get_forward_messages_excludes_reasoning(self, tmp_path):
        pm = PersistentMemory(memory_dir=tmp_path, system_prompt="system")
        pm.add_message({"role": "user", "content": [{"type": "text", "text": "hi"}]})
        pm.add_message({"role": "reasoning", "content": "thinking..."})
        fwd = pm.get_forward_messages()
        roles = [m["role"] for m in fwd]
        assert "reasoning" not in roles

    def test_message_usage(self, tmp_path):
        pm = PersistentMemory(memory_dir=tmp_path, system_prompt="system")
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        pm.set_message_usage(usage)
        assert pm.get_message_usage() == usage

    def test_update_system_prompt(self, tmp_path):
        pm = PersistentMemory(memory_dir=tmp_path, system_prompt="Old prompt")
        pm.update_system_prompt("New prompt")
        msgs = pm.get_all_messages()
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "New prompt"


class TestPersistentMemoryDumpLoad:
    def test_dump_includes_metadata(self, tmp_path):
        pm = PersistentMemory(
            memory_dir=tmp_path, session_id="sid-1", system_prompt="system"
        )
        pm.add_message({"role": "user", "content": [{"type": "text", "text": "hi"}]})
        data = pm.dump_memory()
        assert data["extra"]["session_id"] == "sid-1"
        assert "title" in data["extra"]
        assert "created_at" in data["extra"]
        assert "agent_name" in data["extra"]
        assert "selected_tools" in data["extra"]

    def test_dump_memory_json_round_trip(self, tmp_path):
        pm = PersistentMemory(
            memory_dir=tmp_path, session_id="sid-1", system_prompt="system"
        )
        s = pm.dump_memory_json()
        data = json.loads(s)
        assert data["extra"]["session_id"] == "sid-1"

    def test_load_memory_restores_state(self, tmp_path):
        pm = PersistentMemory(
            memory_dir=tmp_path, session_id="sid-1", system_prompt="system"
        )
        pm.add_message({"role": "user", "content": [{"type": "text", "text": "hello"}]})
        data = pm.dump_memory()

        pm2 = PersistentMemory(
            memory_dir=tmp_path, session_id="sid-2", system_prompt="system"
        )
        pm2.load_memory(data)
        assert pm2.session_id == "sid-1"
        assert len(pm2.get_all_messages()) == 2

    def test_load_memory_restores_metadata(self, tmp_path):
        pm = PersistentMemory(
            memory_dir=tmp_path,
            session_id="sid-1",
            system_prompt="system",
            agent_name="test_agent",
            selected_tools=["bash"],
        )
        pm.add_message({"role": "user", "content": [{"type": "text", "text": "hello"}]})
        data = pm.dump_memory()

        pm2 = PersistentMemory(memory_dir=tmp_path)
        pm2.load_memory(data)
        assert pm2.agent_name == "test_agent"
        assert pm2.selected_tools == ["bash"]

    def test_load_memory_json_round_trip(self, tmp_path):
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


class TestPersistentMemoryFlush:
    def test_flush_writes_to_disk_on_add_message(self, tmp_path):
        pm = PersistentMemory(
            memory_dir=tmp_path, session_id="flush-test", system_prompt="sys"
        )
        pm.add_message({"role": "user", "content": [{"type": "text", "text": "hi"}]})
        session_file = tmp_path / "flush-test.json"
        assert session_file.exists()
        data = json.loads(session_file.read_text(encoding="utf-8"))
        assert len(data["messages"]) == 2

    def test_flush_after_clear(self, tmp_path):
        pm = PersistentMemory(
            memory_dir=tmp_path, session_id="clear-test", system_prompt="sys"
        )
        pm.add_message({"role": "user", "content": [{"type": "text", "text": "hi"}]})
        pm.clear_messages()
        session_file = tmp_path / "clear-test.json"
        data = json.loads(session_file.read_text(encoding="utf-8"))
        assert len(data["messages"]) == 1
        assert data["messages"][0]["role"] == "system"

    def test_flush_updates_system_prompt(self, tmp_path):
        pm = PersistentMemory(
            memory_dir=tmp_path, session_id="sys-flush", system_prompt="Old"
        )
        pm.update_system_prompt("New")
        session_file = tmp_path / "sys-flush.json"
        data = json.loads(session_file.read_text(encoding="utf-8"))
        assert data["messages"][0]["content"] == "New"

    def test_flush_writes_usage(self, tmp_path):
        pm = PersistentMemory(
            memory_dir=tmp_path, session_id="usage-flush", system_prompt="sys"
        )
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        pm.set_message_usage(usage)
        session_file = tmp_path / "usage-flush.json"
        data = json.loads(session_file.read_text(encoding="utf-8"))
        assert data["usage"]["total_tokens"] == 15

    def test_explicit_flush(self, tmp_path):
        pm = PersistentMemory(
            memory_dir=tmp_path, session_id="explicit-flush", system_prompt="sys"
        )
        pm.flush()
        session_file = tmp_path / "explicit-flush.json"
        assert session_file.exists()


class TestPersistentMemoryFromSession:
    def test_loads_session_successfully(self, tmp_path):
        hex_id = "a" * 32
        pm = PersistentMemory(
            memory_dir=tmp_path, session_id=hex_id, system_prompt="system"
        )
        pm.add_message({"role": "user", "content": [{"type": "text", "text": "hello"}]})

        loaded = PersistentMemory.from_session(hex_id, memory_dir=tmp_path)
        assert loaded.session_id == hex_id
        assert len(loaded.get_all_messages()) == 2

    def test_session_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PersistentMemory.from_session("a" * 32, memory_dir=tmp_path)

    def test_load_restores_title_and_created_at(self, tmp_path):
        sid = "b" * 32
        pm = PersistentMemory(
            memory_dir=tmp_path, session_id=sid, system_prompt="system"
        )
        pm.add_message(
            {"role": "user", "content": [{"type": "text", "text": "my title"}]}
        )
        original_created = pm.created_at

        loaded = PersistentMemory.from_session(sid, memory_dir=tmp_path)
        assert loaded.title == "my title"
        assert loaded.created_at == original_created


class TestPersistentMemoryListSessions:
    def test_empty_directory(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir(parents=True)
        sessions = PersistentMemory.list_sessions(memory_dir=empty_dir)
        assert sessions == []

    def test_nonexistent_directory_returns_empty(self, tmp_path):
        sessions = PersistentMemory.list_sessions(memory_dir=tmp_path / "nope")
        assert sessions == []

    def test_lists_multiple_sessions(self, tmp_path):
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

    def test_skips_corrupted_files(self, tmp_path):
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

    def test_sessions_sorted_by_created_at_desc(self, tmp_path):
        pm_old = PersistentMemory(
            memory_dir=tmp_path, session_id="o" * 32, system_prompt="sys"
        )
        pm_old.add_message(
            {"role": "user", "content": [{"type": "text", "text": "old"}]}
        )

        pm_new = PersistentMemory(
            memory_dir=tmp_path, session_id="n" * 32, system_prompt="sys"
        )
        pm_new.add_message(
            {"role": "user", "content": [{"type": "text", "text": "new"}]}
        )

        sessions = PersistentMemory.list_sessions(memory_dir=tmp_path)
        timestamps = [s["created_at"] for s in sessions]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_message_count_in_listing(self, tmp_path):
        pm = PersistentMemory(
            memory_dir=tmp_path, session_id="c" * 32, system_prompt="sys"
        )
        pm.add_message({"role": "user", "content": [{"type": "text", "text": "msg1"}]})
        pm.add_message({"role": "assistant", "content": "resp1"})

        sessions = PersistentMemory.list_sessions(memory_dir=tmp_path)
        assert sessions[0]["message_count"] >= 3

    def test_agent_name_in_listing(self, tmp_path):
        pm = PersistentMemory(
            memory_dir=tmp_path,
            session_id="d" * 32,
            system_prompt="sys",
            agent_name="code_bot",
        )
        pm.add_message({"role": "user", "content": [{"type": "text", "text": "hi"}]})

        sessions = PersistentMemory.list_sessions(memory_dir=tmp_path)
        assert sessions[0]["agent_name"] == "code_bot"
