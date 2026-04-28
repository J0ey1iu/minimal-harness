from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from minimal_harness.client.built_in import config as cfg_mod
from minimal_harness.client.built_in.config import (
    DEFAULT_CONFIG,
    add_model,
    collect_tools,
    ensure_system_prompts_dir,
    list_system_prompts,
    load_config,
    load_models,
    read_system_prompt,
    save_config,
    save_models,
)
from minimal_harness.tool.base import StreamingTool
from minimal_harness.tool.registry import ToolRegistry


class TestEnsureSystemPromptsDir:
    def test_creates_dir_and_default_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        sp_dir = tmp_path / "system-prompts"
        monkeypatch.setattr(cfg_mod, "SYSTEM_PROMPTS_DIR", sp_dir)
        ensure_system_prompts_dir()
        assert sp_dir.exists()
        default_file = sp_dir / "default.md"
        assert default_file.exists()
        assert (
            default_file.read_text(encoding="utf-8") == "You are a helpful assistant."
        )


class TestLoadModels:
    def test_no_file_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        models_file = tmp_path / "models.json"
        monkeypatch.setattr(cfg_mod, "MODELS_FILE", models_file)
        assert load_models() == []

    def test_loads_models_from_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        models_file = tmp_path / "models.json"
        models_file.write_text(json.dumps(["gpt-4", "claude-3"]), encoding="utf-8")
        monkeypatch.setattr(cfg_mod, "MODELS_FILE", models_file)
        assert load_models() == ["gpt-4", "claude-3"]

    def test_invalid_json_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        models_file = tmp_path / "models.json"
        models_file.write_text("not json", encoding="utf-8")
        monkeypatch.setattr(cfg_mod, "MODELS_FILE", models_file)
        assert load_models() == []

    def test_not_a_list_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        models_file = tmp_path / "models.json"
        models_file.write_text(json.dumps({"model": "gpt-4"}), encoding="utf-8")
        monkeypatch.setattr(cfg_mod, "MODELS_FILE", models_file)
        assert load_models() == []


class TestSaveModels:
    def test_saves_models(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        models_file = tmp_path / "models.json"
        monkeypatch.setattr(cfg_mod, "MODELS_FILE", models_file)
        save_models(["gpt-4", "claude-3"])
        assert models_file.exists()
        data = json.loads(models_file.read_text(encoding="utf-8"))
        assert data == ["gpt-4", "claude-3"]


class TestAddModel:
    def test_adds_new_model(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        models_file = tmp_path / "models.json"
        models_file.write_text(json.dumps(["claude-3"]), encoding="utf-8")
        monkeypatch.setattr(cfg_mod, "MODELS_FILE", models_file)
        add_model("gpt-4")
        data = json.loads(models_file.read_text(encoding="utf-8"))
        assert data == ["gpt-4", "claude-3"]

    def test_does_not_duplicate_existing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        models_file = tmp_path / "models.json"
        models_file.write_text(json.dumps(["gpt-4"]), encoding="utf-8")
        monkeypatch.setattr(cfg_mod, "MODELS_FILE", models_file)
        add_model("gpt-4")
        data = json.loads(models_file.read_text(encoding="utf-8"))
        assert data == ["gpt-4"]

    def test_empty_model_does_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        models_file = tmp_path / "models.json"
        monkeypatch.setattr(cfg_mod, "MODELS_FILE", models_file)
        add_model("")
        assert not models_file.exists()

    def test_inserts_at_front(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        models_file = tmp_path / "models.json"
        models_file.write_text(json.dumps(["a", "b"]), encoding="utf-8")
        monkeypatch.setattr(cfg_mod, "MODELS_FILE", models_file)
        add_model("c")
        data = json.loads(models_file.read_text(encoding="utf-8"))
        assert data == ["c", "a", "b"]


class TestLoadConfig:
    def test_returns_defaults_when_no_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        cfg_file = tmp_path / "config.json"
        sp_dir = tmp_path / "system-prompts"
        models_file = tmp_path / "models.json"
        monkeypatch.setattr(cfg_mod, "CONFIG_FILE", cfg_file)
        monkeypatch.setattr(cfg_mod, "SYSTEM_PROMPTS_DIR", sp_dir)
        monkeypatch.setattr(cfg_mod, "MODELS_FILE", models_file)

        result = load_config()
        for key in DEFAULT_CONFIG:
            assert key in result
        assert result["provider"] == "openai"
        assert sp_dir.exists()

    def test_merges_with_defaults_when_file_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        cfg_file = tmp_path / "config.json"
        sp_dir = tmp_path / "system-prompts"
        models_file = tmp_path / "models.json"
        monkeypatch.setattr(cfg_mod, "CONFIG_FILE", cfg_file)
        monkeypatch.setattr(cfg_mod, "SYSTEM_PROMPTS_DIR", sp_dir)
        monkeypatch.setattr(cfg_mod, "MODELS_FILE", models_file)

        cfg_file.write_text(
            json.dumps({"model": "custom-model", "theme": "nord"}), encoding="utf-8"
        )
        result = load_config()
        assert result["model"] == "custom-model"
        assert result["theme"] == "nord"
        assert cfg_file.exists()

    def test_invalid_json_returns_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        cfg_file = tmp_path / "config.json"
        sp_dir = tmp_path / "system-prompts"
        models_file = tmp_path / "models.json"
        monkeypatch.setattr(cfg_mod, "CONFIG_FILE", cfg_file)
        monkeypatch.setattr(cfg_mod, "SYSTEM_PROMPTS_DIR", sp_dir)
        monkeypatch.setattr(cfg_mod, "MODELS_FILE", models_file)

        cfg_file.write_text("not valid json", encoding="utf-8")
        result = load_config()
        for key in DEFAULT_CONFIG:
            assert key in result


class TestSaveConfig:
    def test_writes_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        cfg_file = tmp_path / "config.json"
        monkeypatch.setattr(cfg_mod, "CONFIG_FILE", cfg_file)
        save_config({"model": "test", "provider": "openai"})
        assert cfg_file.exists()
        data = json.loads(cfg_file.read_text(encoding="utf-8"))
        assert data["model"] == "test"
        assert data["provider"] == "openai"


class TestListSystemPrompts:
    def test_empty_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(cfg_mod, "SYSTEM_PROMPTS_DIR", tmp_path / "nonexistent")
        assert list_system_prompts() == []

    def test_lists_md_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        sp_dir = tmp_path / "system-prompts"
        sp_dir.mkdir(parents=True)
        (sp_dir / "a.md").write_text("a")
        (sp_dir / "b.md").write_text("b")
        (sp_dir / "c.txt").write_text("c")
        monkeypatch.setattr(cfg_mod, "SYSTEM_PROMPTS_DIR", sp_dir)
        result = list_system_prompts()
        assert len(result) == 2
        assert all(p.suffix == ".md" for p in result)


class TestReadSystemPrompt:
    def test_reads_existing_file(self, tmp_path: Path):
        f = tmp_path / "prompt.md"
        f.write_text("You are a bot.", encoding="utf-8")
        assert read_system_prompt(f) == "You are a bot."

    def test_nonexistent_file(self, tmp_path: Path):
        assert read_system_prompt(tmp_path / "nope.md") == ""


class TestCollectTools:
    def test_merges_built_in_and_external(self):
        registry = ToolRegistry()
        config = {"tools_path": ""}

        with (
            patch("minimal_harness.client.built_in.config.get_bash_tools") as mock_bash,
            patch(
                "minimal_harness.client.built_in.config.get_local_file_operation_tools"
            ) as mock_lfo,
        ):
            mock_bash.return_value = {
                "bash": StreamingTool(
                    name="bash",
                    description="Run bash",
                    parameters={"type": "object", "properties": {}},
                    fn=lambda: (yield),
                )
            }
            mock_lfo.return_value = {
                "read_file": StreamingTool(
                    name="read_file",
                    description="Read file",
                    parameters={"type": "object", "properties": {}},
                    fn=lambda: (yield),
                )
            }
            tools = collect_tools(config, registry)
        assert "bash" in tools
        assert "read_file" in tools
        assert len(tools) == 2

    def test_loads_external_tools(self):
        registry = ToolRegistry()
        ext_tool = StreamingTool(
            name="ext_tool",
            description="External",
            parameters={"type": "object", "properties": {}},
            fn=lambda: (yield),
        )
        registry.register(ext_tool)
        config = {"tools_path": "/some/path"}

        with (
            patch("minimal_harness.client.built_in.config.get_bash_tools") as mock_bash,
            patch(
                "minimal_harness.client.built_in.config.get_local_file_operation_tools"
            ) as mock_lfo,
            patch(
                "minimal_harness.client.built_in.config.load_external_tools"
            ) as mock_load,
        ):
            mock_bash.return_value = {}
            mock_lfo.return_value = {}
            mock_load.return_value = None
            tools = collect_tools(config, registry)

        assert "ext_tool" in tools
        mock_load.assert_called_once_with("/some/path", registry)

    def test_warns_on_name_collision(self):
        registry = ToolRegistry()
        ext_tool = StreamingTool(
            name="bash",
            description="External bash",
            parameters={"type": "object", "properties": {}},
            fn=lambda: (yield),
        )
        registry.register(ext_tool)
        config = {"tools_path": "/path"}

        with (
            patch("minimal_harness.client.built_in.config.get_bash_tools") as mock_bash,
            patch(
                "minimal_harness.client.built_in.config.get_local_file_operation_tools"
            ) as mock_lfo,
            patch(
                "minimal_harness.client.built_in.config.load_external_tools"
            ) as mock_load,
            patch("warnings.warn") as mock_warn,
        ):
            mock_bash.return_value = {
                "bash": StreamingTool(
                    name="bash",
                    description="Built-in bash",
                    parameters={"type": "object", "properties": {}},
                    fn=lambda: (yield),
                )
            }
            mock_lfo.return_value = {}
            mock_load.return_value = None
            tools = collect_tools(config, registry)

        assert "bash" in tools
        mock_warn.assert_called_once()
        assert "External tool 'bash' overwrites built-in" in mock_warn.call_args[0][0]
