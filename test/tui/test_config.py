from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
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


def _patch_config_dir(monkeypatch, base: Path) -> None:
    """Point get_config_dir() to *base* so all config derives from it."""
    monkeypatch.setattr(
        "minimal_harness.client.built_in.config.paths.get_config_dir",
        lambda: base,
    )


class TestEnsureSystemPromptsDir:
    def test_creates_dir(self, tmp_path, monkeypatch):
        _patch_config_dir(monkeypatch, tmp_path)
        ensure_system_prompts_dir()
        assert (tmp_path / "system-prompts").exists()


class TestLoadModels:
    def test_no_file_returns_empty(self, tmp_path, monkeypatch):
        _patch_config_dir(monkeypatch, tmp_path)
        assert load_models() == []

    def test_loads_from_file(self, tmp_path, monkeypatch):
        _patch_config_dir(monkeypatch, tmp_path)
        models_file = tmp_path / "models.json"
        models_file.write_text(json.dumps(["gpt-4", "claude-3"]), encoding="utf-8")
        assert load_models() == ["gpt-4", "claude-3"]

    def test_invalid_json_returns_empty(self, tmp_path, monkeypatch):
        _patch_config_dir(monkeypatch, tmp_path)
        models_file = tmp_path / "models.json"
        models_file.write_text("not json", encoding="utf-8")
        assert load_models() == []

    def test_not_a_list_returns_empty(self, tmp_path, monkeypatch):
        _patch_config_dir(monkeypatch, tmp_path)
        models_file = tmp_path / "models.json"
        models_file.write_text(json.dumps({"model": "gpt-4"}), encoding="utf-8")
        assert load_models() == []


class TestSaveModels:
    def test_saves(self, tmp_path, monkeypatch):
        _patch_config_dir(monkeypatch, tmp_path)
        save_models(["gpt-4", "claude-3"])
        data = json.loads((tmp_path / "models.json").read_text(encoding="utf-8"))
        assert data == ["gpt-4", "claude-3"]


class TestAddModel:
    def test_adds_new_model(self, tmp_path, monkeypatch):
        _patch_config_dir(monkeypatch, tmp_path)
        models_file = tmp_path / "models.json"
        models_file.write_text(json.dumps(["claude-3"]), encoding="utf-8")
        add_model("gpt-4")
        data = json.loads(models_file.read_text(encoding="utf-8"))
        assert data == ["gpt-4", "claude-3"]

    def test_no_duplicate(self, tmp_path, monkeypatch):
        _patch_config_dir(monkeypatch, tmp_path)
        models_file = tmp_path / "models.json"
        models_file.write_text(json.dumps(["gpt-4"]), encoding="utf-8")
        add_model("gpt-4")
        data = json.loads(models_file.read_text(encoding="utf-8"))
        assert data == ["gpt-4"]

    def test_empty_model_does_nothing(self, tmp_path, monkeypatch):
        _patch_config_dir(monkeypatch, tmp_path)
        add_model("")
        assert not (tmp_path / "models.json").exists()

    def test_inserts_at_front(self, tmp_path, monkeypatch):
        _patch_config_dir(monkeypatch, tmp_path)
        models_file = tmp_path / "models.json"
        models_file.write_text(json.dumps(["a", "b"]), encoding="utf-8")
        add_model("c")
        data = json.loads(models_file.read_text(encoding="utf-8"))
        assert data == ["c", "a", "b"]


class TestLoadConfig:
    def test_defaults_when_no_file(self, tmp_path, monkeypatch):
        _patch_config_dir(monkeypatch, tmp_path)

        result = load_config()
        for key in DEFAULT_CONFIG:
            assert key in result
        assert (tmp_path / "system-prompts").exists()

    def test_merges_with_file(self, tmp_path, monkeypatch):
        _patch_config_dir(monkeypatch, tmp_path)

        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(
            json.dumps({"model": "custom-model", "theme": "nord"}), encoding="utf-8"
        )
        result = load_config()
        assert result["model"] == "custom-model"
        assert result["theme"] == "nord"

    def test_invalid_json_returns_defaults(self, tmp_path, monkeypatch):
        _patch_config_dir(monkeypatch, tmp_path)

        cfg_file = tmp_path / "config.json"
        cfg_file.write_text("not valid json", encoding="utf-8")
        result = load_config()
        for key in DEFAULT_CONFIG:
            assert key in result


class TestSaveConfig:
    def test_writes(self, tmp_path, monkeypatch):
        _patch_config_dir(monkeypatch, tmp_path)
        save_config({"model": "test", "provider": "openai"})
        data = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
        assert data["model"] == "test"


class TestListSystemPrompts:
    def test_empty_dir(self, tmp_path, monkeypatch):
        _patch_config_dir(monkeypatch, tmp_path)
        assert list_system_prompts() == []

    def test_lists_only_md_files(self, tmp_path, monkeypatch):
        _patch_config_dir(monkeypatch, tmp_path)
        sp_dir = tmp_path / "system-prompts"
        sp_dir.mkdir(parents=True)
        (sp_dir / "a.md").write_text("a")
        (sp_dir / "b.md").write_text("b")
        (sp_dir / "c.txt").write_text("c")
        result = list_system_prompts()
        assert len(result) == 2
        assert all(p.suffix == ".md" for p in result)


class TestReadSystemPrompt:
    def test_reads_existing(self, tmp_path):
        f = tmp_path / "prompt.md"
        f.write_text("You are a bot.", encoding="utf-8")
        assert read_system_prompt(f) == "You are a bot."

    def test_nonexistent_returns_empty(self, tmp_path):
        assert read_system_prompt(tmp_path / "nope.md") == ""


class TestCollectTools:
    @pytest.mark.asyncio
    async def test_merges_built_in(self):
        registry = ToolRegistry()
        config = {"tools_path": ""}

        with (
            patch("minimal_harness.tool.built_in.bash.get_tools") as mock_bash,
            patch(
                "minimal_harness.tool.built_in.local_file_operation.get_tools"
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
            await collect_tools(config, registry)

        assert (await registry.get("bash")) is not None
        assert (await registry.get("read_file")) is not None
        assert len(await registry.get_all()) == 2

    @pytest.mark.asyncio
    async def test_loads_external_tools(self):
        registry = ToolRegistry()
        from minimal_harness.types import LocalToolBinding, ToolMetadata

        ext_tool_meta = ToolMetadata(
            name="ext_tool",
            description="External",
            parameters={"type": "object", "properties": {}},
            binding=LocalToolBinding(fn=lambda: (yield)),
        )
        await registry.register(ext_tool_meta)
        config = {"tools_path": "/some/path"}

        with (
            patch("minimal_harness.tool.built_in.bash.get_tools") as mock_bash,
            patch(
                "minimal_harness.tool.built_in.local_file_operation.get_tools"
            ) as mock_lfo,
            patch("minimal_harness.tool.collector.load_external_tools") as mock_load,
        ):
            mock_bash.return_value = {}
            mock_lfo.return_value = {}
            mock_load.return_value = None
            await collect_tools(config, registry)

        assert (await registry.get("ext_tool")) is not None

    @pytest.mark.asyncio
    async def test_warns_on_name_collision(self):
        registry = ToolRegistry()
        from minimal_harness.types import LocalToolBinding, ToolMetadata

        ext_tool_meta = ToolMetadata(
            name="bash",
            description="External bash",
            parameters={"type": "object", "properties": {}},
            binding=LocalToolBinding(fn=lambda: (yield)),
        )
        await registry.register(ext_tool_meta)
        config = {"tools_path": "/path"}

        with (
            patch("minimal_harness.tool.built_in.bash.get_tools") as mock_bash,
            patch(
                "minimal_harness.tool.built_in.local_file_operation.get_tools"
            ) as mock_lfo,
            patch("minimal_harness.tool.collector.load_external_tools") as mock_load,
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
            await collect_tools(config, registry)

        assert (await registry.get("bash")) is not None
        mock_warn.assert_called_once()
        assert "External tool 'bash' overwrites built-in" in mock_warn.call_args[0][0]
