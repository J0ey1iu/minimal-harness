from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from minimal_harness.client.built_in.context import AppContext
from minimal_harness.tool.base import StreamingTool
from minimal_harness.tool.registry import ToolRegistry


@pytest.fixture
def sample_tool():
    return StreamingTool(
        name="sample_tool",
        description="A sample tool",
        parameters={"type": "object", "properties": {}},
        fn=lambda: (yield),
    )


class TestAppContextInit:
    def test_default_init(self):
        ctx = AppContext()
        assert isinstance(ctx.registry, ToolRegistry)
        assert ctx._all_tools == {}
        assert ctx.active_tools == []
        assert ctx.memory is None
        assert ctx.agent is None

    def test_with_provided_config(self):
        config = {"model": "custom-model", "provider": "openai"}
        ctx = AppContext(config=config)
        assert ctx.config["model"] == "custom-model"

    def test_with_provided_registry(self):
        registry = ToolRegistry()
        registry.register(
            StreamingTool(
                name="test_tool",
                description="Tool",
                parameters={"type": "object", "properties": {}},
                fn=lambda: (yield),
            )
        )
        ctx = AppContext(registry=registry)
        assert len(ctx.registry.get_all()) == 1
        assert ctx.registry.get("test_tool") is not None

    def test_with_agent_factory(self):
        factory = MagicMock(return_value="fake_agent")
        ctx = AppContext(agent_factory=factory)
        assert ctx._agent_factory is factory

    def test_all_tools_property(self):
        ctx = AppContext()
        assert ctx.all_tools == {}


class TestAppContextCreateLLMProvider:
    def test_uses_factory_when_provided(self):
        factory = MagicMock(return_value="custom_provider")
        ctx = AppContext(llm_provider_factory=factory)
        result = ctx._create_llm_provider({"provider": "openai"})
        assert result == "custom_provider"
        factory.assert_called_once_with({"provider": "openai"})

    @patch("minimal_harness.client.built_in.context.OpenAILLMProvider")
    @patch("minimal_harness.client.built_in.context.AsyncOpenAI")
    def test_creates_openai_provider(self, mock_async_openai, mock_provider):
        ctx = AppContext()
        cfg = {"provider": "openai", "model": "gpt-4", "base_url": "", "api_key": ""}
        result = ctx._create_llm_provider(cfg)
        assert result is mock_provider.return_value
        mock_provider.assert_called_once()

    @patch("minimal_harness.client.built_in.context.AnthropicLLMProvider")
    @patch("minimal_harness.client.built_in.context.AsyncAnthropic")
    def test_creates_anthropic_provider(self, mock_async_anthropic, mock_provider):
        ctx = AppContext()
        cfg = {
            "provider": "anthropic",
            "model": "claude-3",
            "base_url": "",
            "api_key": "",
        }
        result = ctx._create_llm_provider(cfg)
        assert result is mock_provider.return_value


class TestAppContextRebuild:
    def test_rebuild_uses_all_tools_by_default(self, tmp_path: Path, sample_tool):
        registry = ToolRegistry()

        with (
            patch("minimal_harness.client.built_in.context.collect_tools") as mock_ct,
            patch.object(AppContext, "_create_llm_provider") as mock_clp,
        ):
            mock_ct.return_value = {"sample_tool": sample_tool}
            mock_clp.return_value = MagicMock()

            ctx = AppContext(
                config={"selected_tools": []},
                registry=registry,
            )
            ctx.rebuild()

        assert "sample_tool" in ctx._all_tools
        assert ctx.active_tools == [sample_tool]
        assert ctx.memory is not None
        assert ctx.agent is not None

    def test_rebuild_filters_selected_tools(self, sample_tool):
        tool_b = StreamingTool(
            name="tool_b",
            description="B",
            parameters={"type": "object", "properties": {}},
            fn=lambda: (yield),
        )

        with (
            patch("minimal_harness.client.built_in.context.collect_tools") as mock_ct,
            patch.object(AppContext, "_create_llm_provider") as mock_clp,
        ):
            mock_ct.return_value = {"sample_tool": sample_tool, "tool_b": tool_b}
            mock_clp.return_value = MagicMock()

            ctx = AppContext(config={"selected_tools": ["sample_tool"]})
            ctx.rebuild()

        assert ctx.active_tools == [sample_tool]

    def test_rebuild_updates_existing_memory_system_prompt(self, sample_tool):
        with (
            patch("minimal_harness.client.built_in.context.collect_tools") as mock_ct,
            patch.object(AppContext, "_create_llm_provider") as mock_clp,
        ):
            mock_ct.return_value = {"sample_tool": sample_tool}
            mock_clp.return_value = MagicMock()

            ctx = AppContext(config={"selected_tools": []})
            ctx.rebuild()
            initial_memory = ctx.memory

            ctx.rebuild()
            assert ctx.memory is initial_memory


class TestAppContextUpdateConfig:
    def test_updates_config_and_saves(self):
        with patch("minimal_harness.client.built_in.context.save_config") as mock_save:
            with patch("minimal_harness.client.built_in.context.add_model") as mock_add:
                ctx = AppContext(config={"model": "old", "provider": "openai"})
                ctx.update_config({"model": "new-model", "theme": "nord"})

        assert ctx.config["model"] == "new-model"
        assert ctx.config["theme"] == "nord"
        mock_add.assert_called_once_with("new-model")
        mock_save.assert_called_once()

    def test_update_does_not_add_model_if_not_in_result(self):
        with patch("minimal_harness.client.built_in.context.save_config") as mock_save:
            with patch("minimal_harness.client.built_in.context.add_model") as mock_add:
                ctx = AppContext(config={"provider": "openai"})
                ctx.update_config({"theme": "nord"})

        mock_add.assert_not_called()
        mock_save.assert_called_once()


class TestAppContextSelectTools:
    def test_select_tools_filters(self, sample_tool):
        ctx = AppContext()
        ctx._all_tools = {"sample_tool": sample_tool, "other": sample_tool}

        with patch("minimal_harness.client.built_in.context.save_config") as mock_save:
            ctx.select_tools(["sample_tool"])

        assert ctx.active_tools == [sample_tool]
        assert ctx.config["selected_tools"] == ["sample_tool"]
        mock_save.assert_called_once()

    def test_select_tools_skips_unknown(self):
        ctx = AppContext()
        ctx._all_tools = {}

        with patch("minimal_harness.client.built_in.context.save_config"):
            ctx.select_tools(["nonexistent"])

        assert ctx.active_tools == []


class TestAppContextResetMemory:
    def test_reset_creates_new_memory(self):
        ctx = AppContext()
        ctx.memory = MagicMock()
        old_memory = ctx.memory
        ctx.reset_memory()
        assert ctx.memory is not old_memory
        assert ctx.memory is not None


class TestCreateSimpleAgent:
    def test_creates_simple_agent(self):
        from minimal_harness.agent import SimpleAgent
        from minimal_harness.client.built_in.context import _create_simple_agent

        provider = MagicMock()
        tools = []
        memory = MagicMock()
        agent = _create_simple_agent(provider, tools, memory)
        assert isinstance(agent, SimpleAgent)
