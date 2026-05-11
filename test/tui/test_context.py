from __future__ import annotations

from unittest.mock import patch

import pytest

from minimal_harness.client.built_in.context import AppContext
from minimal_harness.client.built_in.memory_store import DiskMemoryStore
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
        assert ctx.all_tools == {}
        assert isinstance(ctx.memory_store, DiskMemoryStore)

    def test_with_provided_config(self):
        config = {"model": "custom-model", "provider": "openai"}
        ctx = AppContext(config=config)
        assert ctx.config["model"] == "custom-model"

    @pytest.mark.asyncio
    async def test_with_provided_registry(self):
        registry = ToolRegistry()
        await registry.register(
            StreamingTool(
                name="test_tool",
                description="Tool",
                parameters={"type": "object", "properties": {}},
                fn=lambda: (yield),
            )
        )
        ctx = AppContext(registry=registry)
        assert len(await ctx.registry.get_all()) == 1
        assert await ctx.registry.get("test_tool") is not None

    def test_memory_store_is_created(self):
        ctx = AppContext()
        assert isinstance(ctx.memory_store, DiskMemoryStore)

    def test_all_tools_property_default(self):
        ctx = AppContext()
        assert ctx.all_tools == {}


class TestAppContextRebuild:
    @pytest.mark.asyncio
    async def test_rebuild_populates_all_tools(self, sample_tool):
        ctx = AppContext()
        with patch("minimal_harness.client.built_in.context.collect_tools") as mock_ct:

            async def _collect(config, registry):
                await registry.register(sample_tool)

            mock_ct.side_effect = _collect
            await ctx.rebuild()
            mock_ct.assert_called_once()
        assert "sample_tool" in ctx.all_tools


class TestAppContextConfig:
    def test_update_config_saves(self):
        with patch("minimal_harness.client.built_in.context.save_config") as mock_save:
            with patch("minimal_harness.client.built_in.context.add_model") as mock_add:
                ctx = AppContext(config={"model": "old", "provider": "openai"})
                ctx.update_config({"model": "new-model", "theme": "nord"})
        assert ctx.config["model"] == "new-model"
        assert ctx.config["theme"] == "nord"
        mock_add.assert_called_once_with("new-model")
        mock_save.assert_called_once()

    def test_update_config_without_model(self):
        with patch("minimal_harness.client.built_in.context.save_config") as mock_save:
            with patch("minimal_harness.client.built_in.context.add_model") as mock_add:
                ctx = AppContext(config={"provider": "openai"})
                ctx.update_config({"theme": "nord"})
        mock_add.assert_not_called()
        mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_refresh_tools_clears_and_reloads(self):
        ctx = AppContext()
        with (
            patch("minimal_harness.client.built_in.context.collect_tools") as mock_ct,
        ):
            mock_ct.return_value = {}
            await ctx.refresh_tools()
        assert ctx.all_tools == {}


class TestCreateLLMProvider:
    @patch("minimal_harness.llm.factory.OpenAILLMProvider")
    @patch("minimal_harness.llm.factory.AsyncOpenAI")
    def test_creates_openai_provider(self, mock_async_openai, mock_provider):
        ctx = AppContext()
        cfg = {"provider": "openai", "model": "gpt-4", "base_url": "", "api_key": ""}
        result = ctx.create_llm_provider(cfg)
        assert result is mock_provider.return_value

    @patch("minimal_harness.llm.factory.AnthropicLLMProvider")
    @patch("minimal_harness.llm.factory.AsyncAnthropic")
    def test_creates_anthropic_provider(self, mock_async_anthropic, mock_provider):
        ctx = AppContext()
        cfg = {
            "provider": "anthropic",
            "model": "claude-3",
            "base_url": "",
            "api_key": "",
        }
        result = ctx.create_llm_provider(cfg)
        assert result is mock_provider.return_value
