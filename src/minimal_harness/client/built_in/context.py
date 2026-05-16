"""Application context that owns configuration, registry, session store, and LLM provider."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from minimal_harness.tool.base import Tool

from minimal_harness.client.built_in.config import (
    add_model,
    collect_tools,
    load_config,
    save_config,
)
from minimal_harness.client.built_in.memory_store import DiskSessionStore
from minimal_harness.llm import LLMProvider, create_llm_provider
from minimal_harness.tool.registry import ToolRegistry


class TUIConfig:
    """Configuration loading, saving, and model management."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = config or load_config()

    def update_config(self, result: dict[str, Any]) -> None:
        self.config.update(result)
        if "model" in result:
            add_model(result["model"])
        save_config(self.config)


class AppContext:
    """Application context — facade over TUIConfig, ToolRegistry, and DiskSessionStore."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        self._config_manager = TUIConfig(config=config)
        self._registry: ToolRegistry = registry or ToolRegistry()
        self._session_store = DiskSessionStore()

    @property
    def config(self) -> dict[str, Any]:
        return self._config_manager.config

    @config.setter
    def config(self, value: dict[str, Any]) -> None:
        self._config_manager.config = value

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    @property
    def all_tools(self) -> dict[str, "Tool"]:
        return {t.name: t for t in self._registry._data.values()}

    @property
    def session_store(self) -> DiskSessionStore:
        return self._session_store

    async def rebuild(self) -> None:
        await self._registry.clear()
        await collect_tools(self.config, self._registry)

    async def refresh_tools(self) -> None:
        await self.rebuild()

    def update_config(self, result: dict[str, Any]) -> None:
        self._config_manager.update_config(result)

    def create_llm_provider(self, cfg: dict[str, Any] | None = None) -> LLMProvider:
        effective = cfg if cfg is not None else self.config
        return create_llm_provider(effective)
