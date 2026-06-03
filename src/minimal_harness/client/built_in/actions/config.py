"""Configuration action — handles Ctrl+O."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from minimal_harness.client.built_in.constants import THEMES
from minimal_harness.client.built_in.modals import ConfigScreen
from minimal_harness.tool.built_in.runtime_tools import register_runtime_tools

if TYPE_CHECKING:
    from minimal_harness.client.built_in.app import TUIApp


def action_config(app: TUIApp) -> None:
    def done(result: dict | None) -> None:
        if result is None:
            return
        d = app._chat_display
        if d is None:
            return
        app.ctx.update_config(result)
        if (t := result.get("theme")) in THEMES:
            app.theme = t
            d.theme = t
        app._ctrl.rebuild_current_session()
        d.say("\u2713 Configuration saved", "bold bright_green")

        provider_cfg: dict[str, Any] = {}
        if app.ctx.config.get("api_key"):
            provider_cfg["api_key"] = app.ctx.config["api_key"]
        if app.ctx.config.get("base_url"):
            provider_cfg["base_url"] = app.ctx.config["base_url"]
        if app.ctx.config.get("model"):
            provider_cfg["model"] = app.ctx.config["model"]
        if provider_cfg:
            app._llm_registry.set_default_config("openai", provider_cfg)

        async def _post_config() -> None:
            await app.ctx.refresh_tools()
            await register_runtime_tools(
                agent_registry=app._agent_registry,
                session_store=app.ctx.session_store,
                tool_registry=app.ctx.registry,
                run_fn=app._runtime.run,
            )
            await app._banner(show=app._first)

        asyncio.create_task(_post_config())

    app.push_screen(ConfigScreen(app.ctx.config), done)
