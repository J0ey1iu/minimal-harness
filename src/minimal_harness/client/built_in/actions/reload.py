"""Reload action — re-reads agents and tools from disk."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from minimal_harness.tool.built_in.runtime_tools import register_runtime_tools

if TYPE_CHECKING:
    from minimal_harness.client.built_in.app import TUIApp


def action_reload(app: TUIApp) -> None:
    async def _reload() -> None:
        if app._ctrl.is_any_session_running():
            app.notify(
                "Cannot reload while a session is running",
                severity="warning",
                timeout=5,
            )
            return

        await app._agent_registry.clear()
        await app.ctx.rebuild()
        await register_runtime_tools(
            agent_registry=app._agent_registry,
            session_store=app.ctx.session_store,
            tool_registry=app.ctx.registry,
            run_fn=app._runtime.run,
        )
        await app._ctrl.register_preset_agents()

        agent_count = len(await app._agent_registry.get_all())
        tool_count = len(await app.ctx.registry.get_all())
        msg = f"\u21bb Reloaded: {agent_count} agents, {tool_count} tools"
        app.notify(msg, severity="information", timeout=5)
        if app._chat_display is not None:
            app._chat_display.say(f"  {msg}", "dim")

    asyncio.create_task(_reload())
