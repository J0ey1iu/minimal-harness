"""Configuration action — handles Ctrl+O."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from minimal_harness.client.built_in.constants import THEMES
from minimal_harness.client.built_in.modals import ConfigScreen

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

        async def _post_config() -> None:
            await app.ctx.refresh_tools()
            await app._runtime.register_runtime_tools()
            await app._banner(show=app._first)

        asyncio.create_task(_post_config())

    app.push_screen(ConfigScreen(app.ctx.config), done)
