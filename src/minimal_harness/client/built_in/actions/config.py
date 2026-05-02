"""Configuration action — handles Ctrl+O."""

from __future__ import annotations

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
        app.ctx.refresh_tools()
        if (t := result.get("theme")) in THEMES:
            app.theme = t
            d.theme = t
        app._ctrl.rebuild_current_session(
            llm_provider=app.ctx.create_llm_provider(),
        )
        d.say("\u2713 Configuration saved", "bold bright_green")
        app._banner(show=app._first)

    app.push_screen(ConfigScreen(app.ctx.config), done)
