"""Tool selection action — handles Ctrl+T."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from minimal_harness.client.built_in.modals import ToolSelectScreen

if TYPE_CHECKING:
    from minimal_harness.client.built_in.app import TUIApp


def action_tools(app: TUIApp) -> None:
    if not app._all_tools:
        return
    sess = app._ctrl.current_session
    selected = set(sess.tool_names) if sess else set()

    def done(chosen: list[str] | None) -> None:
        if chosen is None:
            return
        d = app._chat_display
        if d is None:
            return
        resolved = [app.ctx.all_tools[n] for n in chosen if n in app.ctx.all_tools]
        sess = app._ctrl.current_session
        if sess:
            app._ctrl.rebuild_current_session(tools=resolved)
            sess.tool_names = chosen

        def _display_name(t):
            resolve = getattr(t, "resolve_display_name", None)
            return (
                resolve() if resolve else (getattr(t, "display_name", None) or t.name)
            )

        names = ", ".join(_display_name(t) for t in resolved) or "(none)"
        d.say(f"\u2713 Tools: {names}", "bold bright_green")
        if app._first:
            asyncio.create_task(app._banner())

    app.push_screen(ToolSelectScreen(app._all_tools, selected), done)
