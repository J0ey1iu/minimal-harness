"""Tool selection action — handles Ctrl+T."""

from __future__ import annotations

from typing import TYPE_CHECKING

from minimal_harness.client.built_in.modals import ToolSelectScreen

if TYPE_CHECKING:
    from minimal_harness.client.built_in.app import TUIApp


def action_tools(app: TUIApp) -> None:
    if not app._all_tools:
        return
    selected = {t.name for t in app.active_tools}

    def done(chosen: list[str] | None) -> None:
        if chosen is None:
            return
        d = app._chat_display
        if d is None:
            return
        resolved = [app.ctx.all_tools[n] for n in chosen if n in app.ctx.all_tools]
        sess = app._ctrl.current_session
        if sess:
            app._ctrl.rebuild_current_session(
                llm_provider=app.ctx.create_llm_provider(),
                tools=resolved,
            )
            sess.tool_names = chosen
        names = ", ".join(t.name for t in resolved) or "(none)"
        d.say(f"\u2713 Tools: {names}", "bold bright_green")
        if app._first:
            app._banner()

    app.push_screen(ToolSelectScreen(app._all_tools, selected), done)
