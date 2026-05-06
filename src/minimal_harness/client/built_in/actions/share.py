"""Export/share action — handles /share."""

from __future__ import annotations

from typing import TYPE_CHECKING

from minimal_harness.client.built_in.modals import PromptScreen

if TYPE_CHECKING:
    from minimal_harness.client.built_in.app import TUIApp


def action_share(app: TUIApp) -> None:
    sid = app._ctrl.current_session_id
    if sid and app._ctrl.is_session_streaming(sid):
        return
    d = app._chat_display
    e = app._exporter
    if d is None or e is None:
        return

    def done(path: str | None) -> None:
        if path:
            e.export_svg(
                path,
                export_history=d.export_history,
                chat_width=app._chat_width,
            )

    app.push_screen(
        PromptScreen("\U0001f4f8  Export chat as SVG", "./chat-container.svg"), done
    )
