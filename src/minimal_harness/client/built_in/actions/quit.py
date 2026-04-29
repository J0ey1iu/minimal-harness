"""Quit confirmation action — handles Ctrl+C."""

from __future__ import annotations

from typing import TYPE_CHECKING

from minimal_harness.client.built_in.modals import ConfirmScreen

if TYPE_CHECKING:
    from minimal_harness.client.built_in.app import TUIApp


def action_request_quit(app: TUIApp) -> None:
    def done(ok: bool | None) -> None:
        if ok:
            app.exit()

    app.push_screen(
        ConfirmScreen("Quit?", "Session is saved.", ok="Quit", variant="error"),
        done,
    )
