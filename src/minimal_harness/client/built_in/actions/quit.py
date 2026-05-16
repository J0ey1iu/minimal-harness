"""Quit confirmation action — handles Ctrl+C."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from minimal_harness.client.built_in.modals import ConfirmScreen

if TYPE_CHECKING:
    from minimal_harness.client.built_in.app import TUIApp


def action_request_quit(app: TUIApp) -> None:
    async def _do_quit() -> None:
        for sid in list(app._ctrl._active_runs):
            task, stop_event, _ = app._ctrl._active_runs[sid]
            stop_event.set()
            task.cancel()
        await asyncio.sleep(0)
        app.exit()

    def done(ok: bool | None) -> None:
        if ok:
            asyncio.create_task(_do_quit())

    app.push_screen(
        ConfirmScreen("Quit?", "Session is saved.", ok="Quit", variant="error"),
        done,
    )
