"""Interrupt action — handles Escape."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minimal_harness.client.built_in.app import TUIApp


def action_interrupt(app: TUIApp) -> None:
    if not app._ctrl.streaming:
        return
    d = app._chat_display
    if d is None:
        return
    app._ctrl.interrupt()
    d.say("  \u2717 interrupted", "bold bright_red")
