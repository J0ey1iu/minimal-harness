"""Interrupt action — handles Escape."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minimal_harness.client.built_in.app import TUIApp


def action_interrupt(app: TUIApp) -> None:
    sid = app._ctrl.current_session_id
    if not sid or not app._ctrl.is_session_streaming(sid):
        return
    app._ctrl.interrupt()
