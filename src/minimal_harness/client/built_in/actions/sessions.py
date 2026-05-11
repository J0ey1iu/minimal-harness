"""Session selection action — handles /sessions."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from minimal_harness.client.built_in.modals import SessionSelectScreen

if TYPE_CHECKING:
    from minimal_harness.client.built_in.app import TUIApp


def action_sessions(app: TUIApp) -> None:
    async def _load_sessions() -> None:
        metadata = await app._ctrl.get_all_sessions_metadata()

        def done(session_id: str | None) -> None:
            if not session_id or app._session_manager is None:
                return
            d = app._chat_display
            if d is None:
                return
            app._first = True
            asyncio.create_task(app._switch_to_session(session_id))

        app.push_screen(SessionSelectScreen(metadata, controller=app._ctrl), done)

    asyncio.create_task(_load_sessions())
