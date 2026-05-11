"""Memory dump action — handles Ctrl+D."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from minimal_harness.client.built_in.modals import PromptScreen

if TYPE_CHECKING:
    from minimal_harness.client.built_in.app import TUIApp


def action_dump(app: TUIApp) -> None:
    sid = app._ctrl.current_session_id
    if sid is None:
        return

    def done(path: str | None) -> None:
        if not path:
            return
        d = app._chat_display
        if d is None:
            return

        async def _do_dump() -> None:
            try:
                p = Path(path)
                p.parent.mkdir(parents=True, exist_ok=True)
                content = await app.ctx.memory_store.export_memory_json(sid, indent=2)
                p.write_text(content, encoding="utf-8")
                d.say(f"\u2713 Memory dumped \u2192 {path}", "bold bright_green")
            except Exception as e:
                d.say(f"\u2717 {e}", "bold bright_red")

        asyncio.create_task(_do_dump())

    app.push_screen(
        PromptScreen("\U0001f4be  Dump memory to file", "./memory_dump.json"), done
    )
