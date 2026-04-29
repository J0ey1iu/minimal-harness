"""Memory dump action — handles Ctrl+D."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from minimal_harness.client.built_in.modals import PromptScreen

if TYPE_CHECKING:
    from minimal_harness.client.built_in.app import TUIApp


def action_dump(app: TUIApp) -> None:
    if app.memory is None:
        return
    memory = app.memory

    def done(path: str | None) -> None:
        if not path:
            return
        d = app._chat_display
        if d is None:
            return
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                memory.dump_memory_json(indent=2),
                encoding="utf-8",
            )
            d.say(f"\u2713 Memory dumped \u2192 {path}", "bold bright_green")
        except Exception as e:
            d.say(f"\u2717 {e}", "bold bright_red")

    app.push_screen(
        PromptScreen("\U0001f4be  Dump memory to file", "./memory_dump.json"), done
    )
