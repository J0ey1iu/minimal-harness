"""Compact action — injects a summarization prompt and auto-submits via the agent pipeline.

After the agent finishes responding, the forward offset is updated so future
LLM calls only see the compact summary, not the old conversation history.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from minimal_harness.client.built_in.session import SessionStatus

if TYPE_CHECKING:
    from minimal_harness.client.built_in.app import TUIApp

COMPACT_PROMPT = (
    "Summarize our conversation so far. Write in English.\n"
    "\n"
    "Include these sections:\n"
    "1. **Objective**: What I am trying to accomplish (the goal).\n"
    "2. **Completed Work**: What has been done so far — specific actions, "
    "decisions, and outcomes.\n"
    "3. **Files Modified**: List every file that was created, edited, or "
    "deleted, with a brief note of what changed.\n"
    "4. **Current Progress**: Where things stand right now. What is finished, "
    "what is in progress, and what remains to be done.\n"
    "5. **Key Context**: Important details, constraints, or decisions that "
    "must be remembered for future work.\n"
    "\n"
    "Be thorough but concise. Preserve technical details like file paths, "
    "command-line arguments, API names, and error messages exactly."
)


def action_compact(app: TUIApp) -> None:
    sess = app._ctrl.current_session
    if sess is None:
        app.notify("No active session to compact", severity="warning")
        return

    sid = sess.session.memory_id
    if app._ctrl.get_session_status(sid) == SessionStatus.RUNNING:
        app.notify("Cannot compact while agent is running", severity="warning")
        return

    if app._chat_display is None:
        app.notify("Display not ready", severity="warning")
        return

    app._input.text = COMPACT_PROMPT
    app._pending_compact = True
    app.action_submit()
