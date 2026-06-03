"""Learn action — injects a reflection prompt into the input box."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minimal_harness.client.built_in.app import TUIApp

LEARN_PROMPT = (
    "Review our work session thoroughly and produce a structured summary covering:\n"
    "\n"
    "1. **Difficulties & Blockers**: What obstacles, pitfalls, or unexpected issues "
    "were encountered during the work?\n"
    "2. **Resolution Process**: How were these issues discovered and resolved? "
    "What specific steps were taken?\n"
    "3. **Lessons Learned**: What valuable insights were gained? "
    "What should be done differently next time?\n"
    "4. **Actionable Notes**: Based on this experience, provide clear precautions "
    "and best practices for future reference.\n"
    "\n"
    "**IMPORTANT**: Write your response in the same language the user has been "
    "using in this conversation."
)


def action_learn(app: TUIApp) -> None:
    app._input.text = LEARN_PROMPT
    app._input.focus()
