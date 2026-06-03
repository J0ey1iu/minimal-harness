"""Learn action — injects a reflection prompt and submits it."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minimal_harness.client.built_in.app import TUIApp

LEARN_PROMPT = (
    "Review our work session and extract concise, actionable lessons "
    "for the agent's system prompt. Focus on:\n"
    "1. **Operations**: What specific actions, patterns, or approaches "
    "worked well? State them as reusable directives (e.g. "
    '"Always run `npm run typecheck` after editing TypeScript").\n'
    "2. **Principles**: What rules, precautions, or guidelines should "
    'the agent follow in future tasks? (e.g. "Never commit secrets")\n'
    "\n"
    "Output a short list of bullet points only. "
    "Use the same language as this conversation."
)


def action_learn(app: TUIApp) -> None:
    app._input.text = LEARN_PROMPT
    app.action_submit()
