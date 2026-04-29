"""New conversation action — handles Ctrl+N."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from minimal_harness.client.built_in.config.agents import (
    SYSTEM_PROMPTS_DIR,
    load_agents_config,
    read_system_prompt,
)
from minimal_harness.client.built_in.modals import AgentSelectScreen, ConfirmScreen

if TYPE_CHECKING:
    from minimal_harness.client.built_in.app import TUIApp


def action_new(app: TUIApp) -> None:
    agents = load_agents_config()

    def _pick_agent() -> None:
        def on_agent(agent: dict[str, Any] | None) -> None:
            if not agent:
                return
            d = app._chat_display
            if d is None:
                return
            prompt = read_system_prompt(
                SYSTEM_PROMPTS_DIR / agent["system_prompt"]
            ) or agent.get("description", "")
            d.clear_chat()
            app._ctrl.buf.clear()
            app._first = True
            app._ctrl.create_session(
                agent_name=agent["name"],
                system_prompt=prompt,
                default_tools=agent.get("default_tools"),
            )
            app._banner_widget.display = True
            app._chat.display = False
            app._banner()
            app._update_top_bar()

        app.push_screen(AgentSelectScreen(agents), on_agent)

    if app._first:
        _pick_agent()
    else:
        app.push_screen(
            ConfirmScreen(
                "Start new chat?",
                "Session is saved.",
                ok="New Chat",
                variant="primary",
            ),
            lambda ok: _pick_agent() if ok else None,
        )
