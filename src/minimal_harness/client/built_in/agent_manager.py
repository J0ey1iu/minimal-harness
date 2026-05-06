"""Agent preset registration and default agent lifecycle."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from minimal_harness.client.built_in.config.agents import (
    SYSTEM_PROMPTS_DIR,
    load_agents_config,
    read_system_prompt,
)
from minimal_harness.client.built_in.session import ConversationSession

if TYPE_CHECKING:
    from minimal_harness.agent.registry import AgentRegistryProtocol
    from minimal_harness.client.built_in.context import AppContext


class AgentManager:
    def __init__(
        self,
        ctx: AppContext,
        agent_registry: AgentRegistryProtocol,
    ) -> None:
        self._ctx = ctx
        self._agent_registry = agent_registry
        self._sessions: dict[str, ConversationSession] = {}

    @property
    def sessions(self) -> dict[str, ConversationSession]:
        return self._sessions

    def register_preset_agents(self) -> None:
        from minimal_harness.agent.registry import AgentMetadata

        agents = load_agents_config()
        if not agents:
            return
        for a in agents:
            prompt_path = SYSTEM_PROMPTS_DIR / a["system_prompt"]
            system_prompt = read_system_prompt(prompt_path) or a.get("description", "")
            default_tools = a.get("default_tools") or []

            resolved_tool_names = [n for n in default_tools if n in self._ctx.all_tools]

            metadata = AgentMetadata(
                name=a["name"],
                description=a.get("description", ""),
                system_prompt=system_prompt,
                agent_type="simple",
                tool_names=resolved_tool_names,
                metadata_id=a["name"],
            )
            self._agent_registry.register(metadata)

    def start_with_default_agent(
        self,
        create_session_fn: Any,
    ) -> None:
        agents = load_agents_config()
        default_name = self._ctx.config.get("default_agent", "general_assistant")
        agent_cfg = self._get_default_agent(agents, default_name)
        if agent_cfg:
            create_session_fn(
                agent_name=agent_cfg["name"],
                default_tools=agent_cfg.get("default_tools"),
            )
        else:
            create_session_fn()

    @staticmethod
    def _get_default_agent(
        agents: list[dict[str, Any]],
        default_name: str = "general_assistant",
    ) -> dict[str, Any] | None:
        for a in agents:
            if a.get("name") == default_name:
                return a
        for a in agents:
            if a.get("name") == "general_assistant":
                return a
        return agents[0] if agents else None
