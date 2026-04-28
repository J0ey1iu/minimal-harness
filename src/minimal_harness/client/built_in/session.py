from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Sequence

if TYPE_CHECKING:
    from minimal_harness.agent import Agent
    from minimal_harness.client.built_in.memory import PersistentMemory
    from minimal_harness.llm import LLMProvider
    from minimal_harness.tool.base import Tool


@dataclass
class TUISession:
    session_id: str
    name: str
    agent: Agent
    memory: PersistentMemory
    tools: list[Tool]
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)

    def interrupt(self) -> None:
        self.stop_event.set()

    def reset(self) -> None:
        self.stop_event.clear()

    def rebuild(
        self,
        llm_provider: LLMProvider,
        tools: Sequence[Tool] | None = None,
        agent_factory: Callable[..., Agent] | None = None,
    ) -> None:
        from minimal_harness.agent.simple import SimpleAgent

        if tools is not None:
            self.tools = list(tools)

        factory = agent_factory or SimpleAgent
        self.agent = factory(
            llm_provider=llm_provider, tools=self.tools, memory=self.memory
        )
