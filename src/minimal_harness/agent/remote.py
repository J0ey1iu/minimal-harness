from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterable, Sequence

if TYPE_CHECKING:
    from minimal_harness.agent.driver import RemoteAgentDriver
    from minimal_harness.memory import ExtendedInputContentPart, Memory
    from minimal_harness.tool.base import Tool
    from minimal_harness.types import AgentEvent


class RemoteAgent:
    """An ``Agent`` that delegates ``run()`` to a ``RemoteAgentDriver``.

    The driver handles all communication with an external agent service
    and maps its response stream back into ``AgentEvent`` types.

    Usage::

        agent = RemoteAgent(driver=SSEAgentDriver(binding))
        async for event in agent.run(
            user_input=[...],
            stop_event=...,
            memory=...,
            tools=[...],
        ):
            ...
    """

    def __init__(self, driver: RemoteAgentDriver) -> None:
        self._driver = driver

    def run(
        self,
        user_input: Iterable[ExtendedInputContentPart],
        stop_event: asyncio.Event | None = None,
        memory: Memory | None = None,
        tools: Sequence[Tool] | None = None,
        system_prompt: str = "",
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AgentEvent]:
        assert memory is not None, "memory must be provided"
        return self._driver.run(
            user_input=user_input,
            stop_event=stop_event,
            memory=memory,
            tools=tools or [],
            system_prompt=system_prompt,
            context=context,
            llm_kwargs=kwargs.get("llm_kwargs"),
        )
