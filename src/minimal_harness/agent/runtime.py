from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Iterable, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    from minimal_harness.agent.protocol import Agent
    from minimal_harness.memory import ExtendedInputContentPart, Memory
    from minimal_harness.tool.base import Tool
    from minimal_harness.types import AgentEvent


@runtime_checkable
class AgentRuntimeProtocol(Protocol):
    """Stateless async task manager for running agents.

    The runtime's only responsibility is to start an agent run and return
    controls (stop_event, event_queue) to the caller. The caller manages
    all state, event consumption, and lifecycle.
    """

    def run(
        self,
        agent: Agent,
        memory: Memory,
        tools: Sequence[Tool],
        user_input: Iterable[ExtendedInputContentPart],
    ) -> tuple[asyncio.Event, asyncio.Queue[AgentEvent | None]]: ...


class AgentRuntime:
    """Stateless async task manager.

    Usage::

        stop_event, queue = runtime.run(agent, memory, tools, user_input)
        while True:
            event = await queue.get()
            if event is None:
                break
            # process event
    """

    def run(
        self,
        agent: Agent,
        memory: Memory,
        tools: Sequence[Tool],
        user_input: Iterable[ExtendedInputContentPart],
    ) -> tuple[asyncio.Event, asyncio.Queue[AgentEvent | None]]:
        stop_event = asyncio.Event()
        event_queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()

        async def _run() -> None:
            try:
                async for event in agent.run(
                    user_input=user_input,
                    stop_event=stop_event,
                    memory=memory,
                    tools=tools,
                ):
                    await event_queue.put(event)
            finally:
                await event_queue.put(None)

        asyncio.create_task(_run())
        return stop_event, event_queue
