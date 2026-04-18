"""Framework client that emits events to a queue for decoupled consumption."""

import asyncio
from typing import Any, AsyncIterator, Iterable, Sequence

from minimal_harness.agent import OpenAIAgent
from minimal_harness.agent.protocol import InputContentConversionFunction
from minimal_harness.llm.openai import OpenAILLMProvider
from minimal_harness.memory import ExtendedInputContentPart, Memory
from minimal_harness.tool.base import StreamingTool

from .events import (
    ChunkEvent,
    DoneEvent,
    Event,
    ExecutionStartEvent,
    StoppedEvent,
    ToolEndEvent,
    ToolProgressEvent,
    ToolStartEvent,
)


class FrameworkClient:
    def __init__(
        self,
        llm_provider: OpenAILLMProvider,
        tools: Sequence[StreamingTool] | None = None,
        memory: Memory | None = None,
        max_iterations: int = 50,
    ) -> None:
        self._llm_provider = llm_provider
        self._tools = tools
        self._memory = memory
        self._max_iterations = max_iterations
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._stop_event: asyncio.Event | None = None
        self._agent_task: asyncio.Task[str] | None = None
        self._response: str = ""

    async def run(
        self,
        user_input: Iterable[ExtendedInputContentPart],
        custom_input_conversion: InputContentConversionFunction | None = None,
        stop_event: asyncio.Event | None = None,
    ) -> AsyncIterator[Event]:
        self._stop_event = stop_event
        self._agent_task = asyncio.create_task(
            self._run_agent(user_input, custom_input_conversion)
        )
        try:
            while True:
                event = await self._queue.get()
                yield event
                if isinstance(event, (DoneEvent, StoppedEvent)):
                    break
        except asyncio.CancelledError:
            if self._agent_task and not self._agent_task.done():
                self._agent_task.cancel()
            raise
        finally:
            if self._agent_task and not self._agent_task.done():
                self._agent_task.cancel()
            try:
                await self._agent_task
            except asyncio.CancelledError:
                pass

    async def _run_agent(
        self,
        user_input: Iterable[ExtendedInputContentPart],
        custom_input_conversion: InputContentConversionFunction | None = None,
    ) -> str:
        agent = OpenAIAgent(
            llm_provider=self._llm_provider,
            tools=self._tools,
            max_iterations=self._max_iterations,
            memory=self._memory,
        )

        async def on_chunk(chunk: Any | None, is_done: bool) -> None:
            await self._queue.put(ChunkEvent(chunk, is_done))

        async def on_execution_start(tool_calls: Any) -> None:
            await self._queue.put(ExecutionStartEvent(tool_calls))

        async def on_tool_start(tool_call: Any, _: Any) -> None:
            await self._queue.put(ToolStartEvent(tool_call, None))

        async def on_tool_progress(tc: Any, chunk: Any) -> None:
            await self._queue.put(ToolProgressEvent(tc, chunk))

        async def on_tool_end(tool_call: Any, result: Any) -> None:
            await self._queue.put(ToolEndEvent(tool_call, result))

        try:
            self._response = await agent.run(
                user_input=user_input,
                custom_input_conversion=custom_input_conversion,
                on_tool_start=on_tool_start,
                on_tool_end=on_tool_end,
                on_execution_start=on_execution_start,
                on_tool_progress=on_tool_progress,
                on_chunk=on_chunk,
                stop_event=self._stop_event,
            )
            await self._queue.put(DoneEvent(self._response))
        except asyncio.CancelledError:
            await self._queue.put(StoppedEvent(self._response))
            raise

        return self._response

    def stop(self) -> None:
        if self._stop_event:
            self._stop_event.set()
        if self._agent_task and not self._agent_task.done():
            self._agent_task.cancel()
