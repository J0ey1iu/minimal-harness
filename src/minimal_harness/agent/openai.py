import asyncio
from typing import Iterable, cast

from minimal_harness.llm.openai import OpenAILLMProvider
from minimal_harness.memory import (
    ConversationMemory,
    ExtendedInputContentPart,
    Memory,
    Message,
    UserMessage,
)
from minimal_harness.tool import ProgressCallback, Tool, UserInputCallback
from minimal_harness.tool_executor import (
    ExecutionStartCallback,
    ToolEndCallback,
    ToolExecutor,
    ToolStartCallback,
)

from .protocol import InputContentConversionFunction


class OpenAIAgent:
    def __init__(
        self,
        llm_provider: OpenAILLMProvider,
        tools: list[Tool] | None = None,
        max_iterations: int = 50,
        memory: Memory | None = None,
        tool_executor: ToolExecutor | None = None,
        on_tool_start: ToolStartCallback | None = None,
        on_tool_end: ToolEndCallback | None = None,
        on_execution_start: ExecutionStartCallback | None = None,
        wait_for_user_input: UserInputCallback | None = None,
    ):
        self._llm_provider = llm_provider
        self._tools: dict[str, Tool] = {t.name: t for t in (tools or [])}
        self._tool_executor = tool_executor or ToolExecutor(
            self._tools,
            on_tool_start,
            on_tool_end,
            on_execution_start,
            wait_for_user_input,
        )
        self._max_iterations = max_iterations
        self._memory = memory or ConversationMemory()

    async def run(
        self,
        user_input: Iterable[ExtendedInputContentPart],
        custom_input_conversion: InputContentConversionFunction | None = None,
        on_tool_start: ToolStartCallback | None = None,
        on_tool_end: ToolEndCallback | None = None,
        on_execution_start: ExecutionStartCallback | None = None,
        wait_for_user_input: UserInputCallback | None = None,
        on_tool_progress: ProgressCallback | None = None,
        stop_event: asyncio.Event | None = None,
    ) -> str:
        if on_tool_start:
            self._tool_executor._on_tool_start = on_tool_start
        if on_tool_end:
            self._tool_executor._on_tool_end = on_tool_end
        if on_execution_start:
            self._tool_executor._on_execution_start = on_execution_start
        if wait_for_user_input:
            self._tool_executor._wait_for_user_input = wait_for_user_input
        if on_tool_progress:
            self._tool_executor._on_tool_progress = on_tool_progress

        converted_user_input = user_input
        if custom_input_conversion:
            converted_user_input = await custom_input_conversion(converted_user_input)
        self._memory.add_message(
            cast(UserMessage, {"role": "user", "content": converted_user_input})
        )

        try:
            for _ in range(self._max_iterations):
                if stop_event and stop_event.is_set():
                    break

                response = await self._llm_provider.chat(
                    messages=self._memory.get_all_messages(),
                    tools=list(self._tools.values()),
                    stop_event=stop_event,
                )

                async for _ in response:
                    if stop_event and stop_event.is_set():
                        break

                if stop_event and stop_event.is_set():
                    self._memory.add_message(
                        cast(
                            Message,
                            {
                                "role": "assistant",
                                "content": "[Response stopped by user]",
                                "tool_calls": None,
                            },
                        )
                    )
                    break

                llm_response = response.response
                self._memory.add_message(
                    cast(
                        Message,
                        {
                            "role": "assistant",
                            "content": llm_response.content,
                            "tool_calls": llm_response.tool_calls or None,
                        },
                    )
                )

                if llm_response.usage:
                    self._memory.add_usage(llm_response.usage)

                if not llm_response.tool_calls:
                    return str(llm_response.content) or ""

                results = await self._tool_executor.execute(
                    llm_response.tool_calls, stop_event
                )
                for msg in results:
                    self._memory.add_message(msg)

                if stop_event and stop_event.is_set():
                    break

        except asyncio.CancelledError:
            return str(self._memory.get_all_messages()[-1].get("content", "")) or ""

        if stop_event and stop_event.is_set():
            last = self._memory.get_all_messages()[-1]
            return str(last.get("content", "")) or ""

        raise RuntimeError(
            f"Agent exceeded maximum iterations ({self._max_iterations})"
        )
