from __future__ import annotations

from typing import Iterable, cast

from minimal_harness.llm import LiteLLMProvider
from minimal_harness.memory import (
    ConversationMemory,
    ExtendedInputContentPart,
    Memory,
    Message,
    UserMessage,
)
from minimal_harness.tool import Tool
from minimal_harness.tool_executor import ToolExecutor
from minimal_harness.types import (
    ExecutionStartCallback,
    ToolEndCallback,
    ToolStartCallback,
    UserInputCallback,
)

from .protocol import InputContentConversionFunction


class LiteLLMAgent:
    def __init__(
        self,
        llm_provider: LiteLLMProvider,
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
    ) -> str:
        if on_tool_start:
            self._tool_executor._on_tool_start = on_tool_start
        if on_tool_end:
            self._tool_executor._on_tool_end = on_tool_end
        if on_execution_start:
            self._tool_executor._on_execution_start = on_execution_start
        if wait_for_user_input:
            self._tool_executor._wait_for_user_input = wait_for_user_input

        converted_user_input = user_input
        if custom_input_conversion:
            converted_user_input = await custom_input_conversion(converted_user_input)
        self._memory.add_message(
            cast(UserMessage, {"role": "user", "content": converted_user_input})
        )

        for _ in range(self._max_iterations):
            response = await self._llm_provider.chat(
                messages=self._memory.get_all_messages(),
                tools=list(self._tools.values()),
            )

            async for _ in response:
                pass

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

            results = await self._tool_executor.execute(llm_response.tool_calls)
            for msg in results:
                self._memory.add_message(msg)

        raise RuntimeError(
            f"Agent exceeded maximum iterations ({self._max_iterations})"
        )
