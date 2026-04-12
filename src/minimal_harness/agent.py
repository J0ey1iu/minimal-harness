import warnings
from typing import Awaitable, Callable, Iterable, Protocol, cast

from openai.types.chat import ChatCompletionChunk

from minimal_harness.llm import ChunkCallback, ToolResultCallback
from minimal_harness.llm.openai import OpenAILLMProvider
from minimal_harness.memory import (
    ConversationMemory,
    ExtendedInputContentPart,
    InputContentPart,
    Memory,
    Message,
    UserMessage,
)
from minimal_harness.tool import Tool
from minimal_harness.tool_executor import ToolExecutor

InputContentConversionFunction = Callable[
    [Iterable[ExtendedInputContentPart]], Awaitable[Iterable[InputContentPart]]
]


class Agent(Protocol):
    async def run(
        self,
        user_input: Iterable[InputContentPart],
        on_chunk: ChunkCallback | None = None,
        on_tool_result: ToolResultCallback | None = None,
    ) -> str: ...


class OpenAIAgent:
    def __init__(
        self,
        llm_provider: OpenAILLMProvider,
        tools: list[Tool] | None = None,
        max_iterations: int = 10,
        memory: Memory | None = None,
        tool_executor: ToolExecutor | None = None,
        on_tool_result: ToolResultCallback | None = None,
    ):
        warnings.warn(
            "OpenAIAgent is deprecated, use LiteLLMAgent instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self._llm_provider = llm_provider
        self._tools: dict[str, Tool] = {t.name: t for t in (tools or [])}
        self._tool_executor = tool_executor or ToolExecutor(self._tools, on_tool_result)
        self._max_iterations = max_iterations
        self._memory = memory or ConversationMemory()

    async def run(
        self,
        user_input: Iterable[ExtendedInputContentPart],
        custom_input_conversion: InputContentConversionFunction | None = None,
        on_chunk: ChunkCallback[ChatCompletionChunk] | None = None,
        on_tool_result: ToolResultCallback | None = None,
    ) -> str:
        if on_tool_result:
            self._tool_executor._on_tool_result = on_tool_result

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
                on_chunk=on_chunk,
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
