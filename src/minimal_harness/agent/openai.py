import asyncio
import json
from typing import Any, AsyncIterator, Iterable, Sequence, cast

from minimal_harness.llm.openai import OpenAILLMProvider
from minimal_harness.memory import (
    ConversationMemory,
    ExtendedInputContentPart,
    Memory,
    Message,
    UserMessage,
)
from minimal_harness.tool.base import StreamingTool
from minimal_harness.types import (
    AgentEvent,
    AgentStart,
    Chunk,
    Done,
    ExecutionStart,
    Stopped,
    ToolCall,
    ToolEnd,
    ToolStart,
)

from .protocol import InputContentConversionFunction


class OpenAIAgent:
    def __init__(
        self,
        llm_provider: OpenAILLMProvider,
        tools: Sequence[StreamingTool] | None = None,
        max_iterations: int = 50,
        memory: Memory | None = None,
    ):
        self._llm_provider = llm_provider
        self._tools: dict[str, StreamingTool] = {t.name: t for t in (tools or [])}
        self._max_iterations = max_iterations
        self._memory = memory or ConversationMemory()

    async def run(
        self,
        user_input: Iterable[ExtendedInputContentPart],
        custom_input_conversion: InputContentConversionFunction | None = None,
        stop_event: asyncio.Event | None = None,
    ) -> AsyncIterator[AgentEvent]:
        yield AgentStart(user_input)

        converted_user_input = list(user_input)
        if custom_input_conversion:
            converted_user_input = list(
                await custom_input_conversion(converted_user_input)
            )
        self._memory.add_message(
            cast(UserMessage, {"role": "user", "content": converted_user_input})
        )

        response_text = ""
        exceeded_max_iterations = False
        try:
            for _ in range(self._max_iterations):
                if stop_event and stop_event.is_set():
                    break

                response = await self._llm_provider.chat(
                    messages=self._memory.get_all_messages(),
                    tools=list(self._tools.values()),
                    stop_event=stop_event,
                )

                async for chunk in response:
                    if stop_event and stop_event.is_set():
                        break
                    yield Chunk(chunk, False)

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
                    response_text = str(llm_response.content) or ""
                    break

                async for event in self._execute_tools(
                    llm_response.tool_calls, stop_event
                ):
                    yield event

                if stop_event and stop_event.is_set():
                    break
            else:
                exceeded_max_iterations = True

            if not response_text:
                last = self._memory.get_all_messages()[-1]
                response_text = str(last.get("content", "")) or ""

        except asyncio.CancelledError:
            response_text = (
                str(self._memory.get_all_messages()[-1].get("content", "")) or ""
            )
            yield Stopped(response_text)
            return

        yield Done(response_text)

        if exceeded_max_iterations:
            raise RuntimeError(
                f"Agent exceeded maximum iterations ({self._max_iterations})"
            )

    async def _execute_tools(
        self,
        tool_calls: list[ToolCall],
        stop_event: asyncio.Event | None,
    ) -> AsyncIterator[AgentEvent]:
        yield ExecutionStart(tool_calls)

        messages: list[Message] = []
        for tc in tool_calls:
            yield ToolStart(tc)
            result = await self._run_single_tool(tc, stop_event)
            if isinstance(result, asyncio.CancelledError):
                content = (
                    f"[Tool Execution Stopped] {tc['function']['name']}: cancelled"
                )
            elif isinstance(result, Exception):
                content = f"[Tool Error] {tc['function']['name']}: {result}"
            else:
                content = (
                    json.dumps(result, ensure_ascii=False)
                    if not isinstance(result, str)
                    else result
                )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": content,
                }
            )

            yield ToolEnd(tc, result)

        for msg in messages:
            self._memory.add_message(msg)

    async def _run_single_tool(
        self,
        tc: ToolCall,
        stop_event: asyncio.Event | None,
    ) -> Any:
        name = tc["function"]["name"]
        raw_args = tc["function"]["arguments"]

        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")

        tool = self._tools[name]
        args = json.loads(raw_args) if raw_args else {}

        if stop_event and stop_event.is_set():
            raise asyncio.CancelledError("Execution cancelled by user")

        final_result = None
        try:
            async for chunk in tool.fn(**args):
                if stop_event and stop_event.is_set():
                    raise asyncio.CancelledError("Execution cancelled by user")
                final_result = chunk

            return final_result
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return e
