import asyncio
import json
from typing import Any, AsyncIterator, Iterable, Sequence

from minimal_harness.llm.llm import LLMProvider
from minimal_harness.memory import (
    ConversationMemory,
    ExtendedInputContentPart,
    Memory,
    assistant_message,
    user_message,
)
from minimal_harness.settings import Settings
from minimal_harness.tool.base import StreamingTool
from minimal_harness.types import (
    AgentEnd,
    AgentEvent,
    AgentStart,
    ExecutionEnd,
    ExecutionStart,
    LLMChunk,
    LLMEnd,
    LLMStart,
    MemoryUpdate,
    ToolCall,
    ToolEnd,
)

from .protocol import InputContentConversionFunction


class SimpleAgent:
    def __init__(
        self,
        llm_provider: LLMProvider,
        tools: Sequence[StreamingTool] | None = None,
        max_iterations: int | None = None,
        memory: Memory | None = None,
        custom_input_conversion: InputContentConversionFunction | None = None,
    ):
        self._llm_provider = llm_provider
        self._tools: dict[str, StreamingTool] = {t.name: t for t in (tools or [])}
        self._max_iterations = (
            max_iterations if max_iterations is not None else Settings.max_iterations()
        )
        self._memory = memory or ConversationMemory()
        self._custom_input_conversion = custom_input_conversion

    def run(
        self,
        user_input: Iterable[ExtendedInputContentPart],
        stop_event: asyncio.Event | None = None,
        memory: Memory | None = None,
        tools: Sequence[StreamingTool] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        response_text = ""
        stopped = False
        effective_memory = memory or self._memory
        effective_tools = tools or list(self._tools.values())

        async def agen() -> AsyncIterator[AgentEvent]:
            nonlocal response_text, stopped

            yield AgentStart(user_input)

            memory = effective_memory
            tools = effective_tools

            converted_user_input = list(user_input)
            if self._custom_input_conversion:
                converted_user_input = list(
                    await self._custom_input_conversion(converted_user_input)
                )
            memory.add_message(user_message(converted_user_input))

            response_text = ""
            exceeded_max_iterations = False
            stopped = False
            try:
                for _ in range(self._max_iterations):
                    if stop_event and stop_event.is_set():
                        stopped = True
                        break

                    response = await self._llm_provider.chat(
                        messages=memory.get_forward_messages(),
                        tools=tools,
                        stop_event=stop_event,
                    )

                    yield LLMStart(
                        messages=memory.get_forward_messages(),
                        tools=[t.to_schema() for t in tools],
                    )
                    async for chunk in response:
                        if stop_event and stop_event.is_set():
                            stopped = True
                            break
                        yield LLMChunk(chunk, False)

                    if stopped or (stop_event and stop_event.is_set()):
                        memory.add_message(
                            assistant_message("[Response stopped by user]", None)
                        )
                        yield LLMEnd(
                            "[Response stopped by user]",
                            None,
                            [],
                            None,
                        )
                        break

                    llm_response = response.response
                    yield LLMEnd(
                        llm_response.content,
                        llm_response.reasoning_content,
                        llm_response.tool_calls,
                        llm_response.usage,
                    )
                    memory.add_message(
                        assistant_message(
                            llm_response.content, llm_response.tool_calls or None
                        )
                    )

                    if llm_response.usage:
                        memory.set_message_usage(llm_response.usage)
                        yield MemoryUpdate(llm_response.usage)

                    if not llm_response.tool_calls:
                        response_text = str(llm_response.content) or ""
                        break

                    async for event in self._execute_tools(
                        llm_response.tool_calls, stop_event, memory, tools
                    ):
                        yield event

                    if stop_event and stop_event.is_set():
                        stopped = True
                        break
                else:
                    exceeded_max_iterations = True

                if not response_text:
                    last = memory.get_all_messages()[-1]
                    response_text = str(last.get("content", "")) or ""

            except asyncio.CancelledError:
                response_text = (
                    str(memory.get_all_messages()[-1].get("content", "")) or ""
                )
                yield AgentEnd(response_text)
                return

            yield AgentEnd(response_text)

            if exceeded_max_iterations:
                raise RuntimeError(
                    f"Agent exceeded maximum iterations ({self._max_iterations})"
                )

        return agen()

    async def _execute_tools(
        self,
        tool_calls: list[ToolCall],
        stop_event: asyncio.Event | None,
        memory: Memory,
        tools: Sequence[StreamingTool],
    ) -> AsyncIterator[AgentEvent]:
        yield ExecutionStart(tool_calls)

        tools_dict = {t.name: t for t in tools}
        results: list[tuple[ToolCall, Any]] = []

        for tc in tool_calls:
            name = tc["function"]["name"]
            raw_args = tc["function"]["arguments"]

            if name not in tools_dict:
                raise ValueError(f"Unknown tool: {name}")

            tool = tools_dict[name]
            args = json.loads(raw_args) if raw_args else {}

            async for event in tool.execute(args, tc, stop_event):
                yield event
                if isinstance(event, ToolEnd):
                    result = event.result
                    results.append((tc, result))
                    if isinstance(result, asyncio.CancelledError):
                        content = f"[Tool Execution Stopped] {tc['function']['name']}: cancelled"
                    elif isinstance(result, Exception):
                        content = f"[Tool Error] {tc['function']['name']}: {result}"
                    else:
                        content = (
                            json.dumps(result, ensure_ascii=False)
                            if not isinstance(result, str)
                            else result
                        )
                    memory.add_message(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": content,
                        }
                    )
        yield ExecutionEnd(results)
