import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Iterable, Sequence

from minimal_harness.llm.llm import LLMProvider
from minimal_harness.memory import (
    ExtendedInputContentPart,
    Memory,
    assistant_message,
    user_message,
)
from minimal_harness.tool.base import Tool
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
    MessageEvent,
    ToolCall,
    ToolEnd,
    ToolProgress,
    ToolResult,
    ToolStart,
)

from .middleware import Middleware
from .protocol import InputContentConversionFunction

logger = logging.getLogger(__name__)


class SimpleAgent:
    def __init__(
        self,
        llm_provider: LLMProvider,
        max_iterations: int,
        custom_input_conversion: InputContentConversionFunction | None = None,
        middleware: Sequence[Middleware] = (),
        emit_message_events: bool = False,
    ):
        self._llm_provider = llm_provider
        self._max_iterations = max_iterations
        self._custom_input_conversion = custom_input_conversion
        self._middleware = middleware
        self._emit_message_events = emit_message_events

    def run(
        self,
        user_input: Iterable[ExtendedInputContentPart],
        stop_event: asyncio.Event | None = None,
        memory: Memory | None = None,
        tools: Sequence[Tool] | None = None,
        system_prompt: str = "",
        context: dict[str, Any] | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AgentEvent]:
        """Run the agentic loop.

        memory and tools are required at runtime — the Agent Protocol
        declares them as optional for structural compatibility, but
        AgentRuntime always provides them.

        ``context`` is an opaque dict threaded through the execution
        pipeline. Middleware hooks (notably ``should_allow_tool``)
        receive its items as ``**kwargs`` so that implementations can
        access per-request state without the framework needing to know
        about it.
        """
        assert memory is not None, "memory must be provided"
        assert tools is not None, "tools must be provided"
        response_text = ""

        def _messages_with_system() -> list:
            msgs = memory.get_forward_messages()
            if system_prompt:
                msgs = [{"role": "system", "content": system_prompt}] + msgs
            return msgs

        async def agen() -> AsyncIterator[AgentEvent]:
            nonlocal response_text

            for m in self._middleware:
                await m.on_agent_start(user_input)
            yield AgentStart(user_input)
            start_time = time.time()

            converted_user_input = list(user_input)
            if self._custom_input_conversion:
                converted_user_input = list(
                    await self._custom_input_conversion(converted_user_input)
                )
            await memory.add_message(user_message(converted_user_input))

            response_text = ""
            exceeded_max_iterations = False
            llm_started = False
            try:
                for _ in range(self._max_iterations):
                    if stop_event and stop_event.is_set():
                        break

                    llm_messages = _messages_with_system()

                    for m in self._middleware:
                        await m.on_llm_start(llm_messages, tools)

                    yield LLMStart(
                        messages=llm_messages,
                        tools=[t.to_schema() for t in tools],
                    )
                    llm_started = True

                    response = await self._llm_provider.chat(
                        messages=llm_messages,
                        tools=tools,
                        stop_event=stop_event,
                        **(llm_kwargs or {}),
                    )
                    async for chunk in response:
                        yield LLMChunk(chunk=chunk)

                    llm_response = response.response
                    llm_end = LLMEnd(
                        llm_response.content,
                        llm_response.reasoning_content,
                        llm_response.tool_calls,
                        llm_response.usage,
                    )
                    for m in self._middleware:
                        await m.on_llm_end(llm_end)
                    yield llm_end
                    llm_started = False

                    if llm_response.reasoning_content:
                        await memory.add_message(
                            {
                                "role": "reasoning",
                                "content": llm_response.reasoning_content,
                            }
                        )
                        if self._emit_message_events:
                            yield MessageEvent(
                                message={
                                    "role": "reasoning",
                                    "content": llm_response.reasoning_content,
                                }
                            )
                    await memory.add_message(
                        assistant_message(
                            llm_response.content, llm_response.tool_calls or None
                        )
                    )
                    if self._emit_message_events:
                        yield MessageEvent(
                            message={
                                "role": "assistant",
                                "content": llm_response.content,
                                "tool_calls": llm_response.tool_calls or None,
                            }
                        )

                    if llm_response.usage:
                        memory.set_message_usage(llm_response.usage)
                        yield MemoryUpdate(llm_response.usage)

                    if not llm_response.tool_calls:
                        response_text = str(llm_response.content) or ""
                        break

                    async for event in self._execute_tools(
                        llm_response.tool_calls, stop_event, tools, memory, context
                    ):
                        yield event

                else:
                    exceeded_max_iterations = True

                if not response_text:
                    for msg in reversed(memory.get_all_messages()):
                        if msg.get("role") == "assistant" and msg.get("content"):
                            response_text = str(msg.get("content", ""))
                            break

            except asyncio.CancelledError:
                if llm_started:
                    yield LLMEnd(
                        content="",
                        reasoning_content=None,
                        tool_calls=[],
                        usage=None,
                        error="LLM call interrupted",
                    )
                agent_end = AgentEnd(
                    "",
                    time.time() - start_time,
                    interrupted=True,
                )
                for m in self._middleware:
                    await m.on_agent_end(agent_end)
                yield agent_end
                return

            except Exception as exc:
                logger.exception("agent.step.error")
                error_msg = f"{type(exc).__name__}: {exc}"
                if llm_started:
                    yield LLMEnd(
                        content="",
                        reasoning_content=None,
                        tool_calls=[],
                        usage=None,
                        error=error_msg,
                    )
                for m in self._middleware:
                    await m.on_error(exc)
                agent_end = AgentEnd(
                    str(exc),
                    time.time() - start_time,
                    error=error_msg,
                )
                for m in self._middleware:
                    await m.on_agent_end(agent_end)
                yield agent_end
                return

            agent_end = AgentEnd(
                response_text,
                time.time() - start_time,
                exceeded=exceeded_max_iterations,
            )
            for m in self._middleware:
                await m.on_agent_end(agent_end)
            yield agent_end

        return agen()

    @staticmethod
    def _serialize_content_for_llm(result: Any) -> str:
        if isinstance(result, dict):
            return json.dumps(
                {k: v for k, v in result.items() if not k.startswith("_")},
                ensure_ascii=False,
            )
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)

    async def _execute_tools(
        self,
        tool_calls: list[ToolCall],
        stop_event: asyncio.Event | None,
        tools: Sequence[Tool],
        memory: Memory,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        yield ExecutionStart(tool_calls)

        tools_dict = {t.name: t for t in tools}
        unknown_tool_calls = []
        known_tool_calls = []
        for tc in tool_calls:
            name = tc["function"]["name"]
            if name not in tools_dict:
                unknown_tool_calls.append(tc)
            else:
                known_tool_calls.append(tc)

        results_by_id: dict[str, tuple[ToolCall, Any]] = {}
        progress_data: dict[str, list[str]] = {}
        event_queue: asyncio.Queue[Any] = asyncio.Queue()
        exec_results: list[tuple[ToolCall, Any]] = []

        async def run_single(tc: ToolCall) -> None:
            tool = tools_dict[tc["function"]["name"]]
            raw_args = tc["function"]["arguments"]
            args = json.loads(raw_args) if raw_args else {}
            result = None
            progress_chunks: list[str] = []
            try:
                for m in self._middleware:
                    allow = await m.should_allow_tool(tc, **(context or {}))
                    if allow is not True:
                        reason = (
                            allow
                            if isinstance(allow, str)
                            else f"Tool '{tc['function']['name']}' execution denied by policy"
                        )
                        permission_error = PermissionError(reason)
                        await event_queue.put(ToolStart(tc))
                        for m2 in self._middleware:
                            await m2.on_tool_start(tc)
                            await m2.on_tool_error(tc, permission_error)
                        await event_queue.put(ToolEnd(tc, permission_error))
                        results_by_id[tc["id"]] = (tc, permission_error)
                        await event_queue.put(None)
                        return

                for m in self._middleware:
                    await m.on_tool_start(tc)
                async for event in tool.execute(args, tc, stop_event):
                    await event_queue.put(event)
                    if isinstance(event, ToolEnd):
                        result = event.result
                    elif isinstance(event, ToolProgress):
                        chunk = event.chunk
                        progress_chunks.append(
                            json.dumps(chunk, ensure_ascii=False, default=str)
                            if not isinstance(chunk, str)
                            else chunk
                        )
            except asyncio.CancelledError:
                if result is None:
                    result = RuntimeError("Tool execution was interrupted")
                for m in self._middleware:
                    await m.on_tool_error(tc, result)
                await event_queue.put(ToolEnd(tc, result))
                results_by_id[tc["id"]] = (tc, result)
                if progress_chunks:
                    progress_data[tc["id"]] = progress_chunks
                await event_queue.put(None)
                raise
            except Exception as exc:
                result = exc
                for m in self._middleware:
                    await m.on_tool_error(tc, exc)
                await event_queue.put(ToolEnd(tc, result))
            else:
                for m in self._middleware:
                    await m.on_tool_end(tc, result)
            results_by_id[tc["id"]] = (tc, result)
            if progress_chunks:
                progress_data[tc["id"]] = progress_chunks
            await event_queue.put(None)

        tasks = [asyncio.create_task(run_single(tc)) for tc in known_tool_calls]
        remaining = len(tasks)

        for tc in unknown_tool_calls:
            err = ValueError(f"Unknown tool: {tc['function']['name']}")
            for m in self._middleware:
                await m.on_tool_start(tc)
                await m.on_tool_error(tc, err)
            await event_queue.put(ToolStart(tc))
            await event_queue.put(ToolEnd(tc, err))
            results_by_id[tc["id"]] = (tc, err)
            remaining += 1
            await event_queue.put(None)

        try:
            try:
                while remaining > 0:
                    if stop_event and stop_event.is_set():
                        break
                    effective_timeout = 1.0 if stop_event else None
                    try:
                        item = await asyncio.wait_for(
                            event_queue.get(), timeout=effective_timeout
                        )
                    except asyncio.TimeoutError:
                        if stop_event and not stop_event.is_set():
                            continue
                        break
                    if item is None:
                        remaining -= 1
                    else:
                        yield item
            finally:
                for t in tasks:
                    if not t.done():
                        t.cancel()

            if tasks:
                done, pending = await asyncio.wait(tasks, timeout=10.0)
                for p in pending:
                    p.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

            if remaining > 0:
                while remaining > 0:
                    item = await event_queue.get()
                    if item is None:
                        remaining -= 1
                    else:
                        yield item

            exec_results = [
                results_by_id[tc["id"]]
                for tc in tool_calls
                if tc["id"] in results_by_id
            ]

            for tc, result in exec_results:
                if isinstance(result, Exception):
                    content = f"[Error] {result}"
                    result_meta = None
                elif isinstance(result, ToolResult):
                    content = self._serialize_content_for_llm(result.content)
                    result_meta = result.meta
                else:
                    content = self._serialize_content_for_llm(result)
                    result_meta = None
                tool_msg: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": content,
                }
                if result_meta:
                    tool_msg["meta"] = result_meta
                tc_progress = progress_data.get(tc["id"])
                if tc_progress:
                    tool_msg["progress"] = tc_progress
                await memory.add_message(tool_msg)  # type: ignore[arg-type]
                if self._emit_message_events:
                    yield MessageEvent(message=tool_msg)
        except (Exception, asyncio.CancelledError) as exc:
            yield ExecutionEnd(exec_results, error=f"{type(exc).__name__}: {exc}")
            raise

        yield ExecutionEnd(exec_results)
