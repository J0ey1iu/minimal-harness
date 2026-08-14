"""Shared agent loop used by :class:`SimpleAgent` and :class:`CompactionAgent`.

Both agents share the same agentic loop: system-prompt injection,
LLM call, tool execution, event emission, error handling, and the
``MaxIterationsExceeded`` fallback. The only difference is the
*post-LLM hook* — :class:`CompactionAgent` overrides
:meth:`BaseAgent._post_llm_response` to run a context fold before the
next iteration, while :class:`SimpleAgent` uses the default no-op.

Pulling the loop out of both classes removes the ~90% duplication
that lived in the original ``simple.py`` / ``compacting.py`` files
(commit ``76051c2`` introduced the duplication; this module is the
remedy). Bug fixes to the loop, tool execution, or error handling
now live in a single place.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Iterable, Sequence

from minimal_harness.llm.llm import LLMProvider
from minimal_harness.memory import (
    ExtendedInputContentPart,
    Memory,
    Message,
    assistant_message,
    user_message,
    verify_memory_contract,
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

# 每个工具调用持久化的进度事件上限：长输出命令（bash 大目录遍历等）
# 会产出海量 ToolProgress，全量落盘会使会话文件与后续 LLM 请求膨胀。
MAX_PERSISTED_PROGRESS_CHUNKS = 40

logger = logging.getLogger(__name__)


def _format_exception(exc: BaseException) -> str:
    """Render an exception with provider-specific diagnostic fields."""
    _exc_status_code = getattr(exc, "status_code", None)
    _exc_code = getattr(exc, "code", None)
    _exc_type = getattr(exc, "type", None)
    _exc_request_id = getattr(exc, "request_id", None)
    _exc_body = getattr(exc, "body", None)
    _error_parts = [f"{type(exc).__name__}: {exc}"]
    if _exc_status_code is not None:
        _error_parts.append(f"http_status={_exc_status_code}")
    if _exc_code:
        _error_parts.append(f"code={_exc_code}")
    if _exc_type:
        _error_parts.append(f"type={_exc_type}")
    if _exc_request_id:
        _error_parts.append(f"request_id={_exc_request_id}")
    if _exc_body is not None:
        _body_str = str(_exc_body)
        if len(_body_str) > 500:
            _body_str = _body_str[:500] + "..."
        _error_parts.append(f"body={_body_str}")
    return " | ".join(_error_parts)


def _serialize_content_for_llm(result: Any) -> str:
    if isinstance(result, dict):
        return json.dumps(
            {k: v for k, v in result.items() if not k.startswith("_")},
            ensure_ascii=False,
        )
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, default=str)


def _last_assistant_message_id(memory: Memory) -> str | None:
    """Return the canonical id of the last assistant message in the live
    buffer (``Memory.add_message`` stamps it), or ``None`` if the run never
    produced one (error/interrupt before any assistant turn).
    """
    for msg in reversed(memory.get_all_messages()):
        if msg.get("role") == "assistant":
            mid = msg.get("id")
            return mid if isinstance(mid, str) else None
    return None


class BaseAgent:
    """Agentic loop shared by simple and compacting agents.

    Subclasses override :meth:`_post_llm_response` to inject behaviour
    after the LLM turn but before the next iteration (or before tool
    execution in the same turn). :class:`CompactionAgent` uses this
    hook to run :meth:`Memory.compact`; :class:`SimpleAgent` is the
    default no-op.

    The lifecycle of one ``run()`` call is:

    1. :meth:`_pre_agent_start` — subclass hook, called once before
       any LLM call. The default yields ``AgentStart``.
    2. For each iteration up to ``max_iterations``:
       a. Yield ``LLMStart``, await LLM, yield ``LLMChunk`` and
          ``LLMEnd``.
       b. Record the assistant turn (and optional reasoning) in
          memory and emit ``MessageEvent`` for each.
       c. :meth:`_post_llm_response` — subclass hook.
       d. If the LLM produced no tool calls, break out of the loop.
       e. :meth:`_execute_tools` — runs the tool calls, yields tool
          events, records tool messages.
       f. :meth:`_post_tool_execution` — subclass hook (after tool
          execution, before the next LLM call in the same iteration).
       g. (back to step a for the next iteration).
    3. :meth:`_finalize` — picks the final response text and yields
       ``AgentEnd`` (with ``exceeded=True`` if the iteration budget
       was hit).

    Cancellation and exceptions are both routed through a single
    error-handling block that emits ``LLMEnd(error=...)`` /
    ``AgentEnd(error=...)`` and calls the middleware hooks.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        max_iterations: int = 2000,
        custom_input_conversion: InputContentConversionFunction | None = None,
        middleware: Sequence[Middleware] = (),
        emit_message_events: bool = True,
    ):
        self._llm_provider = llm_provider
        self._max_iterations = max_iterations
        self._custom_input_conversion = custom_input_conversion
        self._middleware = middleware
        self._emit_message_events = emit_message_events

    async def _post_llm_response(
        self,
        llm_response: Any,
        memory: Memory,
    ) -> AsyncIterator[AgentEvent]:
        """Hook fired after the LLM response is recorded in memory.

        Subclasses yield additional events (e.g. ``CompactionStart``)
        and may mutate memory (e.g. fold old messages). The default
        implementation is a no-op. Errors raised here are caught by
        the loop and surfaced through ``AgentEnd.error`` — they do
        not abort the iteration by themselves.
        """
        return
        yield  # Make this an async generator.

    async def _on_run_end(
        self,
        memory: Memory,
    ) -> AsyncIterator[AgentEvent]:
        """Hook called after the entire agent loop finishes, before AgentEnd.

        Subclasses may override to run post-loop operations such as
        conversation compaction.  The default implementation is a no-op.
        """
        return
        yield  # Make this an async generator.

    async def _on_run_error(
        self,
        memory: Memory,
    ) -> AsyncIterator[AgentEvent]:
        """Hook called when the agent loop exits with an exception.

        Subclasses may override to run lightweight cleanup that does
        **not** call an LLM (e.g. stripping dangling tool call pairs).
        The default implementation is a no-op.
        """
        return
        yield  # Make this an async generator.

    async def _post_tool_execution(
        self,
        memory: Memory,
    ) -> AsyncIterator[AgentEvent]:
        """Hook fired after tool execution, before the next LLM call.

        Subclasses yield additional events (e.g. ``CompactionStart``)
        and may mutate memory (e.g. compress tool results). The default
        implementation is a no-op. Errors raised here are caught by
        the loop and surfaced through ``AgentEnd.error`` — they do
        not abort the iteration by themselves.
        """
        return
        yield  # Make this an async generator.

    def run(
        self,
        user_input: Iterable[ExtendedInputContentPart],
        stop_event: asyncio.Event | None = None,
        memory: Memory | None = None,
        tools: Sequence[Tool] | None = None,
        system_prompt: str = "",
        context: dict[str, Any] | None = None,
        llm_kwargs: dict[str, Any] | None = None,
        user_message_meta: dict[str, Any] | None = None,
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

        ``user_message_meta`` is an optional dict merged into the
        persisted user message (e.g. ``{"source": "auto"}`` to tag a
        controller-generated prompt). It is stored with the message but
        stripped from LLM payloads by the providers' message converters.
        """
        assert memory is not None, "memory must be provided"
        assert tools is not None, "tools must be provided"
        # Fail fast at run start, not mid-stream on an edge path (mh-incubator #58).
        verify_memory_contract(memory)
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
            user_msg = user_message(converted_user_input)
            if user_message_meta:
                user_msg.update(user_message_meta)  # type: ignore[call-overload]
            await memory.add_message(user_msg)

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

                    # Accumulate streaming content so partial responses
                    # can be saved to memory if the stream errors out.
                    accumulated_content = ""
                    accumulated_reasoning = ""
                    try:
                        async for chunk in response:
                            if chunk:
                                if chunk.content:
                                    accumulated_content += chunk.content
                                if chunk.reasoning:
                                    accumulated_reasoning += chunk.reasoning
                            yield LLMChunk(chunk=chunk)
                    except Exception:
                        # Partial content received — save it before
                        # re-raising so it isn't permanently lost.
                        if accumulated_content or accumulated_reasoning:
                            if accumulated_reasoning:
                                partial_reasoning_msg: Message = {
                                    "role": "reasoning",
                                    "content": accumulated_reasoning,
                                }
                                await memory.add_message(partial_reasoning_msg)
                                if self._emit_message_events:
                                    yield MessageEvent(
                                        message=dict(partial_reasoning_msg)
                                    )
                            partial_assistant_msg = assistant_message(
                                accumulated_content,
                                tool_calls=None,
                            )
                            await memory.add_message(partial_assistant_msg)
                            if self._emit_message_events:
                                yield MessageEvent(message=dict(partial_assistant_msg))
                            logger.warning(
                                "agent.stream.partial-saved role=assistant chars=%d",
                                len(accumulated_content),
                            )
                        raise

                    llm_response = response.response

                    # Persist the produced messages BEFORE broadcasting
                    # LLMEnd, so the event can carry the canonical ids the
                    # messages just received (streaming consumers need them
                    # as soon as the turn completes — not only at AgentEnd).
                    if llm_response.reasoning_content:
                        reasoning_msg: Message = {
                            "role": "reasoning",
                            "content": llm_response.reasoning_content,
                        }
                        await memory.add_message(reasoning_msg)
                        if self._emit_message_events:
                            yield MessageEvent(message=dict(reasoning_msg))
                    assistant_msg = assistant_message(
                        llm_response.content, llm_response.tool_calls or None
                    )
                    await memory.add_message(assistant_msg)
                    if self._emit_message_events:
                        yield MessageEvent(message=dict(assistant_msg))

                    llm_end = LLMEnd(
                        llm_response.content,
                        llm_response.reasoning_content,
                        llm_response.tool_calls,
                        llm_response.usage,
                        message_id=assistant_msg.get("id"),
                    )
                    for m in self._middleware:
                        await m.on_llm_end(llm_end)
                    yield llm_end
                    llm_started = False

                    if llm_response.usage:
                        memory.set_message_usage(llm_response.usage)
                        yield MemoryUpdate(llm_response.usage)

                    # Subclass hook: compaction, retry-on-failure, etc.
                    async for hook_evt in self._post_llm_response(llm_response, memory):
                        yield hook_evt

                    if not llm_response.tool_calls:
                        if not llm_response.content:
                            # Model produced nothing: no text, no tool calls.
                            # Ending the run here used to look like a silent
                            # stop (the replay fallback then surfaced stale
                            # text from a previous round) — surface it as an
                            # error instead (mh-incubator #58).
                            raise RuntimeError(
                                "LLM returned an empty response (no content, "
                                "no tool calls)"
                            )
                        response_text = str(llm_response.content)
                        break

                    should_stop = False
                    async for event in self._execute_tools(
                        llm_response.tool_calls, stop_event, tools, memory, context
                    ):
                        if isinstance(event, ExecutionEnd) and event.should_stop:
                            should_stop = True
                            if event.response_text:
                                response_text = event.response_text
                        yield event
                    if should_stop:
                        break

                    # Subclass hook: tool-result compression, etc.
                    async for hook_evt in self._post_tool_execution(memory):
                        yield hook_evt

                else:
                    exceeded_max_iterations = True

                if not response_text:
                    for msg in reversed(memory.get_replay_messages()):
                        if msg.get("role") == "assistant" and msg.get("content"):
                            response_text = str(msg.get("content", ""))
                            break

                # Run-end hook: compaction, cleanup, etc.
                async for evt in self._on_run_end(memory):
                    yield evt

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
                    message_id=_last_assistant_message_id(memory),
                )
                for m in self._middleware:
                    await m.on_agent_end(agent_end)
                yield agent_end
                return

            except Exception as exc:
                error_msg = _format_exception(exc)
                logger.exception("agent.step.error %s", error_msg)
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

                # Run error-path cleanup hook (lightweight, no LLM calls).
                async for evt in self._on_run_error(memory):
                    yield evt

                agent_end = AgentEnd(
                    str(exc),
                    time.time() - start_time,
                    error=error_msg,
                    message_id=_last_assistant_message_id(memory),
                )
                for m in self._middleware:
                    await m.on_agent_end(agent_end)
                yield agent_end
                return

            agent_end = AgentEnd(
                response_text,
                time.time() - start_time,
                exceeded=exceeded_max_iterations,
                message_id=_last_assistant_message_id(memory),
            )
            for m in self._middleware:
                await m.on_agent_end(agent_end)
            yield agent_end

        return agen()

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
            name = tc.get("function", {}).get("name", "")
            if not name:
                # 截断的流式调用（名称 chunk 未到达）：不执行、不报错、
                # 不进入统计，否则会以 "unknown" 名记进 metrics（issue #62）。
                continue
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
            result = None
            progress_chunks: list[str] = []
            try:
                # Parse inside the try: a truncated ``arguments`` string
                # from a broken/stopped stream must surface as a tool
                # error (ToolEnd + sentinel), not crash the task and
                # leave the loop waiting forever for a sentinel that
                # never arrives.
                args = json.loads(raw_args) if raw_args else {}

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
                        # 持久化的进度只保留尾部 MAX 条：长输出命令（大目录遍历等）
                        # 会产出海量进度事件，全量落盘会让会话文件与 LLM 上下文膨胀。
                        if len(progress_chunks) < MAX_PERSISTED_PROGRESS_CHUNKS:
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

            should_stop = False
            stop_response_text: str | None = None
            for tc, result in exec_results:
                if isinstance(result, Exception):
                    content = f"[Error] {result}"
                    result_meta = None
                    result_stop = False
                elif isinstance(result, ToolResult):
                    content = _serialize_content_for_llm(result.content)
                    result_meta = result.meta
                    result_stop = result.stop
                    if result.stop:
                        should_stop = True
                        if stop_response_text is None:
                            stop_response_text = str(result.content)
                else:
                    content = _serialize_content_for_llm(result)
                    result_meta = None
                    result_stop = False
                tool_msg: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": content,
                }
                if result_meta:
                    tool_msg["meta"] = result_meta
                if result_stop:
                    tool_msg["stop"] = result_stop
                tc_progress = progress_data.get(tc["id"])
                if tc_progress:
                    tool_msg["progress"] = tc_progress
                await memory.add_message(tool_msg)  # type: ignore[arg-type]
                if self._emit_message_events:
                    yield MessageEvent(message=tool_msg)
        except (Exception, asyncio.CancelledError) as exc:
            yield ExecutionEnd(exec_results, error=f"{type(exc).__name__}: {exc}")
            raise

        yield ExecutionEnd(
            exec_results, should_stop=should_stop, response_text=stop_response_text
        )


__all__ = ["BaseAgent", "Message"]
