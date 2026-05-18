from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator

import httpx

from minimal_harness.memory import (
    ConversationMemory,
    assistant_message,
    reasoning_message,
    tool_message,
    user_message,
)
from minimal_harness.sse_serialization import serialize_event
from minimal_harness.types import (
    AgentEnd,
    AgentStart,
    ExecutionEnd,
    ExecutionStart,
    LLMChunk,
    LLMChunkDelta,
    LLMEnd,
    LLMStart,
    MessageEvent,
    ToolEnd,
    ToolProgress,
    ToolStart,
)


class SSEAgentRunner:
    """Shared agent runner that wraps an LLM client + tool service into an SSE event stream.

    Emits the full ``AgentEvent`` protocol (including ``MessageEvent``) so that
    downstream orchestration can collect conversation messages without
    reverse-engineering them from behavioral events.

    Usage::

        runner = SSEAgentRunner(llm_client=my_client, tool_service_url="http://...")
        async for line in runner.run(user_input, tools, memory, system_prompt, config):
            yield line
    """

    def __init__(
        self,
        llm_client: Any,
        tool_service_url: str,
        max_iterations: int = 10,
    ) -> None:
        self._llm_client = llm_client
        self._tool_service_url = tool_service_url
        self._max_iterations = max_iterations

    async def run(
        self,
        user_input: list[dict],
        tools_schema: list[dict],
        memory_messages: list[dict],
        system_prompt: str,
        config: dict,
    ) -> AsyncIterator[str]:
        start_time = time.time()
        yield serialize_event(AgentStart(user_input=user_input))  # type: ignore[arg-type]

        memory = ConversationMemory()
        for msg in memory_messages:
            memory.add_message(msg)  # type: ignore[arg-type]

        text_parts = [p.get("text", "") for p in user_input if p.get("type") == "text"]
        user_text = "\n".join(text_parts) or "Hello"
        memory.add_message(user_message([{"type": "text", "text": user_text}]))

        final_response = ""

        try:
            for iteration in range(self._max_iterations):
                llm_messages = memory.get_forward_messages()
                if system_prompt:
                    llm_messages = (
                        [{"role": "system", "content": system_prompt}] + llm_messages  # type: ignore[operator]
                    )

                yield serialize_event(
                    LLMStart(messages=llm_messages, tools=tools_schema)  # type: ignore[arg-type]
                )

                try:
                    stream = await self._llm_client.chat.completions.create(
                        model=config.get("model", "deepseek-v4-flash"),
                        messages=llm_messages,  # type: ignore[arg-type]
                        tools=tools_schema or None,  # type: ignore[arg-type]
                        stream=True,
                    )
                except Exception:
                    elapsed = time.time() - start_time
                    yield serialize_event(
                        AgentEnd(response=final_response, time_taken=elapsed)
                    )
                    return

                content = ""
                reasoning_content = ""
                tool_calls: dict[int, dict] = {}

                async for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta is None:
                        continue

                    reasoning: str | None = None
                    rc = getattr(delta, "reasoning_content", None)
                    if isinstance(rc, str):
                        reasoning = rc
                    else:
                        r = getattr(delta, "reasoning", None)
                        if isinstance(r, str):
                            reasoning = r

                    if reasoning:
                        reasoning_content += reasoning
                    if delta.content:
                        content += delta.content

                    chunk_tool_calls: list[dict] | None = None
                    if delta.tool_calls:
                        chunk_tool_calls = []
                        for tc in delta.tool_calls:
                            fn = tc.function
                            chunk_tool_calls.append(
                                {
                                    "index": tc.index,
                                    "id": tc.id,
                                    "name": fn.name if fn else None,
                                    "arguments": fn.arguments if fn else None,
                                }
                            )
                            idx = tc.index
                            if idx not in tool_calls:
                                tool_calls[idx] = {
                                    "id": tc.id or "",
                                    "name": fn.name if fn else "",
                                    "arguments": "",
                                }
                            if fn and fn.arguments:
                                tool_calls[idx]["arguments"] += fn.arguments

                    if delta.content or reasoning or chunk_tool_calls:
                        yield serialize_event(
                            LLMChunk(
                                chunk=LLMChunkDelta(
                                    content=delta.content,
                                    reasoning=reasoning,
                                    tool_calls=chunk_tool_calls,  # type: ignore[arg-type]
                                )
                            )
                        )

                final_tool_calls = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in tool_calls.values()
                ]

                yield serialize_event(
                    LLMEnd(
                        content=content,
                        reasoning_content=reasoning_content,
                        tool_calls=final_tool_calls,  # type: ignore[arg-type]
                        usage=None,
                    )
                )

                final_response = content

                if reasoning_content:
                    memory.add_message(reasoning_message(reasoning_content))
                    yield serialize_event(
                        MessageEvent(
                            message={
                                "role": "reasoning",
                                "content": reasoning_content,
                            }
                        )
                    )

                memory.add_message(assistant_message(content, final_tool_calls or None))
                yield serialize_event(
                    MessageEvent(
                        message={
                            "role": "assistant",
                            "content": content,
                            "tool_calls": final_tool_calls or None,
                        }
                    )
                )

                if not final_tool_calls:
                    break

                yield serialize_event(ExecutionStart(tool_calls=final_tool_calls))  # type: ignore[arg-type]

                progress_data: dict[str, list[str]] = {}
                try:
                    async with httpx.AsyncClient(
                        base_url=self._tool_service_url, timeout=30, trust_env=False
                    ) as hc:
                        for tc in final_tool_calls:
                            yield serialize_event(ToolStart(tool_call=tc))  # type: ignore[arg-type]
                            try:
                                args = json.loads(tc["function"]["arguments"])
                            except json.JSONDecodeError:
                                args = {}
                            result = ""
                            progress_chunks: list[str] = []
                            tool_name = tc["function"]["name"]
                            try:
                                async with hc.stream(
                                    "POST",
                                    f"/tools/{tool_name}/execute",
                                    json={"args": args, "tool_call_id": tc["id"]},
                                ) as resp:
                                    async for line in resp.aiter_lines():
                                        if line.startswith("data: "):
                                            ev = json.loads(line[6:])
                                            if ev.get("type") == "tool_progress":
                                                chunk = ev.get("content", "")
                                                progress_chunks.append(chunk)
                                                yield serialize_event(
                                                    ToolProgress(
                                                        tool_call=tc,  # type: ignore[arg-type]
                                                        chunk=chunk,
                                                    )
                                                )
                                            elif ev.get("type") == "tool_end":
                                                result = ev.get("result", "")
                            except Exception as e:
                                result = f"Tool execution error: {e}"
                            yield serialize_event(ToolEnd(tool_call=tc, result=result))  # type: ignore[arg-type]
                            progress_data[tc["id"]] = progress_chunks

                            tc_progress = progress_data.get(tc["id"])
                            memory.add_message(
                                tool_message(tc["id"], result, progress=tc_progress)
                            )
                            tool_msg: dict[str, Any] = {
                                "role": "tool",
                                "content": result,
                                "tool_call_id": tc["id"],
                            }
                            if tc_progress:
                                tool_msg["progress"] = tc_progress
                            yield serialize_event(MessageEvent(message=tool_msg))
                except Exception:
                    yield serialize_event(ExecutionEnd(results=final_tool_calls))  # type: ignore[arg-type]
                    elapsed = time.time() - start_time
                    yield serialize_event(
                        AgentEnd(
                            response=final_response,
                            time_taken=elapsed,
                        )
                    )
                    return

                yield serialize_event(ExecutionEnd(results=final_tool_calls))  # type: ignore[arg-type]

        except asyncio.CancelledError:
            pass

        elapsed = time.time() - start_time
        yield serialize_event(
            AgentEnd(
                response=final_response,
                time_taken=elapsed,
            )
        )
