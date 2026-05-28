from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, AsyncIterator, Sequence

from minimal_harness.agent.runtime import _current_context
from minimal_harness.agent.simple import SimpleAgent
from minimal_harness.llm.llm import LLMResponse, Stream
from minimal_harness.memory import ConversationMemory, Message
from minimal_harness.sse_serialization import serialize_event
from minimal_harness.tool.base import Tool
from minimal_harness.tool.remote import make_remote_tool
from minimal_harness.types import (
    AgentEnd,
    LLMChunkDelta,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
)

logger = logging.getLogger(__name__)


class _RawClientProvider:
    """Wrap an OpenAI-compatible raw client (with ``.chat.completions.create``)
    into an ``LLMProvider`` so that ``SimpleAgent`` can use it."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool],
        stop_event: asyncio.Event | None = None,
        **kwargs: Any,
    ) -> Stream[LLMChunkDelta]:
        tool_schemas: list[dict] | None = (
            [t.to_schema() for t in tools] if tools else None
        )
        model = kwargs.get("model", "deepseek-v4-flash")
        logger.info(
            "llm.chat model=%s msgs=%d tools=%d",
            model,
            len(messages),
            len(tools),
        )

        raw_stream = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tool_schemas or None,
            stream=True,
        )

        async def _gen() -> AsyncIterator[LLMChunkDelta | LLMResponse]:
            content_parts: list[str] = []
            reasoning: str | None = None
            tool_calls_acc: dict[int, dict[str, Any]] = {}
            usage: TokenUsage | None = None

            async for chunk in raw_stream:
                if stop_event and stop_event.is_set():
                    break
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue

                reasoning_str: str | None = None
                rc = getattr(delta, "reasoning_content", None)
                if isinstance(rc, str):
                    reasoning_str = rc
                else:
                    r = getattr(delta, "reasoning", None)
                    if isinstance(r, str):
                        reasoning_str = r

                if reasoning_str:
                    reasoning = (reasoning or "") + reasoning_str

                content: str | None = None
                if delta.content:
                    content = delta.content
                    content_parts.append(delta.content)

                tc_list: list[ToolCallDelta] | None = None
                if delta.tool_calls:
                    tc_list = []
                    for tc in delta.tool_calls:
                        fn = tc.function
                        tc_list.append(
                            ToolCallDelta(
                                index=tc.index,
                                id=tc.id,
                                name=fn.name if fn else None,
                                arguments=fn.arguments if fn else None,
                            )
                        )
                        idx = tc.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": tc.id or "",
                                "name": fn.name if fn else "",
                                "arguments": "",
                            }
                        if fn and fn.arguments:
                            tool_calls_acc[idx]["arguments"] += fn.arguments

                if content or reasoning_str or tc_list:
                    yield LLMChunkDelta(
                        content=content,
                        reasoning=reasoning_str,
                        tool_calls=tc_list,
                    )

                if hasattr(chunk, "usage") and chunk.usage:
                    u = chunk.usage
                    usage = {
                        "prompt_tokens": u.prompt_tokens,
                        "completion_tokens": u.completion_tokens,
                        "total_tokens": u.total_tokens,
                    }

            final_tool_calls: list[ToolCall] = [
                ToolCall(
                    id=tc["id"],
                    type="function",
                    function={"name": tc["name"], "arguments": tc["arguments"]},
                )
                for tc in tool_calls_acc.values()
            ]

            yield LLMResponse(
                content="".join(content_parts) if content_parts else None,
                reasoning_content=reasoning,
                tool_calls=final_tool_calls,
                finish_reason=None,
                usage=usage,
            )

        return Stream(_gen())


class SSEAgentRunner:
    """Shared agent runner that wraps ``SimpleAgent`` into an SSE event stream.

    Emits the full ``AgentEvent`` protocol (including ``MessageEvent``) so that
    downstream orchestration can collect conversation messages without
    reverse-engineering them from behavioral events.

    Each tool in *tools_schema* is expected to carry an ``endpoint_url`` field
    so the runner can execute it directly via HTTP/SSE.

    Usage::

        runner = SSEAgentRunner(llm_client=my_client)
        async for line in runner.run(user_input, tools_schema, memory, system_prompt, config):
            yield line
    """

    def __init__(
        self,
        llm_client: Any,
        max_iterations: int = 10,
    ) -> None:
        self._llm_client = llm_client
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
        correlation_id = uuid.uuid4().hex[:12]

        memory = ConversationMemory()
        for msg in memory_messages:
            await memory.add_message(msg)  # type: ignore[arg-type]

        tools: list[Tool] = [make_remote_tool(s) for s in (tools_schema or [])]

        provider = _RawClientProvider(self._llm_client)
        agent = SimpleAgent(
            llm_provider=provider,  # type: ignore[arg-type]
            max_iterations=self._max_iterations,
            emit_message_events=True,
        )

        run_context = {"correlation_id": correlation_id}
        ctxtoken = _current_context.set(run_context)
        try:
            async for event in agent.run(
                user_input=user_input,  # type: ignore[arg-type]
                memory=memory,
                tools=tools,
                system_prompt=system_prompt,
                context=run_context,
                llm_kwargs={"model": config.get("model", "deepseek-v4-flash")},
            ):
                yield serialize_event(event)
            elapsed = time.time() - start_time
            logger.info("agent.run.end duration=%.2fs", elapsed)
        except asyncio.CancelledError:
            elapsed = time.time() - start_time
            yield serialize_event(
                AgentEnd(
                    response="",
                    time_taken=elapsed,
                    interrupted=True,
                )
            )
        except Exception as e:
            elapsed = time.time() - start_time
            logger.exception(
                "agent.run.error duration=%.2fs error=%s",
                elapsed,
                f"{type(e).__name__}: {e}",
            )
            yield serialize_event(
                AgentEnd(
                    response="",
                    time_taken=elapsed,
                    error=f"{type(e).__name__}: {e}",
                )
            )
        finally:
            _current_context.reset(ctxtoken)
