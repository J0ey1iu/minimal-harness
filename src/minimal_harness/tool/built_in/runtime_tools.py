"""Runtime tools — handoff and discover_agents.

These tools are created by AgentRuntime and injected into agent runs.
They provide multi-agent handoff and agent discovery capabilities.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable

from minimal_harness.agent.runtime import _current_context
from minimal_harness.tool.base import StreamingTool
from minimal_harness.types import (
    AgentEnd,
    AgentEvent,
    ExecutionEnd,
    ExecutionStart,
    LLMEnd,
    LLMStart,
    ToolEnd,
    ToolStart,
)

if TYPE_CHECKING:
    from minimal_harness.agent.registry import AgentRegistryProtocol


def make_handoff_tool(
    agent_registry: AgentRegistryProtocol,
    session_store: Any,
    run_fn: Callable[
        ...,
        Awaitable[tuple[asyncio.Task, asyncio.Event, asyncio.Queue[AgentEvent | None]]],
    ],
    delegating_agent_id: str | None = None,
) -> StreamingTool:
    async def handoff_fn(
        target_agent_name: str, context_summary: str, task_description: str
    ) -> AsyncIterator[Any]:
        metadata = await agent_registry.get(target_agent_name)
        if metadata is None:
            yield {
                "status": "error",
                "message": f"Handoff target '{target_agent_name}' not found",
            }
            return

        combined = f"Context: {context_summary}\n\nTask: {task_description}"
        if delegating_agent_id:
            combined = f"[Delegated by {delegating_agent_id}]{combined}"

        handoff_session_id = uuid.uuid4().hex
        await session_store.create_session(
            session_id=handoff_session_id,
            agent_name=target_agent_name,
            transient=True,
        )

        sub_task = None
        sub_stop_event = None
        try:
            sub_task, sub_stop_event, event_queue = await run_fn(
                user_input=[{"type": "text", "text": combined}],
                agent_metadata_id=metadata.metadata_id,
                memory_id=handoff_session_id,
            )

            yield {
                "status": "handoff_started",
                "message": f"Starting delegated task to {target_agent_name}...",
            }

            result_text = ""
            while True:
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if sub_stop_event.is_set():
                        yield {
                            "status": "error",
                            "message": "Delegated task was interrupted",
                        }
                        break
                    continue

                if event is None:
                    break

                if isinstance(event, LLMStart):
                    yield {
                        "status": "progress",
                        "type": "llm_start",
                        "message": "LLM generating...",
                    }
                elif isinstance(event, LLMEnd):
                    if event.content:
                        result_text = str(event.content)
                    yield {
                        "status": "progress",
                        "type": "llm_end",
                        "message": (event.content or "LLM response generated")[:200],
                    }
                elif isinstance(event, ExecutionStart):
                    names = ", ".join(tc["function"]["name"] for tc in event.tool_calls)
                    yield {
                        "status": "progress",
                        "type": "execution_start",
                        "message": f"Executing: {names}",
                    }
                elif isinstance(event, ExecutionEnd):
                    parts = []
                    for tc, result in event.results:
                        name = tc["function"]["name"]
                        r = (str(result) if result is not None else "")[:200]
                        parts.append(f"{name} => {r}")
                    yield {
                        "status": "progress",
                        "type": "execution_end",
                        "message": " | ".join(parts)
                        if parts
                        else "Tool execution complete",
                    }
                elif isinstance(event, ToolStart):
                    name = event.tool_call["function"]["name"]
                    yield {
                        "status": "progress",
                        "type": "tool_start",
                        "message": f"Tool started: {name}",
                    }
                elif isinstance(event, ToolEnd):
                    name = event.tool_call["function"]["name"]
                    result_str = (
                        str(event.result) if event.result is not None else ""
                    )[:200]
                    yield {
                        "status": "progress",
                        "type": "tool_end",
                        "message": f"Tool {name} completed: {result_str}",
                    }
                elif isinstance(event, AgentEnd):
                    result_text = event.response or result_text
                    yield {
                        "status": "progress",
                        "type": "agent_end",
                        "message": (event.response or "Agent completed")[:200],
                    }

            yield {
                "status": "handoff_complete",
                "message": "Delegated task completed",
                "result": result_text,
            }
        finally:
            if sub_stop_event is not None:
                sub_stop_event.set()
            if sub_task is not None:
                sub_task.cancel()
                try:
                    await sub_task
                except (asyncio.CancelledError, Exception):
                    pass
            await session_store.delete_session(handoff_session_id)

    return StreamingTool(
        name="handoff",
        display_name="Handoff",
        display_name_locale={"zh": "任务移交", "en": "Handoff"},
        description_locale={
            "zh": "将任务移交给其他智能体。先使用 discover_agents 查找可用智能体。",
            "en": "Hand off a task to another agent. Use discover_agents first to find available agents.",
        },
        description="Hand off a task to another agent. Use discover_agents first to find available agents.",
        parameters={
            "type": "object",
            "properties": {
                "target_agent_name": {
                    "type": "string",
                    "description": "The name of the target agent to hand off to.",
                },
                "context_summary": {
                    "type": "string",
                    "description": "Summary of the current context and conversation state.",
                },
                "task_description": {
                    "type": "string",
                    "description": "Description of the task to hand off to the next agent.",
                },
            },
            "required": [
                "target_agent_name",
                "context_summary",
                "task_description",
            ],
        },
        fn=handoff_fn,
    )


def make_discover_agents_tool(
    agent_registry: AgentRegistryProtocol,
) -> StreamingTool:
    async def discover_fn() -> AsyncIterator[Any]:
        ctx = _current_context.get()
        exclude = ctx.get("agent_name") if ctx else None
        locale = ctx.get("locale", "")
        all_agents = await agent_registry.get_all(exclude=exclude)
        agents_list = [
            {
                "name": m.name,
                "display_name": m.resolve_display_name(locale),
                "description": m.resolve_description(locale),
            }
            for m in all_agents
        ]
        yield {
            "status": "ok",
            "agents": agents_list,
        }

    return StreamingTool(
        name="discover_agents",
        display_name="Discover Agents",
        display_name_locale={"zh": "发现智能体", "en": "Discover Agents"},
        description="Discover available agents that can accept handoffs.",
        description_locale={
            "zh": "发现可以接受任务移交的可用智能体。",
            "en": "Discover available agents that can accept handoffs.",
        },
        parameters={
            "type": "object",
            "properties": {},
        },
        fn=discover_fn,
    )
