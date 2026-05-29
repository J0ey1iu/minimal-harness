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
                    msg_count = len(event.messages)
                    last_user_msg = ""
                    for m in reversed(event.messages):
                        if isinstance(m, dict) and m.get("role") == "user":
                            c = m.get("content", "")
                            if isinstance(c, str):
                                last_user_msg = c[:300]
                            break
                    text = f"{target_agent_name} thinking ({msg_count} messages)..."
                    if last_user_msg:
                        text += f"\n  └─ {last_user_msg}"
                    yield {
                        "status": "progress",
                        "type": "llm_start",
                        "message": text,
                    }
                elif isinstance(event, LLMEnd):
                    if event.content:
                        result_text = str(event.content)
                    parts = []
                    if event.reasoning_content:
                        parts.append(f"Reasoning:\n{event.reasoning_content}")
                    if event.content:
                        parts.append(f"Response:\n{event.content}")
                    if event.tool_calls:
                        tc_lines = "\n".join(
                            f"  • {tc['function']['name']}({tc['function']['arguments']})"
                            for tc in event.tool_calls
                        )
                        parts.append(f"Tool calls:\n{tc_lines}")
                    yield {
                        "status": "progress",
                        "type": "llm_end",
                        "message": "\n\n".join(parts) if parts else "LLM finished",
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
                        r = str(result) if result is not None else ""
                        parts.append(f"  • {name} => {r}")
                    yield {
                        "status": "progress",
                        "type": "execution_end",
                        "message": "\n".join(parts)
                        if parts
                        else "Tool execution complete",
                    }
                elif isinstance(event, ToolStart):
                    name = event.tool_call["function"]["name"]
                    args = event.tool_call["function"]["arguments"]
                    yield {
                        "status": "progress",
                        "type": "tool_start",
                        "message": f"Tool: {name}({args})",
                    }
                elif isinstance(event, ToolEnd):
                    name = event.tool_call["function"]["name"]
                    result_str = str(event.result) if event.result is not None else ""
                    yield {
                        "status": "progress",
                        "type": "tool_end",
                        "message": f"Tool {name} => {result_str}",
                    }
                elif isinstance(event, AgentEnd):
                    result_text = event.response or result_text
                    parts = [f"{target_agent_name} completed"]
                    if event.response:
                        parts.append(f"Response: {event.response}")
                    if event.time_taken is not None:
                        parts.append(f"Time: {event.time_taken:.1f}s")
                    if event.error:
                        parts.append(f"Error: {event.error}")
                    yield {
                        "status": "progress",
                        "type": "agent_end",
                        "message": "\n".join(parts),
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


async def register_runtime_tools(
    agent_registry: AgentRegistryProtocol,
    session_store: Any,
    tool_registry: Any,
    run_fn: Any,
) -> None:
    from minimal_harness.types import LocalToolBinding, ToolMetadata

    if await tool_registry.get("handoff") is None:
        tool = make_handoff_tool(
            agent_registry=agent_registry,
            session_store=session_store,
            run_fn=run_fn,
            delegating_agent_id=None,
        )
        await tool_registry.register(
            ToolMetadata(
                name=tool.name,
                display_name=tool.display_name,
                description=tool.description,
                parameters=tool.parameters,
                metadata_id=tool.name,
                display_name_locale=tool.display_name_locale,
                description_locale=tool.description_locale,
                binding=LocalToolBinding(fn=getattr(tool, "fn", None)),
            )
        )
    if await tool_registry.get("discover_agents") is None:
        tool = make_discover_agents_tool(agent_registry=agent_registry)
        await tool_registry.register(
            ToolMetadata(
                name=tool.name,
                display_name=tool.display_name,
                description=tool.description,
                parameters=tool.parameters,
                metadata_id=tool.name,
                display_name_locale=tool.display_name_locale,
                description_locale=tool.description_locale,
                binding=LocalToolBinding(fn=getattr(tool, "fn", None)),
            )
        )
