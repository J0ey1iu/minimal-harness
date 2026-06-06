from __future__ import annotations

import json
from typing import Any

from minimal_harness.types import (
    AgentEnd,
    AgentEvent,
    AgentStart,
    ExecutionEnd,
    ExecutionStart,
    LLMChunk,
    LLMChunkDelta,
    LLMEnd,
    LLMStart,
    MemoryUpdate,
    MessageEvent,
    ToolEnd,
    ToolProgress,
    ToolStart,
)


def serialize_event(event: AgentEvent) -> str:
    """Convert an ``AgentEvent`` to an SSE ``data:`` line.

    Returns ``"data: <json>\\n\\n"`` — the format expected by
    ``SSEAgentDriver`` and consumed by ``deserialize_event``.
    """
    d: dict[str, Any] = {}

    match event:
        case AgentStart():
            d = {"type": "agent_start", "user_input": event.user_input}
        case AgentEnd():
            d = {
                "type": "agent_end",
                "response": event.response,
                "time_taken": event.time_taken,
                "exceeded": event.exceeded,
                "interrupted": event.interrupted,
                "error": event.error,
            }
        case LLMStart():
            d = {"type": "llm_start", "messages": event.messages, "tools": event.tools}
        case LLMChunk():
            chunk_dict: dict[str, Any] = {
                "content": None,
                "reasoning": None,
                "tool_calls": None,
            }
            if event.chunk:
                chunk_dict["content"] = event.chunk.content
                chunk_dict["reasoning"] = event.chunk.reasoning
                chunk_dict["tool_calls"] = event.chunk.tool_calls
            d = {"type": "llm_chunk", "chunk": chunk_dict}
        case LLMEnd():
            d = {
                "type": "llm_end",
                "content": event.content,
                "reasoning_content": event.reasoning_content,
                "tool_calls": event.tool_calls,
                "usage": event.usage,
                "error": event.error,
            }
        case ExecutionStart():
            d = {"type": "execution_start", "tool_calls": event.tool_calls}
        case ExecutionEnd():
            d = {
                "type": "execution_end",
                "results": event.results,
                "error": event.error,
                "should_stop": event.should_stop,
                "response_text": event.response_text,
            }
        case ToolStart():
            d = {"type": "tool_start", "tool_call": event.tool_call}
        case ToolProgress():
            d = {
                "type": "tool_progress",
                "tool_call": event.tool_call,
                "chunk": event.chunk,
            }
        case ToolEnd():
            d = {
                "type": "tool_end",
                "tool_call": event.tool_call,
                "result": event.result,
            }
        case MemoryUpdate():
            d = {"type": "memory_update", "usage": event.usage}
        case MessageEvent():
            d = {"type": "message", "message": event.message}

    return f"data: {json.dumps(d, ensure_ascii=False, default=str)}\n\n"


def deserialize_event(line: str) -> AgentEvent | None:
    """Parse an SSE ``data:`` line into an ``AgentEvent``.

    Returns ``None`` if the line cannot be parsed (malformed JSON,
    unknown event type, or not a ``data:`` line).
    """
    stripped = line.strip()
    if not stripped or not stripped.startswith("data: "):
        return None
    try:
        payload = json.loads(stripped[6:])
    except (json.JSONDecodeError, ValueError):
        return None

    event_type: str = payload.get("type") or payload.get("event", "")
    data: dict[str, Any] = payload.get("data") or payload

    try:
        match event_type:
            case "agent_start":
                return AgentStart(user_input=data.get("user_input", []))
            case "agent_end":
                return AgentEnd(
                    response=data.get("response", ""),
                    time_taken=data.get("time_taken"),
                    exceeded=data.get("exceeded", False),
                    interrupted=data.get("interrupted", False),
                    error=data.get("error"),
                )
            case "llm_start":
                return LLMStart(
                    messages=data.get("messages", []),
                    tools=data.get("tools", []),
                )
            case "llm_chunk":
                chunk_data = data.get("chunk") or data
                delta = LLMChunkDelta(
                    content=chunk_data.get("content"),
                    reasoning=chunk_data.get("reasoning"),
                    tool_calls=chunk_data.get("tool_calls"),
                )
                return LLMChunk(chunk=delta)
            case "llm_end":
                return LLMEnd(
                    content=data.get("content"),
                    reasoning_content=data.get("reasoning_content"),
                    tool_calls=data.get("tool_calls", []),
                    usage=data.get("usage"),
                    error=data.get("error"),
                )
            case "execution_start":
                return ExecutionStart(tool_calls=data.get("tool_calls", []))
            case "execution_end":
                return ExecutionEnd(
                    results=data.get("results", []),
                    error=data.get("error"),
                    should_stop=data.get("should_stop", False),
                    response_text=data.get("response_text"),
                )
            case "tool_start":
                return ToolStart(tool_call=data.get("tool_call", {}))
            case "tool_progress":
                return ToolProgress(
                    tool_call=data.get("tool_call", {}),
                    chunk=data.get("chunk", data),
                )
            case "tool_end":
                return ToolEnd(
                    tool_call=data.get("tool_call", {}),
                    result=data.get("result"),
                )
            case "memory_update":
                return MemoryUpdate(usage=data.get("usage", {}))
            case "message":
                return MessageEvent(message=data.get("message", {}))
    except Exception:
        return None
    return None
