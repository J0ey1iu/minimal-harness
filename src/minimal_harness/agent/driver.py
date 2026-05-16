from __future__ import annotations

import asyncio
import json
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Iterable,
    Protocol,
    Sequence,
    runtime_checkable,
)

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
    RemoteAgentBinding,
    ToolEnd,
    ToolProgress,
    ToolStart,
)

if TYPE_CHECKING:
    from minimal_harness.memory import ExtendedInputContentPart, Memory
    from minimal_harness.tool.base import Tool

if TYPE_CHECKING:
    import httpx
else:
    try:
        import httpx
    except ImportError:  # pragma: no cover
        httpx = None  # type: ignore[assignment]


@runtime_checkable
class RemoteAgentDriver(Protocol):
    """Protocol for executing an agent remotely.

    Users implement this protocol to bridge framework-internal
    ``Agent.run()`` calls to an external agent service over any
    transport protocol.
    """

    def run(
        self,
        user_input: Iterable[ExtendedInputContentPart],
        stop_event: asyncio.Event | None,
        memory: Memory,
        tools: Sequence[Tool],
        system_prompt: str,
        context: dict[str, Any] | None,
        llm_kwargs: dict[str, Any] | None,
    ) -> AsyncIterator[AgentEvent]: ...


class RemoteAgentDriverFactory(Protocol):
    """Factory that creates a ``RemoteAgentDriver`` from a binding."""

    def create(self, binding: RemoteAgentBinding) -> RemoteAgentDriver: ...


class DefaultAgentDriverFactory:
    """Default factory: returns ``SSEAgentDriver`` for any remote binding."""

    def create(self, binding: RemoteAgentBinding) -> RemoteAgentDriver:
        return SSEAgentDriver(binding)


class SSEAgentDriver:
    """Default remote-agent driver based on SSE over HTTP.

    Delegates the full agent loop to a remote service that speaks the
    same ``AgentEvent`` protocol.
    """

    def __init__(self, binding: RemoteAgentBinding) -> None:
        if httpx is None:
            raise ImportError(
                "httpx is required for SSEAgentDriver. "
                "Install it via `pip install httpx`."
            )
        self._url = binding.url
        self._headers = dict(binding.headers)
        self._timeout = binding.timeout

    async def run(
        self,
        user_input: Iterable[ExtendedInputContentPart],
        stop_event: asyncio.Event | None,
        memory: Memory,
        tools: Sequence[Tool],
        system_prompt: str,
        context: dict[str, Any] | None,
        llm_kwargs: dict[str, Any] | None,
    ) -> AsyncIterator[AgentEvent]:
        payload = {
            "user_input": list(user_input),
            "system_prompt": system_prompt,
            "tools": [t.to_schema() for t in tools],
            "context": context or {},
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST",
                self._url,
                json=payload,
                headers=self._headers,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if stop_event and stop_event.is_set():
                        break
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    payload = json.loads(line[6:])
                    event_type = payload.get("type") or payload.get("event", "")
                    data = payload.get("data") or payload

                    event = self._deserialize_event(event_type, data)
                    if event is not None:
                        yield event

    @staticmethod
    def _deserialize_event(event_type: str, data: dict[str, Any]) -> AgentEvent | None:
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
                    )
                case "execution_start":
                    return ExecutionStart(tool_calls=data.get("tool_calls", []))
                case "execution_end":
                    return ExecutionEnd(results=data.get("results", []))
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
        except Exception:
            return None
        return None
