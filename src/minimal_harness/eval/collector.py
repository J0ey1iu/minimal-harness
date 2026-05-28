from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from minimal_harness.agent.middleware import Middleware

if TYPE_CHECKING:
    from minimal_harness.eval.types import EvalRunRecord
    from minimal_harness.types import AgentEnd, LLMEnd, ToolCall

from .persistence import EvalPersistence
from .types import EvalEventRecord, TokenUsageRecord


class EvalCollector(Middleware):
    def __init__(self, run_id: str, persistence: EvalPersistence) -> None:
        super().__init__()
        self._run_id = run_id
        self._persistence = persistence
        self._events: list[EvalEventRecord] = []
        self.llm_call_count: int = 0
        self.tool_call_count: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_tokens: int = 0

    def _record(self, event_type: str, data: dict[str, Any]) -> None:
        record = EvalEventRecord(
            event_type=event_type,
            timestamp=time.time(),
            data=_safe(data),
        )
        self._events.append(record)
        self._persistence.write_event(self._run_id, record)

    @property
    def events(self) -> list[EvalEventRecord]:
        return list(self._events)

    @property
    def token_usage(self) -> TokenUsageRecord:
        return TokenUsageRecord(
            input_tokens=self.total_input_tokens,
            output_tokens=self.total_output_tokens,
            total_tokens=self.total_tokens,
        )

    def consume_event(self, event: Any) -> None:
        from minimal_harness.types import (
            AgentEnd as _AgentEnd,
            AgentStart as _AgentStart,
            LLMEnd as _LLMEnd,
            LLMStart as _LLMStart,
            ToolEnd as _ToolEnd,
            ToolStart as _ToolStart,
        )

        if isinstance(event, _AgentStart):
            self._record("agent_start", {"user_input": _safe(event.user_input)})
        elif isinstance(event, _AgentEnd):
            self._record(
                "agent_end",
                {
                    "response": event.response,
                    "time_taken": event.time_taken,
                    "exceeded": event.exceeded,
                    "interrupted": event.interrupted,
                },
            )
        elif isinstance(event, _LLMStart):
            self._record(
                "llm_start",
                {"messages": _safe(event.messages), "tools": _safe(event.tools)},
            )
        elif isinstance(event, _LLMEnd):
            self.llm_call_count += 1
            usage = event.usage
            if usage:
                self.total_input_tokens += usage.get("prompt_tokens", 0)
                self.total_output_tokens += usage.get("completion_tokens", 0)
                self.total_tokens += usage.get("total_tokens", 0)
            self._record(
                "llm_end",
                {
                    "content": event.content,
                    "reasoning_content": event.reasoning_content,
                    "tool_calls": event.tool_calls,
                    "usage": dict(usage) if usage else None,
                },
            )
        elif isinstance(event, _ToolStart):
            self.tool_call_count += 1
            self._record("tool_start", {"tool_call": _safe(event.tool_call)})
        elif isinstance(event, _ToolEnd):
            self._record(
                "tool_end",
                {"tool_call": _safe(event.tool_call), "result": _safe(event.result)},
            )

    @staticmethod
    def apply_agent_end(
        run_record: EvalRunRecord,
        event: Any,
    ) -> None:
        from minimal_harness.types import AgentEnd as _AgentEnd

        if isinstance(event, _AgentEnd):
            run_record.response = event.response
            run_record.time_taken = event.time_taken
            run_record.exceeded = event.exceeded
            if event.interrupted:
                run_record.status = "interrupted"
            elif event.error:
                run_record.status = "failed"
                run_record.error = event.error

    async def on_agent_start(self, user_input: Any) -> None:
        self._record("agent_start", {"user_input": _safe(user_input)})

    async def on_agent_end(self, event: AgentEnd) -> None:
        self._record(
            "agent_end",
            {
                "response": event.response,
                "time_taken": event.time_taken,
                "exceeded": event.exceeded,
                "interrupted": event.interrupted,
            },
        )

    async def on_llm_start(self, messages: list[dict[str, Any]], tools: Any) -> None:
        self._record(
            "llm_start",
            {"messages": _safe(messages), "tools": _safe(tools)},
        )

    async def on_llm_end(self, event: LLMEnd) -> None:
        self.llm_call_count += 1
        usage = event.usage
        if usage:
            self.total_input_tokens += usage.get("prompt_tokens", 0)
            self.total_output_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)
        self._record(
            "llm_end",
            {
                "content": event.content,
                "reasoning_content": event.reasoning_content,
                "tool_calls": event.tool_calls,
                "usage": dict(usage) if usage else None,
            },
        )

    async def on_tool_start(self, tool_call: ToolCall) -> None:
        self.tool_call_count += 1
        self._record("tool_start", {"tool_call": _safe(tool_call)})

    async def on_tool_end(self, tool_call: ToolCall, result: Any) -> None:
        self._record(
            "tool_end",
            {"tool_call": _safe(tool_call), "result": _safe(result)},
        )

    async def on_tool_error(self, tool_call: ToolCall, error: Exception) -> None:
        self._record(
            "tool_error",
            {"tool_call": _safe(tool_call), "error": str(error)},
        )

    async def on_error(self, error: BaseException) -> None:
        self._record("error", {"error": str(error)})


def _safe(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, dict):
        return {k: _safe(v) for k, v in v.items()}
    if isinstance(v, (list, tuple)):
        return [_safe(i) for i in v]
    if isinstance(v, Exception):
        return f"{type(v).__name__}: {v}"
    if hasattr(v, "__dict__"):
        return _safe(v.__dict__)
    return str(v)
