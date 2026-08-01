from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from minimal_harness.agent.controller import DefaultController
from minimal_harness.agent.runtime import ControllerRegistry
from minimal_harness.memory import (
    ConversationMemory,
    TextContentPart,
)
from minimal_harness.types import (
    AgentEnd,
    AgentEvent,
    AgentStart,
    ControllerEnd,
    ControllerEvent,
    ControllerStart,
)

# ── Fakes ─────────────────────────────────────────────────────────────────


class _FakeAgent:
    """记录 run 参数并按预设事件序列执行的假 Agent。"""

    def __init__(self, events: list[Any] | None = None) -> None:
        self.events: list[Any] = events or []
        self.run_inputs: list[Any] = []
        self._raise: BaseException | None = None

    def raise_on_run(self, exc: BaseException) -> None:
        self._raise = exc

    async def run(
        self,
        user_input: Any,
        stop_event: asyncio.Event | None = None,
        memory: Any = None,
        tools: Any = None,
        system_prompt: str = "",
        context: Any = None,
        llm_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        self.run_inputs.append(user_input)
        if self._raise is not None:
            raise self._raise
        for event in self.events:
            yield event


def _memory() -> ConversationMemory:
    return ConversationMemory()


def _input(text: str = "hi") -> list[TextContentPart]:
    return [TextContentPart(type="text", text=text)]


def _agent_end(
    response: str = "done",
    *,
    interrupted: bool = False,
    error: str | None = None,
) -> AgentEnd:
    return AgentEnd(
        response=response,
        time_taken=0.5,
        interrupted=interrupted,
        error=error,
    )


async def _collect(
    controller: Any,
    agent: _FakeAgent,
    controller_config: dict | None = None,
):
    mem = _memory()
    events: list[AgentEvent | ControllerEvent] = []
    async for event in controller.execute(
        agent=agent,
        user_input=_input(),
        stop_event=None,
        memory=mem,
        tools=[],
        controller_config=controller_config,
    ):
        events.append(event)
    return events


# ── DefaultController ─────────────────────────────────────────────────────


class TestDefaultController:
    async def test_passthrough_yields_agent_events(self):
        """AgentStart/AgentEnd 原样透传，其余事件也透传。"""
        agent = _FakeAgent(
            [
                AgentStart(user_input=_input()),
                SimpleNamespace(),  # 任意 AgentEvent 形状的事件
                _agent_end(response="hello"),
            ]
        )
        events = await _collect(DefaultController(), agent)

        types = [type(e).__name__ for e in events]
        assert types == [
            "ControllerStart",
            "AgentStart",
            "SimpleNamespace",
            "AgentEnd",
            "ControllerEnd",
        ]

        assert isinstance(events[0], ControllerStart)
        assert events[0].controller_type == "default"

        end = events[-1]
        assert isinstance(end, ControllerEnd)
        assert end.response == "hello"
        assert end.error is None
        assert end.interrupted is False

    async def test_agent_start_and_end_passed_through(self):
        """DefaultController 不吞事件：AgentStart/AgentEnd 原样透传。"""
        agent = _FakeAgent([AgentStart(user_input=_input()), _agent_end()])
        events = await _collect(DefaultController(), agent)
        assert any(isinstance(e, AgentStart) for e in events)
        assert any(isinstance(e, AgentEnd) for e in events)

    async def test_agent_exception_wrapped_in_controller_end(self):
        agent = _FakeAgent()
        agent.raise_on_run(RuntimeError("boom"))
        events = await _collect(DefaultController(), agent)

        end = events[-1]
        assert isinstance(end, ControllerEnd)
        assert end.error == "RuntimeError: boom"
        assert end.response == ""

    async def test_cancelled_error_sets_interrupted(self):
        agent = _FakeAgent()
        agent.raise_on_run(asyncio.CancelledError())
        events = await _collect(DefaultController(), agent)

        end = events[-1]
        assert isinstance(end, ControllerEnd)
        assert end.interrupted is True
        assert end.error == "Controller execution cancelled"


# ── ControllerRegistry ────────────────────────────────────────────────────


class TestControllerRegistry:
    async def test_register_and_create(self):
        reg = ControllerRegistry()
        reg.register("custom", lambda llm_provider: DefaultController())
        ctrl = reg.create("custom", llm_provider=None)
        assert isinstance(ctrl, DefaultController)

    async def test_unknown_type_falls_back_to_default(self):
        reg = ControllerRegistry()
        ctrl = reg.create("nope", llm_provider=None)
        assert isinstance(ctrl, DefaultController)

    async def test_list_types(self):
        reg = ControllerRegistry()
        reg.register("a", lambda llm_provider: DefaultController())
        reg.register(
            "b",
            lambda llm_provider: DefaultController(),
            metadata={"display_name": "B", "settings": []},
        )
        assert reg.list_types() == ["a", "b"]
        assert reg.catalog() == [
            {"value": "a"},
            {"value": "b", "display_name": "B", "settings": []},
        ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
