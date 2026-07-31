from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, AsyncIterator, Sequence
from unittest import mock

import pytest

from minimal_harness.agent.controller import (
    DefaultController,
    GoalController,
    TimerController,
    _format_duration,
    _parse_duration,
)
from minimal_harness.agent.runtime import ControllerRegistry
from minimal_harness.llm.llm import LLMResponse, Stream
from minimal_harness.memory import (
    ConversationMemory,
    Message,
    TextContentPart,
    user_message,
)
from minimal_harness.types import (
    AgentEnd,
    AgentEvent,
    AgentStart,
    ControllerContinue,
    ControllerEnd,
    ControllerEvent,
    ControllerStart,
    LLMChunkDelta,
)

# ── Fakes ─────────────────────────────────────────────────────────────────


async def _stream_of(content: str | None) -> AsyncIterator[LLMChunkDelta | LLMResponse]:
    if content:
        yield LLMChunkDelta(content=content)
    yield LLMResponse(
        content=content,
        reasoning_content=None,
        tool_calls=[],
        finish_reason=None,
    )


class FakeLLMProvider:
    """可编程模拟 LLM：chat() 依次返回预设内容列表。"""

    def __init__(self, responses: list[str | None]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Any = None,
        stop_event: asyncio.Event | None = None,
        **kwargs: Any,
    ) -> Stream[LLMChunkDelta]:
        self.calls.append(
            {"messages": messages, "tools": tools, "stop_event": stop_event}
        )
        content = self.responses.pop(0) if self.responses else "DONE"
        return Stream[LLMChunkDelta](_stream_of(content))


class RaisingLLMProvider:
    """chat() 永远抛异常——验证 judge 失败的安全默认行为。"""

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Any = None,
        stop_event: asyncio.Event | None = None,
        **kwargs: Any,
    ) -> Stream[LLMChunkDelta]:
        raise RuntimeError("provider down")


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


class _FakeClock:
    """可控时钟，替换 ``controller.time.time`` 以精确控制 elapsed。

    ``advance`` 是每次调用递增的秒数——execute 里第一调用记 start_time，
    后续调用读取 elapsed，模拟真实流逝。
    """

    def __init__(self, start: float = 1000.0, advance: float = 0.0) -> None:
        self.now = start
        self.advance = advance

    def __call__(self) -> float:
        now = self.now
        self.now += self.advance
        return now


def _memory() -> ConversationMemory:
    mem = ConversationMemory()
    return mem


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


async def _collect(controller: Any, agent: _FakeAgent, context: dict | None = None):
    mem = _memory()
    events: list[AgentEvent | ControllerEvent] = []
    async for event in controller.execute(
        agent=agent,
        user_input=_input(),
        stop_event=None,
        memory=mem,
        tools=[],
        context=context,
    ):
        events.append(event)
    return events


# ── DefaultController ─────────────────────────────────────────────────────


class TestDefaultController:
    async def test_passthrough_yields_agent_events(self):
        """AgentStart/AgentEnd 被吞掉，其余事件透传。"""
        agent = _FakeAgent(
            [
                AgentStart(user_input=_input()),
                SimpleNamespace(),  # 任意 AgentEvent 形状的事件
                _agent_end(response="hello"),
            ]
        )
        events = await _collect(DefaultController(), agent)

        types = [type(e).__name__ for e in events]
        assert types == ["ControllerStart", "SimpleNamespace", "ControllerEnd"]

        assert isinstance(events[0], ControllerStart)
        assert events[0].controller_type == "default"

        end = events[-1]
        assert isinstance(end, ControllerEnd)
        assert end.response == "hello"
        assert end.error is None
        assert end.interrupted is False

    async def test_agent_start_and_end_suppressed(self):
        """DefaultController 不对外发 AgentStart/AgentEnd。"""
        agent = _FakeAgent([AgentStart(user_input=_input()), _agent_end()])
        events = await _collect(DefaultController(), agent)
        assert not any(isinstance(e, (AgentStart, AgentEnd)) for e in events)

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


# ── GoalController ────────────────────────────────────────────────────────


class TestGoalController:
    async def test_single_round_judge_says_done(self):
        agent = _FakeAgent([_agent_end(response="answer")])
        controller = GoalController(FakeLLMProvider(["DONE"]))
        events = await _collect(controller, agent)

        assert isinstance(events[0], ControllerStart)
        assert events[0].controller_type == "goal"
        assert len(events) == 2  # Start + End，无 Continue
        end = events[-1]
        assert isinstance(end, ControllerEnd)
        assert end.response == "answer"
        assert end.exceeded is False
        assert end.error is None

    async def test_two_rounds_judge_next_then_done(self):
        agent = _FakeAgent([_agent_end(response="part1"), _agent_end(response="part2")])
        controller = GoalController(FakeLLMProvider(["NEXT: do more", "DONE"]))
        events = await _collect(controller, agent)

        continues = [e for e in events if isinstance(e, ControllerContinue)]
        assert len(continues) == 1
        assert continues[0].next_prompt == "do more"
        assert continues[0].meta == {"round": 1, "max_rounds": 5}

        # 第二轮 agent 收到的输入是 judge 的 next_prompt
        assert agent.run_inputs[1] == [{"type": "text", "text": "do more"}]

        end = events[-1]
        assert isinstance(end, ControllerEnd)
        assert end.response == "part2"

    async def test_max_rounds_exceeded(self):
        agent = _FakeAgent([_agent_end(response="r1"), _agent_end(response="r2")])
        controller = GoalController(FakeLLMProvider(["NEXT: a", "NEXT: b"]))
        events = await _collect(
            controller, agent, context={"controller_config": {"max_goal_rounds": 2}}
        )

        continues = [e for e in events if isinstance(e, ControllerContinue)]
        assert len(continues) == 2
        end = events[-1]
        assert isinstance(end, ControllerEnd)
        assert end.exceeded is True
        assert end.response == "r2"

    async def test_agent_interrupted_stops_immediately(self):
        agent = _FakeAgent([_agent_end(response="partial", interrupted=True)])
        controller = GoalController(FakeLLMProvider(["NEXT: continue"]))
        events = await _collect(controller, agent)

        end = events[-1]
        assert isinstance(end, ControllerEnd)
        assert end.interrupted is True
        # judge 不该被调用
        assert controller._llm_provider.calls == []  # type: ignore[attr-defined]

    async def test_agent_error_stops_immediately(self):
        agent = _FakeAgent([_agent_end(response="", error="tool failed")])
        controller = GoalController(FakeLLMProvider(["NEXT: continue"]))
        events = await _collect(controller, agent)

        end = events[-1]
        assert isinstance(end, ControllerEnd)
        assert end.error == "tool failed"

    async def test_judge_error_defaults_to_stop(self):
        agent = _FakeAgent([_agent_end(response="answer")])
        controller = GoalController(RaisingLLMProvider())
        events = await _collect(controller, agent)

        # 安全默认：judge 异常 → DONE（不继续，不报错）
        assert len(events) == 2
        end = events[-1]
        assert isinstance(end, ControllerEnd)
        assert end.error is None
        assert end.response == "answer"

    async def test_judge_call_receives_stop_event(self):
        """验证 _call_judge 把 stop_event 传给了 llm_provider.chat()。"""
        agent = _FakeAgent([_agent_end()])
        provider = FakeLLMProvider(["DONE"])
        controller = GoalController(provider)
        stop_event = asyncio.Event()

        mem = _memory()
        async for _ in controller.execute(
            agent=agent,
            user_input=_input(),
            stop_event=stop_event,
            memory=mem,
            tools=[],
        ):
            pass

        assert len(provider.calls) == 1
        assert provider.calls[0]["stop_event"] is stop_event

    async def test_judge_parse_done_case_variants(self):
        c = GoalController(FakeLLMProvider([]))
        for content in ["DONE", "done", "Done", "DONE.", "DONE 全部完成", "done "]:
            assert c._parse_judge_response(content) is None, content

    async def test_judge_parse_next_format_variants(self):
        c = GoalController(FakeLLMProvider([]))
        assert c._parse_judge_response("NEXT: do it") == "do it"
        assert c._parse_judge_response("Next: do it") == "do it"
        assert c._parse_judge_response("NEXT：做吧") == "做吧"
        assert c._parse_judge_response("NEXT - do it") == "do it"
        assert c._parse_judge_response("NEXT") is None
        assert c._parse_judge_response("") is None
        assert c._parse_judge_response("whatever") is None

    async def test_judge_receives_conversation_messages(self):
        agent = _FakeAgent([_agent_end()])
        provider = FakeLLMProvider(["DONE"])
        controller = GoalController(provider)

        mem = _memory()
        await mem.add_message(user_message([{"type": "text", "text": "original goal"}]))
        async for _ in controller.execute(
            agent=agent,
            user_input=_input(),
            stop_event=None,
            memory=mem,
            tools=[],
        ):
            pass

        msgs = provider.calls[0]["messages"]
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user", "user"]
        assert msgs[1]["content"][0]["text"] == "original goal"
        assert "Reply DONE or NEXT" in msgs[-1]["content"]


# ── TimerController ───────────────────────────────────────────────────────


class TestTimerController:
    async def _collect_with_clock(self, controller, agent, clock, context=None):
        with mock.patch("minimal_harness.agent.controller.time.time", clock):
            return await _collect(controller, agent, context=context)

    async def test_elapsed_under_duration_continues(self):
        clock = _FakeClock(1000.0)
        agent = _FakeAgent([_agent_end(response="r1"), _agent_end(response="r2")])
        controller = TimerController(
            FakeLLMProvider(["NEXT: keep going"]), default_duration="30m", max_rounds=2
        )
        events = await self._collect_with_clock(controller, agent, clock)

        continues = [e for e in events if isinstance(e, ControllerContinue)]
        assert len(continues) == 2  # 第 1 轮 judge NEXT；第 2 轮 judge DONE→forced
        assert continues[0].next_prompt == "keep going"
        assert continues[0].meta == {
            "elapsed": 0,
            "remaining": 1800,
            "duration": 1800,
        }
        end = events[-1]
        assert isinstance(end, ControllerEnd)
        assert end.exceeded is True  # 时钟不动 → 时间永不耗尽 → round 上限兜底

    async def test_elapsed_exceeds_duration_stops(self):
        clock = _FakeClock(1000.0, advance=6.0)  # 每次调用 +6s
        agent = _FakeAgent([_agent_end(response="r1")])
        controller = TimerController(
            FakeLLMProvider(["NEXT: x"]), default_duration="5s"
        )
        events = await self._collect_with_clock(controller, agent, clock)

        assert len(events) == 2  # Start + End，judge 不调用
        end = events[-1]
        assert isinstance(end, ControllerEnd)
        assert end.exceeded is False
        assert end.response == "r1"
        assert controller._llm_provider.calls == []  # type: ignore[attr-defined]

    async def test_judge_returns_done_but_time_not_up_uses_forced_prompt(self):
        clock = _FakeClock(1000.0)
        agent = _FakeAgent([_agent_end(response="r1")])
        controller = TimerController(
            FakeLLMProvider(["DONE"]), default_duration="30m", max_rounds=1
        )
        events = await self._collect_with_clock(controller, agent, clock)

        continues = [e for e in events if isinstance(e, ControllerContinue)]
        assert len(continues) == 1
        assert "Continue working on the original task" in continues[0].next_prompt
        assert "30m" in continues[0].next_prompt

    async def test_judge_returns_next_with_time_context(self):
        clock = _FakeClock(1000.0)
        agent = _FakeAgent([_agent_end(response="r1")])
        provider = FakeLLMProvider(["NEXT: finish the rest"])
        controller = TimerController(provider, default_duration="30m", max_rounds=1)
        events = await self._collect_with_clock(controller, agent, clock)

        continues = [e for e in events if isinstance(e, ControllerContinue)]
        assert continues[0].next_prompt == "finish the rest"

        # judge 的系统 prompt 带时间上下文
        system_msg = provider.calls[0]["messages"][0]["content"]
        assert "Time context" in system_msg
        assert "30m" in system_msg

    async def test_duration_parsing(self):
        assert _parse_duration("30m") == 1800
        assert _parse_duration("1h") == 3600
        assert _parse_duration("300s") == 300
        assert _parse_duration("1.5h") == 5400
        assert _parse_duration("90") == 90
        assert _parse_duration(90) == 90
        assert _parse_duration("garbage") == 300  # 默认 5 分钟
        assert _parse_duration("") == 300
        assert _format_duration(180) == "3m 0s"
        assert _format_duration(3661) == "1h 1m"
        assert _format_duration(45) == "45s"

    async def test_agent_error_stops_immediately(self):
        clock = _FakeClock(1000.0)
        agent = _FakeAgent([_agent_end(response="", error="boom")])
        controller = TimerController(
            FakeLLMProvider(["NEXT: x"]), default_duration="30m"
        )
        events = await self._collect_with_clock(controller, agent, clock)

        end = events[-1]
        assert isinstance(end, ControllerEnd)
        assert end.error == "boom"

    async def test_max_rounds_safety_cap(self):
        clock = _FakeClock(1000.0)
        agent = _FakeAgent([_agent_end(response="r1")])
        controller = TimerController(
            FakeLLMProvider(["NEXT: a"]), default_duration="30m", max_rounds=1
        )
        events = await self._collect_with_clock(controller, agent, clock)

        end = events[-1]
        assert isinstance(end, ControllerEnd)
        assert end.exceeded is True

    async def test_config_duration_used(self):
        clock = _FakeClock(1000.0)
        agent = _FakeAgent([_agent_end(response="r1")])
        controller = TimerController(FakeLLMProvider([]), default_duration="30m")
        clock.now = 1000.0
        events = await self._collect_with_clock(
            controller,
            agent,
            clock,
            context={"controller_config": {"duration": "10s"}},
        )
        end = events[-1]
        assert isinstance(end, ControllerEnd)
        # 0 < 10s → 继续，但 judge 返回 None（空列表默认 DONE）→ forced prompt
        assert any(isinstance(e, ControllerContinue) for e in events)


# ── ControllerRegistry ────────────────────────────────────────────────────


class TestControllerRegistry:
    async def test_register_and_create(self):
        reg = ControllerRegistry()
        reg.register("goal", lambda llm_provider: GoalController(llm_provider))
        ctrl = reg.create("goal", llm_provider=FakeLLMProvider(["DONE"]))
        assert isinstance(ctrl, GoalController)

    async def test_unknown_type_falls_back_to_default(self):
        reg = ControllerRegistry()
        ctrl = reg.create("nope", llm_provider=None)
        assert isinstance(ctrl, DefaultController)

    async def test_list_types(self):
        reg = ControllerRegistry()
        reg.register("goal", lambda llm_provider: GoalController(llm_provider))
        reg.register("timer", lambda llm_provider: TimerController(llm_provider))
        assert reg.list_types() == ["goal", "timer"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
