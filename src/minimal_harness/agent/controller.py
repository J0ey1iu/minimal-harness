"""Controller layer: multi-round orchestration wrapping a single Agent run.

架构分层：Agent（单次 LLM Loop）之上新增 Controller 层。Controller 包裹
Agent 做多轮编排，持有自有事件（``ControllerStart / ControllerContinue /
ControllerEnd``，见 ``minimal_harness.types``）与生命周期。Agent 对
Controller 零感知——Controller 只是调用 ``agent.run()`` 并消费其事件流。

实现：
- :class:`DefaultController`——所有 agent 类型的兜底，行为上是透传。
- :class:`_LoopingController`——``goal`` / ``timer`` 循环型 Controller 的
  共享骨架。每轮 ``agent.run()`` 之后调一次 judge LLM，一次调用同时产出
  "是否停止"和"下一轮 prompt"（``_evaluate()``）。
- :class:`GoalController`——judge 判定 DONE 就停，否则用 ``NEXT: …`` 继续。
- :class:`TimerController`——累计运行时间 >= 用户指定时长就停；时间未到时
  judge 说 DONE 也强制继续（模板 prompt 兜底）。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Iterable,
    Protocol,
    Sequence,
)

from minimal_harness.memory import ExtendedInputContentPart, Memory, TextContentPart
from minimal_harness.tool.base import Tool
from minimal_harness.types import (
    AgentEnd,
    AgentEvent,
    AgentStart,
    ControllerContinue,
    ControllerEnd,
    ControllerEvent,
    ControllerStart,
)

if TYPE_CHECKING:
    from minimal_harness.agent.protocol import Agent
    from minimal_harness.llm.llm import LLMProvider

logger = logging.getLogger(__name__)

__all__ = [
    "Controller",
    "DefaultController",
    "GoalController",
    "TimerController",
    "_LoopingController",
    "_format_duration",
    "_parse_duration",
]


class Controller(Protocol):
    """多轮编排层协议。签名与 ``Agent.run()`` 一致，多一个 ``agent`` 参数。"""

    def execute(
        self,
        agent: Agent,
        user_input: Iterable[ExtendedInputContentPart],
        stop_event: asyncio.Event | None,
        memory: Memory,
        tools: Sequence[Tool],
        system_prompt: str = "",
        context: dict[str, Any] | None = None,
        llm_kwargs: dict[str, Any] | None = None,
    ) -> AsyncIterator[AgentEvent | ControllerEvent]: ...


# ── DefaultController ────────────────────────────────────────────────────


class DefaultController:
    """所有现有 agent 类型的兜底。透传 ``agent.run()`` 的事件流。

    - ``AgentStart`` 不对外透传（外层已发 ``ControllerStart``）。
    - ``AgentEnd`` 被吞掉，由 ``ControllerEnd`` 统一收束。
    - ``asyncio.CancelledError``（stop_event 触发）记为 interrupted。
    """

    async def execute(
        self,
        agent: Agent,
        user_input: Iterable[ExtendedInputContentPart],
        stop_event: asyncio.Event | None,
        memory: Memory,
        tools: Sequence[Tool],
        system_prompt: str = "",
        context: dict[str, Any] | None = None,
        llm_kwargs: dict[str, Any] | None = None,
    ) -> AsyncIterator[AgentEvent | ControllerEvent]:
        yield ControllerStart(controller_type="default", user_input=user_input)

        _response = ""
        _time_taken = None
        _exceeded = False
        _interrupted = False
        _error = None

        start_time = time.time()
        try:
            run_kwargs: dict[str, Any] = {}
            if llm_kwargs is not None:
                run_kwargs["llm_kwargs"] = llm_kwargs
            async for event in agent.run(
                user_input=user_input,
                stop_event=stop_event,
                memory=memory,
                tools=tools,
                system_prompt=system_prompt,
                context=context,
                **run_kwargs,
            ):
                if isinstance(event, AgentStart):
                    continue
                if isinstance(event, AgentEnd):
                    _response = event.response
                    _time_taken = event.time_taken
                    _exceeded = event.exceeded
                    _interrupted = event.interrupted
                    _error = event.error
                    continue
                yield event
        except asyncio.CancelledError:
            _interrupted = True
            _error = "Controller execution cancelled"
        except Exception as exc:
            _error = f"{type(exc).__name__}: {exc}"
            _time_taken = time.time() - start_time

        yield ControllerEnd(
            controller_type="default",
            response=_response,
            time_taken=_time_taken,
            exceeded=_exceeded,
            interrupted=_interrupted,
            error=_error,
        )


# ── 时长工具（TimerController 用） ───────────────────────────────────────


def _parse_duration(raw: str | int | float) -> int:
    """解析时长字符串为秒数。支持 ``'30m'`` / ``'1h'`` / ``'300s'`` / ``'1.5h'``。

    无单位或无法解析 → 按秒解释；非法值 → 默认 5 分钟（300 秒）。
    """
    if isinstance(raw, (int, float)):
        return int(raw)
    raw = str(raw).strip().lower()
    if not raw:
        return 300
    if raw.endswith("h"):
        return max(1, int(float(raw[:-1]) * 3600))
    if raw.endswith("m"):
        return max(1, int(float(raw[:-1]) * 60))
    if raw.endswith("s"):
        return max(1, int(float(raw[:-1])))
    try:
        return max(1, int(float(raw)))
    except ValueError:
        return 300  # 默认 5 分钟


def _format_duration(seconds: float) -> str:
    """渲染秒数为可读字符串，如 ``'3m 20s'``。"""
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}h {(s % 3600) // 60}m"
    if s >= 60:
        return f"{s // 60}m {s % 60}s"
    return f"{s}s"


# ── _LoopingController 基类 ──────────────────────────────────────────────


class _LoopingController:
    """``GoalController`` 和 ``TimerController`` 的共享骨架。

    循环：跑 ``agent.run()`` 一轮 → ``_evaluate()`` 一次 judge 调用，同时
    产出 (是否停止, 下一轮 prompt) → 停则 ``ControllerEnd``，继续则
    ``ControllerContinue`` + 下一轮。

    子类覆写：
    - ``_controller_type()`` → 事件里标注的类型名
    - ``_resolve_max_rounds(config)`` → 单次请求的安全轮数上限
    - ``_evaluate(...)`` → 停止判定 + 下一轮 prompt
    - ``_continue_meta(...)`` → ``ControllerContinue.meta``（可选）
    """

    JUDGE_SYSTEM_PROMPT = """\
You are a task-completion judge. Your job is to look at a conversation between
a user and an AI assistant, and determine whether the user's original request
has been fully satisfied.

The assistant has just stopped producing its response. Determine WHY it stopped:

1. **Task complete** — the assistant produced a full, correct solution.
2. **Asked for clarification** — the assistant is waiting for user input.
3. **Gave up / error** — the assistant hit a tool failure or couldn't proceed.
4. **Partial completion** — the assistant made progress but more work remains.

Reply with EXACTLY one line of output:

    DONE
    NEXT: <a concise, actionable user prompt that pushes the assistant toward
           completing the remaining work>

Examples:
- DONE
- NEXT: Implement the remaining 3 API endpoints (PUT, PATCH, DELETE)
- NEXT: The previous tool call failed with a 500 error. Retry with a longer timeout.
- NEXT: Write unit tests for the code you just produced.

Rules:
- If the task is truly complete, say DONE. When in doubt, say DONE.
- If the assistant is stuck or asks the user a question, generate a NEXT prompt
  that makes a reasonable assumption and pushes forward.
- The NEXT prompt should be a single line, 3-50 words, directly actionable.
- Do NOT ask questions in the NEXT prompt. Give an instruction."""

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm_provider = llm_provider

    # ── 子类覆写点 ──

    def _controller_type(self) -> str:
        raise NotImplementedError

    def _resolve_max_rounds(self, config: dict[str, Any]) -> int:
        raise NotImplementedError

    async def _evaluate(
        self,
        memory: Memory,
        agent_end: AgentEnd,
        round_count: int,
        elapsed: float,
        start_time: float,
        stop_event: asyncio.Event | None,
    ) -> tuple[bool, str | None]:
        """返回 (should_stop, next_prompt_or_none)。

        - should_stop=True  → 循环结束，忽略 next_prompt
        - should_stop=False → 用 next_prompt 继续下一轮；next_prompt 为
          None 时视为异常，``ControllerEnd(error=…)`` 收束。
        """
        raise NotImplementedError

    def _continue_meta(
        self,
        round_count: int,
        elapsed: float,
        max_rounds: int,
        config: dict[str, Any],
    ) -> dict[str, Any] | None:
        return None

    # ── execute：循环骨架 ──

    async def execute(
        self,
        agent: Agent,
        user_input: Iterable[ExtendedInputContentPart],
        stop_event: asyncio.Event | None,
        memory: Memory,
        tools: Sequence[Tool],
        system_prompt: str = "",
        context: dict[str, Any] | None = None,
        llm_kwargs: dict[str, Any] | None = None,
    ) -> AsyncIterator[AgentEvent | ControllerEvent]:
        ct = self._controller_type()
        config = (context or {}).get("controller_config", {})
        max_rounds = self._resolve_max_rounds(config)

        yield ControllerStart(controller_type=ct, user_input=user_input)

        current_input = user_input
        start_time = time.time()
        agent_end: AgentEnd | None = None

        for round_count in range(1, max_rounds + 1):
            agent_end = None

            run_kwargs: dict[str, Any] = {}
            if llm_kwargs is not None:
                run_kwargs["llm_kwargs"] = llm_kwargs
            async for event in agent.run(
                user_input=current_input,
                stop_event=stop_event,
                memory=memory,
                tools=tools,
                system_prompt=system_prompt,
                context=context,
                **run_kwargs,
            ):
                if isinstance(event, AgentStart):
                    continue
                if isinstance(event, AgentEnd):
                    agent_end = event
                    continue
                yield event

            if agent_end is None:
                yield ControllerEnd(
                    controller_type=ct,
                    response="",
                    error="Agent returned no AgentEnd",
                )
                return

            if agent_end.interrupted:
                yield ControllerEnd(
                    controller_type=ct,
                    response=agent_end.response,
                    time_taken=agent_end.time_taken,
                    interrupted=True,
                )
                return

            if agent_end.error:
                yield ControllerEnd(
                    controller_type=ct,
                    response=agent_end.response,
                    time_taken=agent_end.time_taken,
                    error=agent_end.error,
                )
                return

            elapsed = time.time() - start_time

            should_stop, next_prompt = await self._evaluate(
                memory=memory,
                agent_end=agent_end,
                round_count=round_count,
                elapsed=elapsed,
                start_time=start_time,
                stop_event=stop_event,
            )

            if should_stop:
                yield ControllerEnd(
                    controller_type=ct,
                    response=agent_end.response,
                    time_taken=agent_end.time_taken,
                )
                return

            if not next_prompt or not next_prompt.strip():
                yield ControllerEnd(
                    controller_type=ct,
                    response=agent_end.response,
                    time_taken=agent_end.time_taken,
                    error="Judge returned empty next_prompt",
                )
                return

            yield ControllerContinue(
                controller_type=ct,
                next_prompt=next_prompt,
                meta=self._continue_meta(round_count, elapsed, max_rounds, config),
            )
            current_input: Iterable[ExtendedInputContentPart] = [
                TextContentPart(type="text", text=next_prompt)
            ]

        # 耗尽 max_rounds
        yield ControllerEnd(
            controller_type=ct,
            response=agent_end.response if agent_end else "",
            exceeded=True,
        )

    # ── 共享的 judge 调用 ──

    async def _call_judge(
        self,
        memory: Memory,
        extra_system_text: str = "",
        stop_event: asyncio.Event | None = None,
    ) -> str | None:
        """调 judge LLM 解析对话。返回 None（DONE）或 next_prompt 文本。

        ``extra_system_text`` 注入到系统 prompt 末尾（用于时间上下文等）。
        ``stop_event`` 透传给 LLM chat()——用户按停时 judge 可被中断。
        """
        system_text = self.JUDGE_SYSTEM_PROMPT
        if extra_system_text:
            system_text = system_text + "\n\n" + extra_system_text

        msgs = memory.get_forward_messages()
        payload = [
            {"role": "system", "content": system_text},
            *msgs,
            {
                "role": "user",
                "content": (
                    "Based on the conversation above, is the user's request "
                    "fully satisfied? Reply DONE or NEXT."
                ),
            },
        ]

        try:
            resp = await self._llm_provider.chat(
                messages=payload,
                tools=[],
                stop_event=stop_event,  # ← 传 stop_event，可中断
            )
            parts: list[str] = []
            async for chunk in resp:
                if chunk.content:
                    parts.append(chunk.content)
            content = "".join(parts).strip()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("controller.judge.error", exc_info=True)
            return None  # 安全默认：不继续

        return self._parse_judge_response(content)

    def _parse_judge_response(self, content: str) -> str | None:
        """解析 judge 输出：DONE（任意大小写/变体）→ None；``NEXT: xxx`` → xxx；其他 → None。"""
        if not content:
            return None
        first_line = content.splitlines()[0].strip()
        upper = first_line.upper()
        if upper.startswith("DONE"):
            return None
        if upper.startswith("NEXT"):
            rest = first_line[len("NEXT") :].lstrip(":： \t-–—").strip()
            return rest if rest else None
        return None


# ── GoalController ───────────────────────────────────────────────────────


class GoalController(_LoopingController):
    """judge LLM 判定 DONE 就停，否则用 ``NEXT: …`` 继续，最多 ``max_goal_rounds`` 轮。"""

    def __init__(self, llm_provider: LLMProvider, max_goal_rounds: int = 5) -> None:
        super().__init__(llm_provider)
        self._default_max_rounds = max_goal_rounds

    def _controller_type(self) -> str:
        return "goal"

    def _resolve_max_rounds(self, config: dict[str, Any]) -> int:
        return int(config.get("max_goal_rounds", self._default_max_rounds))

    async def _evaluate(
        self,
        memory: Memory,
        agent_end: AgentEnd,
        round_count: int,
        elapsed: float,
        start_time: float,
        stop_event: asyncio.Event | None,
    ) -> tuple[bool, str | None]:
        """一次 judge：DONE → 停；NEXT → 继续。"""
        next_prompt = await self._call_judge(memory, stop_event=stop_event)
        if next_prompt is None:
            return True, None
        return False, next_prompt

    def _continue_meta(
        self,
        round_count: int,
        elapsed: float,
        max_rounds: int,
        config: dict[str, Any],
    ) -> dict[str, Any] | None:
        return {"round": round_count, "max_rounds": max_rounds}


# ── TimerController ──────────────────────────────────────────────────────


class TimerController(_LoopingController):
    """累计运行时间 >= 用户指定时长就停；时间未到则强制继续（judge 说 DONE 也继续）。"""

    def __init__(
        self,
        llm_provider: LLMProvider,
        default_duration: str = "30m",
        max_rounds: int = 100,
    ) -> None:
        super().__init__(llm_provider)
        self._default_duration = default_duration
        self._max_rounds = max_rounds
        self._config: dict[str, Any] = {}

    def _controller_type(self) -> str:
        return "timer"

    def _resolve_max_rounds(self, config: dict[str, Any]) -> int:
        # timer 的生命周期由时长控制，round 计数只是安全上限
        return self._max_rounds

    def _resolve_duration(self, config: dict[str, Any]) -> int:
        return _parse_duration(config.get("duration", self._default_duration))

    async def execute(
        self,
        agent: Agent,
        user_input: Iterable[ExtendedInputContentPart],
        stop_event: asyncio.Event | None,
        memory: Memory,
        tools: Sequence[Tool],
        system_prompt: str = "",
        context: dict[str, Any] | None = None,
        llm_kwargs: dict[str, Any] | None = None,
    ) -> AsyncIterator[AgentEvent | ControllerEvent]:
        self._config = (context or {}).get("controller_config", {})
        async for event in super().execute(
            agent,
            user_input,
            stop_event,
            memory,
            tools,
            system_prompt,
            context,
            llm_kwargs,
        ):
            yield event

    async def _evaluate(
        self,
        memory: Memory,
        agent_end: AgentEnd,
        round_count: int,
        elapsed: float,
        start_time: float,
        stop_event: asyncio.Event | None,
    ) -> tuple[bool, str | None]:
        duration = self._resolve_duration(self._config)

        if elapsed >= duration:
            return True, None  # 时间到了 → 停

        # 时间没到 → 强制继续。先问 judge 要 prompt
        time_ctx = (
            f"Time context: The user allocated {_format_duration(duration)} for this "
            f"task. {_format_duration(elapsed)} has elapsed, "
            f"{_format_duration(duration - elapsed)} remaining."
        )
        next_prompt = await self._call_judge(
            memory,
            extra_system_text=time_ctx,
            stop_event=stop_event,
        )

        if next_prompt is not None:
            return False, next_prompt

        # judge 说 DONE 或出错，但时间未到——用模板 prompt 强制继续
        forced = (
            f"The allocated time ({_format_duration(duration)}) has not yet elapsed "
            f"({_format_duration(elapsed)} elapsed, "
            f"{_format_duration(duration - elapsed)} remaining). "
            f"Continue working on the original task."
        )
        return False, forced

    def _continue_meta(
        self,
        round_count: int,
        elapsed: float,
        max_rounds: int,
        config: dict[str, Any],
    ) -> dict[str, Any] | None:
        duration = self._resolve_duration(config)
        return {
            "elapsed": int(elapsed),
            "remaining": max(0, duration - int(elapsed)),
            "duration": duration,
        }
