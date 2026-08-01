"""Controller layer: multi-round orchestration wrapping a single Agent run.

架构分层：Agent（单次 LLM Loop）之上新增 Controller 层。Controller 包裹
Agent 做多轮编排，持有自有事件（``ControllerStart / ControllerContinue /
ControllerEnd``，见 ``minimal_harness.types``）与生命周期。Agent 对
Controller 零感知——Controller 只是调用 ``agent.run()`` 并消费其事件流。

框架只提供协议与兜底实现：
- :class:`Controller`——多轮编排协议（``execute()`` 签名）。
- :class:`DefaultController`——所有 agent 类型的兜底，行为上是透传。

具体的应用层策略 Controller（``GoalController`` / ``TimerController`` 等）
由消费方（如 mh-gateway 的 ``mh_gateway.services.controllers``）实现，
通过 ``ControllerRegistry.register()`` 插入 runtime——这同时验证了外部
应用可以在 controller 层插入自定义 controller 的扩展点。
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

from minimal_harness.memory import ExtendedInputContentPart, Memory
from minimal_harness.tool.base import Tool
from minimal_harness.types import (
    AgentEnd,
    AgentEvent,
    ControllerEnd,
    ControllerEvent,
    ControllerStart,
)

if TYPE_CHECKING:
    from minimal_harness.agent.protocol import Agent

logger = logging.getLogger(__name__)

__all__ = [
    "Controller",
    "DefaultController",
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
        controller_config: dict[str, Any] | None = None,
        llm_kwargs: dict[str, Any] | None = None,
    ) -> AsyncIterator[AgentEvent | ControllerEvent]: ...


# ── DefaultController ────────────────────────────────────────────────────


class DefaultController:
    """所有现有 agent 类型的兜底。透传 ``agent.run()`` 的完整事件流。

    - ``AgentStart`` / ``AgentEnd`` 原样透传，不吞任何 agent 事件。
    - ``ControllerStart`` / ``ControllerEnd`` 包裹运行生命周期，收束信息
      与 AgentEnd 保持一致。
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
        controller_config: dict[str, Any] | None = None,
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
                if isinstance(event, AgentEnd):
                    _response = event.response
                    _time_taken = event.time_taken
                    _exceeded = event.exceeded
                    _interrupted = event.interrupted
                    _error = event.error
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
