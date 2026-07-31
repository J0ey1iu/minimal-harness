# Controller Layer Design & Implementation Plan

## 1. 架构分层

当前三层 → 新四层：

```
Agent              → 单次 LLM Loop。run(input, …) → AsyncIterator<AgentEvent>
                    对 Controller 零感知。不变。

Controller         → 包裹 Agent，多轮编排。execute(agent, input, …) → AsyncIterator<AgentEvent | ControllerEvent>
                    自有协议，自有事件层，自定生命周期。                 ← 新层

AgentRuntime       → 不再直接调 agent.run()。改为：
                      controller = resolve_controller(type) → controller.execute(agent, …)

Gateway / Local    → 消费 AgentEvent + ControllerEvent。ChatRequest 加 controller 字段 + controller_config 选择类型与参数。
```

Agent 和 Controller 完全解耦：
- `AgentMetadata` 不加任何 Controller 字段——Agent 不知道 Controller 存在。
- Controller 选择是 **per-request** 的，由用户在每次输入时指定（`ChatRequest.controller`）。
- Controller 运行时参数（`max_goal_rounds`、`duration` 等）来自 `ChatRequest.controller_config: dict`，不来自 AgentMetadata。Gateway config 只存默认值。

## 2. 三种 Controller 一览

| Controller | 停止条件 | 下一轮 prompt 来源 | 典型参数 |
|---|---|---|---|
| `default` | Agent 跑完就停 | 不需要——只跑一轮 | 无 |
| `goal` | judge LLM 判定 DONE | judge LLM 分析对话生成 NEXT | `max_goal_rounds`（默认来自 gateway config） |
| `timer` | 累计运行时间 ≥ 用户指定时长 | judge LLM 分析对话生成 NEXT（注入时间上下文） | `duration`（如 `"30m"`） |

`goal` 和 `timer` 结构高度相似——都是"跑 Agent → 判断 → 生成新 prompt → 再跑"的循环。差异仅在：
- 停止条件（judge 说 DONE vs 时间到）
- judge 的额外上下文（无 vs 已过/剩余时间）

因此抽一个 `_LoopingController` 基类，两个子类只覆写 `_should_stop()` 和 `_judge_extra_context()`。

## 3. Controller 协议（`agent/controller.py`）

```python
class Controller(Protocol):
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
    ) -> AsyncIterator[AgentEvent | ControllerEvent]: ...
```

签名与 `Agent.run()` 高度一致，唯二差异：多了 `agent: Agent` 参数，返回值联合多了 `ControllerEvent`。

Controller 配置从 `context["controller_config"]` 取——runtime 把 `ChatRequest.controller_config` 写入 run_context 透传进来。Controller 从 context 里自取自己需要的 key，不同 Controller 互不感知对方的配置项。

## 4. ControllerEvent 类型（`types.py`）

三个事件，覆盖 Controller 完整生命周期。**不带 Goal/Timer 前缀**——`controller_type` 字段区分具体类型：

```python
@dataclass
class ControllerStart:
    controller_type: str       # "default" / "goal" / "timer" / …
    user_input: Any
    timestamp: float = field(default_factory=time.time)


@dataclass
class ControllerContinue:
    controller_type: str
    next_prompt: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ControllerEnd:
    controller_type: str
    response: str
    time_taken: float | None = None
    exceeded: bool = False
    interrupted: bool = False
    error: str | None = None
    timestamp: float = field(default_factory=time.time)


ControllerEvent = ControllerStart | ControllerContinue | ControllerEnd
```

`ControllerEvent` 独立于 `AgentEvent` 联合。AgentRuntime 的事件队列类型改为 `AgentEvent | ControllerEvent | None`。

**扩展方式**：任何新 Controller 类型加同一个三种事件，只在 `controller_type` 字段里标身份。不加新事件类。

## 5. DefaultController

所有现有 agent 类型的兜底。行为上是透传，语义上是"框架里永远有 Controller 这一层"：

```python
class DefaultController:
    async def execute(self, agent, user_input, stop_event, memory, tools,
                      system_prompt="", context=None, llm_kwargs=None):
        yield ControllerStart(controller_type="default", user_input=user_input)

        response = ""
        time_taken = None
        exceeded = False
        interrupted = False
        error = None

        start_time = time.time()
        try:
            async for event in agent.run(
                user_input=user_input,
                stop_event=stop_event,
                memory=memory,
                tools=tools,
                system_prompt=system_prompt,
                context=context,
                llm_kwargs=llm_kwargs,
            ):
                if isinstance(event, AgentStart):
                    continue
                if isinstance(event, AgentEnd):
                    response = event.response
                    time_taken = event.time_taken
                    exceeded = event.exceeded
                    interrupted = event.interrupted
                    error = event.error
                    continue
                yield event
        except asyncio.CancelledError:
            interrupted = True
            error = "Controller execution cancelled"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            time_taken = time.time() - start_time

        yield ControllerEnd(
            controller_type="default",
            response=response,
            time_taken=time_taken,
            exceeded=exceeded,
            interrupted=interrupted,
            error=error,
        )
```

要点：
- `AgentStart` / `AgentEnd` 原样透传，不吞任何 agent 事件。
- 生命周期由 `ControllerStart` / `ControllerEnd` 包裹，收束信息与 AgentEnd 一致。
- `asyncio.CancelledError` 单独捕获——和 `asyncio.TimeoutError` 不同，这是 stop_event 驱动的中断。
- 框架里不存"无 Controller"的代码路径。所有 agent 调用都经由 Controller。

## 6. `_LoopingController` 基类（共享 Goal + Timer 的骨架）

```python
class _LoopingController:
    """GoalController 和 TimerController 的共享基类。

    子类覆写：
      - _should_stop(memory, agent_end, round_count, elapsed, start_time)
          → (should_stop: bool, stop_reason: str, end_kwargs: dict)
      - _judge_extra_context(memory, round_count, elapsed)
          → str  (注入到 judge 系统 prompt 的额外段落)
    """

    def __init__(self, llm_provider):
        self._llm_provider = llm_provider

    async def execute(self, agent, user_input, stop_event, memory, tools,
                      system_prompt="", context=None, llm_kwargs=None):
        controller_type = self._controller_type()   # "goal" or "timer"
        config = (context or {}).get("controller_config", {})
        max_rounds = self._resolve_max_rounds(config)

        yield ControllerStart(
            controller_type=controller_type,
            user_input=user_input,
        )

        current_input = user_input
        start_time = time.time()
        round_count = 0

        for round_count in range(1, max_rounds + 1):
            agent_end = None
            round_start = time.time()

            async for event in agent.run(
                user_input=current_input,
                stop_event=stop_event,
                memory=memory,
                tools=tools,
                system_prompt=system_prompt,
                context=context,
                llm_kwargs=llm_kwargs,
            ):
                if isinstance(event, AgentStart):
                    continue
                if isinstance(event, AgentEnd):
                    agent_end = event
                    continue
                yield event

            # --- Agent 这一轮结束，判定 ---

            if agent_end is None:
                yield ControllerEnd(
                    controller_type=controller_type,
                    response="",
                    error="Agent returned no AgentEnd",
                )
                return

            if agent_end.interrupted:
                yield ControllerEnd(
                    controller_type=controller_type,
                    response=agent_end.response,
                    time_taken=agent_end.time_taken,
                    interrupted=True,
                )
                return

            if agent_end.error:
                yield ControllerEnd(
                    controller_type=controller_type,
                    response=agent_end.response,
                    time_taken=agent_end.time_taken,
                    error=agent_end.error,
                )
                return

            elapsed = time.time() - start_time

            # 子类决定：停还是继续？
            should_stop, stop_reason, end_kwargs = await self._should_stop(
                memory=memory,
                agent_end=agent_end,
                round_count=round_count,
                elapsed=elapsed,
                start_time=start_time,
                stop_event=stop_event,
            )

            if should_stop:
                yield ControllerEnd(
                    controller_type=controller_type,
                    response=agent_end.response,
                    time_taken=agent_end.time_taken,
                    **end_kwargs,
                )
                return

            # 不停——生成下一个 prompt
            next_prompt = await self._generate_next_prompt(
                memory=memory,
                round_count=round_count,
                elapsed=elapsed,
                stop_event=stop_event,
            )

            if not next_prompt or not next_prompt.strip():
                yield ControllerEnd(
                    controller_type=controller_type,
                    response=agent_end.response,
                    time_taken=agent_end.time_taken,
                    error="Judge returned empty next_prompt",
                )
                return

            yield ControllerContinue(
                controller_type=controller_type,
                next_prompt=next_prompt,
            )
            current_input = [{"type": "text", "text": next_prompt}]

        # 耗尽 max_rounds
        yield ControllerEnd(
            controller_type=controller_type,
            response=agent_end.response if agent_end else "",
            exceeded=True,
        )

    # ── 子类覆写点 ──

    def _controller_type(self) -> str:
        raise NotImplementedError

    def _resolve_max_rounds(self, config: dict) -> int:
        raise NotImplementedError

    async def _should_stop(self, memory, agent_end, round_count,
                           elapsed, start_time, stop_event
                           ) -> tuple[bool, str, dict]:
        """返回 (should_stop, reason_for_log, end_kwargs_for_ControllerEnd)。"""
        raise NotImplementedError

    def _judge_system_prompt_extra(self, memory, round_count, elapsed) -> str:
        """注入到 judge 系统 prompt 里的额外段落。返回空字符串表示不注入。"""
        return ""

    # ── 共享的 judge 调用 ──

    JUDGE_BASE_PROMPT = """\
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

    async def _generate_next_prompt(self, memory, round_count, elapsed,
                                    stop_event=None) -> str | None:
        """调 judge LLM，返回 DONE→None / NEXT→prompt_text / 异常→None。"""
        extra = self._judge_system_prompt_extra(memory, round_count, elapsed)
        system_prompt = self.JUDGE_BASE_PROMPT
        if extra:
            system_prompt = system_prompt + "\n\n" + extra

        msgs = memory.get_forward_messages()
        payload = [
            {"role": "system", "content": system_prompt},
            *msgs,
            {"role": "user", "content": (
                "Based on the conversation above, is the user's request "
                "fully satisfied? Reply DONE or NEXT."
            )},
        ]

        try:
            resp = await self._llm_provider.chat(
                messages=payload,
                tools=[],
                stop_event=stop_event,          # ← 传 stop_event，可中断
            )
            content = (resp.response.content or "").strip()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("controller.judge.error", exc_info=True)
            return None  # 安全默认：不继续

        return self._parse_judge_response(content)

    def _parse_judge_response(self, content: str) -> str | None:
        """解析 judge 输出：DONE→None / NEXT: xxx→xxx / 其他→None。"""
        if not content:
            return None
        first_line = content.splitlines()[0].strip()
        if first_line.upper().startswith("DONE"):
            return None
        if first_line.upper().startswith("NEXT"):
            rest = first_line[len("NEXT"):].lstrip(":： \t-–—").strip()
            return rest if rest else None
        return None
```

## 7. GoalController

```python
class GoalController(_LoopingController):
    def __init__(self, llm_provider, max_goal_rounds):
        super().__init__(llm_provider)
        self._default_max_rounds = max_goal_rounds

    def _controller_type(self) -> str:
        return "goal"

    def _resolve_max_rounds(self, config: dict) -> int:
        return int(config.get("max_goal_rounds", self._default_max_rounds))

    async def _should_stop(self, memory, agent_end, round_count,
                           elapsed, start_time, stop_event):
        verdict = await self._generate_next_prompt(memory, round_count, elapsed,
                                                    stop_event)
        if verdict is None:
            # judge 返回 DONE 或失败 → 停止
            return True, "judge_done", {}
        # verdict 不为 None → 继续（下面调用 _generate_next_prompt 再次生成 prompt）
        # 这里需要存下来避免重复调用 judge：
        self._cached_next_prompt = verdict
        return False, "", {}

    async def _generate_next_prompt(self, memory, round_count, elapsed,
                                    stop_event=None) -> str | None:
        # 如果 _should_stop 已经调过 judge 且结果不是 None，直接用缓存
        cached = getattr(self, '_cached_next_prompt', None)
        if cached is not None:
            self._cached_next_prompt = None
            return cached
        return await super()._generate_next_prompt(memory, round_count, elapsed,
                                                    stop_event)
```

等一下——上面这个设计有问题。`_should_stop` 和 `_generate_next_prompt` 各调一次 judge，重复了。我重新想一下 `_LoopingController` 的结构。

核心问题是：GoalController 的 judge 同时决定了两个事情——"是否停止"和"下一轮 prompt 文字"。而 TimerController 的"是否停止"是计时器决定的（和 judge 无关），judge 只负责生成 prompt。所以 `_should_stop` 不应该调 judge；judge 应该是独立的一步：

```
1. agent_end = (run agent)
2. should_stop = _should_stop(...)         # Goal：看 judge 结果；Timer：看时间
3. if should_stop → ControllerEnd, return
4. next_prompt = _generate_next_prompt()    # 不管谁都要生成 prompt（除非停了）
5. yield ControllerContinue, 下一轮
```

这样 GoalController 的 `_should_stop` 也要调 judge——但 `_generate_next_prompt` 也要调 judge。两次。

**优化**：让 `_should_stop` 返回 `(should_stop, verdict)` 元组——其中 `verdict` 是 judge 的解析结果（None = DONE 或失败停止；非空 = next_prompt）。然后基类的循环逻辑拿这个 verdict 当下一轮输入。TimerController 的 `_should_stop` 自己做 judge 调用并返回 verdict（它的 stop 是时间判断，但 judge 结果依然需要）。

实际代码结构：

```python
# 循环内：
should_stop, next_prompt = await self._evaluate(
    memory, agent_end, round_count, elapsed, start_time, stop_event,
)
# should_stop==True → ControllerEnd
# should_stop==False 且 next_prompt 非空 → ControllerContinue + 下一轮
# should_stop==False 且 next_prompt 为空 → 异常处理
```

这样 GoalController 调一次 judge，TimerController 也调一次 judge。一次 judge 同时出两个结论。完美。

好，修正后的设计：

```python
class _LoopingController:

    async def _evaluate(
        self, memory, agent_end, round_count, elapsed, start_time, stop_event
    ) -> tuple[bool, str | None]:
        """返回 (should_stop, next_prompt_or_none)。
        
        子类覆写此方法实现自己的评估逻辑。
        """
        raise NotImplementedError
```

GoalController:
```python
async def _evaluate(self, memory, agent_end, round_count, elapsed, start_time, stop_event):
    # judge → 得到 DONE 或 NEXT: xxx
    verdict = await self._call_judge(memory, round_count, elapsed, stop_event)
    if verdict is None:
        return True, None  # DONE 或出错 → 停
    return False, verdict  # NEXT → 继续
```

TimerController:
```python
async def _evaluate(self, memory, agent_end, round_count, elapsed, start_time, stop_event):
    config = self._config
    duration = self._resolve_duration(config)
    
    if elapsed >= duration:
        return True, None  # 时间到 → 停
    
    # 时间没到——强制度要继续，但需要 prompt
    verdict = await self._call_judge(memory, round_count, elapsed, stop_event)
    # 不管 judge 说 DONE 还是 NEXT，只要时间没到就继续
    # 如果 judge 说 DONE → 用默认 prompt 推它继续
    if verdict is None:
        next_prompt = (
            f"Continue working on the original task. "
            f"Elapsed: {_format_time(elapsed)}, "
            f"Remaining: {_format_time(duration - elapsed)}."
        )
    else:
        next_prompt = verdict
    return False, next_prompt
```

这样就清晰了。一个 `_evaluate()` 方法替代 `_should_stop` + `_generate_next_prompt`，一次 judge 调用，同时产出停止判定和下一轮 prompt。

好，现在更新文档。这是比较复杂的重构，文档要写清楚。

另外 stop_event 的修复也在 _LoopingController._call_judge 里落地——传 stop_event。

关于 `build_summarizer` 的 stop_event 问题——单独记债，不在本次改。<｜end▁of▁thinking｜>思考过程中发现了一个设计问题：GoalController 里 judge 同时决定"是否停止"和"下一轮 prompt"，如果拆成 `_should_stop` + `_generate_next_prompt` 两个方法，judge 会被调两次。需要合并为一个 `_evaluate()`，一次 judge 出两个结论。TimerController 的评估逻辑也可以塞进同一个方法里——一次 judge 出 prompt，自己拿计时器判停。

我把这个修正和 TimerController 的设计写进去。整份文档重新写过。

<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="write">
<｜｜DSML｜｜parameter name="content" string="true"># Controller Layer Design & Implementation Plan

## 1. 架构分层

当前三层 → 新四层：

```
Agent              → 单次 LLM Loop。run(input, …) → AsyncIterator<AgentEvent>
                    对 Controller 零感知。不变。

Controller         → 包裹 Agent，多轮编排。execute(agent, input, …) → AsyncIterator<AgentEvent | ControllerEvent>
                    自有协议，自有事件层，自定生命周期。                 ← 新层

AgentRuntime       → 不再直接调 agent.run()。改为：
                      controller = resolve_controller(type) → controller.execute(agent, …)

Gateway / Local    → 消费 AgentEvent + ControllerEvent。
                    ChatRequest 加 controller + controller_config 字段选择类型与参数。
```

Agent 和 Controller 完全解耦：
- `AgentMetadata` 不加任何 Controller 字段——Agent 不知道 Controller 存在。
- Controller 选择是 **per-request** 的，在 `ChatRequest.controller` 中指定。
- Controller 运行时参数（`max_goal_rounds`、`duration` 等）来自 `ChatRequest.controller_config: dict`。Gateway config 只存默认值。

## 2. 三种 Controller 一览

| Controller | 停止条件 | 下一轮 prompt 来源 | 典型参数 |
|---|---|---|---|
| `default` | Agent 跑完就停 | 不需要——只跑一轮 | 无 |
| `goal` | judge LLM 判定 DONE | judge LLM 分析对话，输出 `NEXT: <prompt>` | `max_goal_rounds`（默认来自 gateway config） |
| `timer` | 累计运行时间 ≥ 用户指定时长 | 同上（judge prompt 注入时间上下文；若 judge 返回 DONE 但时间未到，用模板 prompt 强制继续） | `duration`（如 `"30m"`） |

`goal` 和 `timer` 结构高度相似的循环——都是"跑 Agent → 调一次 judge 评估 → 停或继续"。差异仅在 stop 条件的判断逻辑。抽一个 `_LoopingController` 基类，子类覆写一个方法 `_evaluate()`。

## 3. Controller 协议（`agent/controller.py`）

```python
class Controller(Protocol):
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
    ) -> AsyncIterator[AgentEvent | ControllerEvent]: ...
```

- 和 `Agent.run()` 签名基本一致，多一个 `agent: Agent`。
- Controller 配置从 `context["controller_config"]` 取——AgentRuntime 把 `ChatRequest.controller_config` 写入 run_context。
- 每个 Controller 只读自己关心的 key，不同 Controller 的配置互不感知。

## 4. ControllerEvent 类型（`types.py`）

三个事件，覆盖 Controller 完整生命周期。`controller_type` 字段区分具体类型——不加任何 Goal/Timer 专属事件：

```python
@dataclass
class ControllerStart:
    controller_type: str       # "default" / "goal" / "timer" / …
    user_input: Any
    timestamp: float = field(default_factory=time.time)


@dataclass
class ControllerContinue:
    controller_type: str
    next_prompt: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ControllerEnd:
    controller_type: str
    response: str
    time_taken: float | None = None
    exceeded: bool = False
    interrupted: bool = False
    error: str | None = None
    timestamp: float = field(default_factory=time.time)


ControllerEvent = ControllerStart | ControllerContinue | ControllerEnd
```

- `ControllerEvent` 独立于 `AgentEvent` 联合。AgentRuntime 的事件队列改为 `AgentEvent | ControllerEvent | None`。

## 5. DefaultController（`agent/controller.py`）

所有现有 agent 类型的兜底。行为上是透传，语义上是"框架里永远有 Controller 这一层"：

```python
class DefaultController:
    async def execute(self, agent, user_input, stop_event, memory, tools,
                      system_prompt="", context=None, llm_kwargs=None):
        yield ControllerStart(controller_type="default", user_input=user_input)

        _response = ""
        _time_taken = None
        _exceeded = False
        _interrupted = False
        _error = None

        start_time = time.time()
        try:
            async for event in agent.run(
                user_input=user_input,
                stop_event=stop_event,
                memory=memory,
                tools=tools,
                system_prompt=system_prompt,
                context=context,
                llm_kwargs=llm_kwargs,
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
```

关键行为：
- `AgentStart` / `AgentEnd` 原样透传，不吞任何 agent 事件。
- 生命周期由 `ControllerStart` / `ControllerEnd` 包裹，收束信息与 AgentEnd 一致。
- `asyncio.CancelledError`（stop_event 触发）单独 catch。
- AgentRuntime 里所有 `agent.run()` 调用都必须经过 Controller——不存"无 Controller"的代码分支。

## 6. `_LoopingController` 基类——Goal / Timer 共享骨架

两个循环型 Controller 的共享逻辑。子类只覆写一个方法：

```python
class _LoopingController:
    """GoalController 和 TimerController 的共享基类。

    子类覆写点：
      - _controller_type() → str
      - _resolve_max_rounds(config) → int（默认无上限，只受 _evaluate 判停约束）
      - _evaluate(memory, elapsed, stop_event)
          → tuple[bool, str | None]
             (should_stop, next_prompt_or_none)
    """

    def __init__(self, llm_provider):
        self._llm_provider = llm_provider

    # ── 子类覆写点 ──

    def _controller_type(self) -> str: raise NotImplementedError
    def _resolve_max_rounds(self, config: dict) -> int | None:
        # 默认无轮数上限：循环只受 _evaluate 判停约束（timer 只受时间限制）
        return None

    async def _evaluate(
        self, memory, elapsed, stop_event
    ) -> tuple[bool, str | None]:
        """返回 (should_stop, next_prompt_or_none)。

        - should_stop=True  → 循环结束，忽略 next_prompt
        - should_stop=False → 用 next_prompt 继续下一轮；next_prompt 为 None 时
          视为异常，ControllerEnd(error=…)。
        """
        raise NotImplementedError

    # ── execute：循环骨架 ──

    async def execute(self, agent, user_input, stop_event, memory, tools,
                      system_prompt="", context=None, llm_kwargs=None):
        ct = self._controller_type()
        config = (context or {}).get("controller_config", {})
        max_rounds = self._resolve_max_rounds(config)

        yield ControllerStart(controller_type=ct, user_input=user_input)

        current_input = user_input
        start_time = time.time()
        agent_end = None

        for round_count in range(1, max_rounds + 1):
            agent_end = None

            async for event in agent.run(
                user_input=current_input,
                stop_event=stop_event,
                memory=memory,
                tools=tools,
                system_prompt=system_prompt,
                context=context,
                llm_kwargs=llm_kwargs,
            ):
                if isinstance(event, AgentStart):
                    continue
                if isinstance(event, AgentEnd):
                    agent_end = event
                    continue
                yield event

            if agent_end is None:
                yield ControllerEnd(ct, response="", error="Agent returned no AgentEnd")
                return

            if agent_end.interrupted:
                yield ControllerEnd(ct, response=agent_end.response,
                                    time_taken=agent_end.time_taken, interrupted=True)
                return

            if agent_end.error:
                yield ControllerEnd(ct, response=agent_end.response,
                                    time_taken=agent_end.time_taken, error=agent_end.error)
                return

            elapsed = time.time() - start_time

            should_stop, next_prompt = await self._evaluate(
                memory, agent_end, round_count, elapsed, start_time, stop_event,
            )

            if should_stop:
                yield ControllerEnd(
                    ct, response=agent_end.response,
                    time_taken=agent_end.time_taken,
                )
                return

            if not next_prompt or not next_prompt.strip():
                yield ControllerEnd(
                    ct, response=agent_end.response,
                    time_taken=agent_end.time_taken,
                    error="Judge returned empty next_prompt",
                )
                return

            yield ControllerContinue(
                controller_type=ct,
                next_prompt=next_prompt,
            )
            current_input = [{"type": "text", "text": next_prompt}]

        # 耗尽 max_rounds
        yield ControllerEnd(
            ct,
            response=agent_end.response if agent_end else "",
            exceeded=True,
        )

    # ── 共享的 judge 调用（所有子类共用） ──

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

    async def _call_judge(
        self, memory, extra_system_text="", stop_event=None
    ) -> str | None:
        """调 judge LLM 解析对话。返回 None（DONE）或 next_prompt 文本。

        ``extra_system_text`` 注入到系统 prompt 末尾（用于时间上下文等）。
        stop_event 透传给 LLM chat()——用户按停时 judge 可被中断。
        """
        system_text = self.JUDGE_SYSTEM_PROMPT
        if extra_system_text:
            system_text = system_text + "\n\n" + extra_system_text

        msgs = memory.get_forward_messages()
        payload = [
            {"role": "system", "content": system_text},
            *msgs,
            {"role": "user", "content": (
                "Based on the conversation above, is the user's request "
                "fully satisfied? Reply DONE or NEXT."
            )},
        ]

        try:
            resp = await self._llm_provider.chat(
                messages=payload,
                tools=[],
                stop_event=stop_event,       # ← 传 stop_event（修复 "内部 LLM 调用不可中断"）
            )
            content = (resp.response.content or "").strip()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("controller.judge.error", exc_info=True)
            return None   # 安全默认：不继续

        return self._parse_judge_response(content)

    def _parse_judge_response(self, content: str) -> str | None:
        """DONE (任意大小写/变体) → None；NEXT: xxx → xxx；其他 → None。"""
        if not content:
            return None
        first_line = content.splitlines()[0].strip()
        upper = first_line.upper()
        if upper.startswith("DONE"):
            return None
        if upper.startswith("NEXT"):
            rest = first_line[len("NEXT"):].lstrip(":： \t-–—").strip()
            return rest if rest else None
        return None
```

## 7. GoalController

```python
class GoalController(_LoopingController):
    def __init__(self, llm_provider, max_goal_rounds):
        super().__init__(llm_provider)
        self._default_max_rounds = max_goal_rounds

    def _controller_type(self) -> str:
        return "goal"

    def _resolve_max_rounds(self, config: dict) -> int:
        return int(config.get("max_goal_rounds", self._default_max_rounds))

    async def _evaluate(self, memory, elapsed, stop_event):
        """Judge 判定：DONE → 停；NEXT → 继续。"""
        next_prompt = await self._call_judge(memory, stop_event=stop_event)
        if next_prompt is None:
            return True, None   # 停
        return False, next_prompt
```

## 8. TimerController

```python
import re

def _parse_duration(raw: str) -> int:
    """解析时长字符串为秒数。支持 '30m' / '1h' / '300s' / '1.5h'。

    默认单位：无单位或无法解析 → 秒。"""
    raw = raw.strip().lower()
    if raw.endswith("h"):
        return int(float(raw[:-1]) * 3600)
    if raw.endswith("m"):
        return int(float(raw[:-1]) * 60)
    if raw.endswith("s"):
        return int(float(raw[:-1]))
    try:
        return int(float(raw))
    except ValueError:
        return 300  # 默认 5 分钟


def _format_duration(seconds: float) -> str:
    """渲染秒数为可读字符串，如 '3m 20s'。"""
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}h {(s % 3600) // 60}m"
    if s >= 60:
        return f"{s // 60}m {s % 60}s"
    return f"{s}s"


class TimerController(_LoopingController):
    def __init__(self, llm_provider, default_duration):
        super().__init__(llm_provider)
        self._default_duration = default_duration

    def _controller_type(self) -> str:
        return "timer"

    def _resolve_duration(self, config: dict) -> int:
        raw = config.get("duration", self._default_duration)
        return _parse_duration(str(raw))

    async def _evaluate(self, memory, agent_end, round_count, elapsed,
                        start_time, stop_event):
        config = ( # 从已保存的 config 读取——execute 里存了
            getattr(self, '_config', {})
        )
        duration = self._resolve_duration(config)

        if elapsed >= duration:
            return True, None   # 时间到了 → 停

        # 时间没到 → 强制继续。先问 judge 要 prompt
        time_ctx = (
            f"Time context: The user allocated {_format_duration(duration)} for this task. "
            f"{_format_duration(elapsed)} has elapsed, "
            f"{_format_duration(duration - elapsed)} remaining."
        )
        next_prompt = await self._call_judge(
            memory, extra_system_text=time_ctx, stop_event=stop_event,
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

    # execute 需要拿到 config 给 _evaluate 用：
    async def execute(self, agent, user_input, stop_event, memory, tools,
                      system_prompt="", context=None, llm_kwargs=None):
        self._config = (context or {}).get("controller_config", {})
        async for event in super().execute(agent, user_input, stop_event, memory,
                                           tools, system_prompt, context, llm_kwargs):
            yield event
```

## 9. 内部 LLM 调用的 stop_event 修复

**问题**：`build_summarizer` 的内部 LLM 调用不传 `stop_event`——用户按停时无法中断压缩过程。Controller 里的 judge 调用如果不传也会复现同一问题。

**本次修复范围**：Controller 层的所有内部 LLM 调用都传 `stop_event`：

| 调用点 | 修复情况 |
|---|---|
| `_LoopingController._call_judge()` | `stop_event` 作为参数传入，透传到 `llm_provider.chat(stop_event=stop_event)` |
| `DefaultController` | 本身不调 LLM |
| `build_summarizer`（`_compaction.py`） | **积债，本次不改**——签名里没有 stop_event 参数，改动涉及 CompactionAgent + ToolCompactionAgent + compact_session 三处调用点，单独修 |

## 10. AgentRuntime 改动（`agent/runtime.py`）

```python
class AgentRuntime:

    def __init__(self, …, controller_registry: ControllerRegistry | None = None):
        …
        self._controller_registry = controller_registry or ControllerRegistry()

    def register_controller(self, name: str, factory: Callable[..., Controller]):
        self._controller_registry.register(name, factory)

    async def run(
        self,
        user_input, agent_metadata_id, memory_id,
        agent_type=None, tool_names=None, context=None, llm_kwargs=None,
        controller_type: str = "default",           # ← 新增
        controller_config: dict[str, Any] | None = None,  # ← 新增
    ) -> tuple[asyncio.Task, asyncio.Event, asyncio.Queue[AgentEvent | ControllerEvent | None]]:

        …  # session、tools（不变）

        run_context = {**base, **(context or {}),
                       "controller_config": controller_config or {},
                       "correlation_id": correlation_id}

        controller = self._controller_registry.create(
            controller_type,
            llm_provider=self._llm_provider_resolver(metadata),
        )

        async def _run():
            …
            async for event in controller.execute(
                agent=agent,
                user_input=user_input,
                stop_event=stop_event,
                memory=session,
                tools=tools,
                system_prompt=…,
                context=run_context,
                llm_kwargs=llm_kwargs,
            ):
                await event_queue.put(event)
            …
```

`ControllerRegistry`：

```python
class ControllerRegistry:
    def __init__(self):
        self._factories: dict[str, Callable[..., Controller]] = {}

    def register(self, name: str, factory: Callable[..., Controller]):
        self._factories[name] = factory

    def create(self, name: str, **kwargs) -> Controller:
        factory = self._factories.get(name)
        if factory is None:
            logger.warning("controller.not_registered type=%s fallback=default", name)
            return DefaultController()
        return factory(**kwargs)

    def list_types(self) -> list[str]:
        return list(self._factories.keys())
```

## 11. Gateway 改动（`mh-gateway`）

### 11.1 ChatRequest（`api/chat.py`）

```python
class ChatRequest(BaseModel):
    message: str
    controller: str = "default"
    controller_config: dict[str, Any] = {}
```

透传：

```python
task, stop_event, queue = await runtime.run(
    …,
    controller_type=body.controller,
    controller_config=body.controller_config,
)
```

### 11.2 Controller 注册（`services/runtime_service.py` 或 `app.py`）

gateway 启动时注册：

```python
runtime.register_controller(
    "goal",
    lambda llm_provider: GoalController(
        llm_provider=llm_provider,
        max_goal_rounds=settings.goal_max_rounds,
    ),
)
runtime.register_controller(
    "timer",
    lambda llm_provider: TimerController(
        llm_provider=llm_provider,
        default_duration=settings.timer_default_duration,
    ),
)
# "default" 不需要显式注册——ControllerRegistry.create() fallback 到 DefaultController
```

### 11.3 SSE 事件序列化（`runtime_service.py`）

`serialize_harness_event` 加三个分支：

```python
if isinstance(event, ControllerStart):
    return {
        "controller_type": event.controller_type,
        "user_input": event.user_input,
    }
if isinstance(event, ControllerContinue):
    return {
        "controller_type": event.controller_type,
        "next_prompt": event.next_prompt,
    }
if isinstance(event, ControllerEnd):
    return {
        "controller_type": event.controller_type,
        "response": event.response,
        "time_taken": event.time_taken,
        "exceeded": event.exceeded,
        "interrupted": event.interrupted,
        "error": event.error,
    }
```

chat.py 里 `type(event).__name__` → SSE `event:` 行 → `"ControllerStart"` / `"ControllerContinue"` / `"ControllerEnd"`。自动通路。

### 11.4 Management API（`api/management.py`）

新增端点：

```
GET /api/v1/controllers

→ [
    {"value": "default", "display_name": "Standard", "display_name_zh": "标准模式"},
    {"value": "goal",     "display_name": "Goal",     "display_name_zh": "目标模式",
     "settings": [{"key": "max_goal_rounds", "type": "number", "default": 5}]},
    {"value": "timer",    "display_name": "Timer",    "display_name_zh": "计时模式",
     "settings": [{"key": "duration", "type": "string", "default": "30m",
                   "placeholder": "e.g. 30m, 1h, 300s"}]},
  ]
```

Controller 列表从 `ControllerRegistry.list_types()` + 配套 metadata 输出。gateway config 加 `goal_max_rounds`、`timer_default_duration`。

### 11.5 mh-local

复用 mh-gateway 的 runtime_service 和 chat，零改。

## 12. 前端改动（`web-frontend`）

### 12.1 SSE 事件类型（`types/index.ts`）

```typescript
export const SSE_EVENTS = {
  …,
  CONTROLLER_START:    "ControllerStart",
  CONTROLLER_CONTINUE: "ControllerContinue",
  CONTROLLER_END:      "ControllerEnd",
} as const;
```

### 12.2 Chat Store（`stores/chat.ts`）

```typescript
case SSE_EVENTS.CONTROLLER_START:
  // 可选 banner，或重置 pending 状态
  break;

case SSE_EVENTS.CONTROLLER_CONTINUE:
  // 向 messages 追加一条 role="user" 的自动消息
  if (!sessionMessagesMap[sid]) sessionMessagesMap[sid] = [];
  sessionMessagesMap[sid].push({
    id: `msg-auto-${Date.now()}`,
    role: "user",
    content: data.next_prompt,
    auto: true,
    orderedItems: [{ type: "content", text: data.next_prompt }],
  });
  break;

case SSE_EVENTS.CONTROLLER_END:
  flushImmediately(sid);
  finalizeStream(sid);
  break;
```

### 12.3 输入框组件

输入框旁加 controller 下拉菜单，从 `GET /api/v1/controllers` 加载。选 "timer" 时展开一个 duration 输入框。选择器的值作为 `ChatRequest.controller` 和 `ChatRequest.controller_config` 发送。

### 12.4 渲染层（可选）

自动消息（`auto === true`）渲染为灰显气泡。不改也能工作——只是一条普通 user 消息。

## 13. Controller 状态持久化

### 13.1 核心结论：v1 不额外持久化 Controller 状态

理由：

**会话 transcript 已经是完整状态。** 每轮 `agent.run()` **内部**自动调用 `memory.add_message(user_message(...))`——合成 prompt 被 Agent 写入 memory，不是 Controller 写的。Gateway `chat.py` 的 `finally` 块执行 `store.save_memory()`，所有消息（含自动 prompt）一起持久化。

**Goal 文本在 transcript 里。** `get_forward_messages()` 第一条 user message 就是原始目标。Judge 不需要额外存储。

**Round 计数不需要跨请求。** `max_goal_rounds` 是单次 HTTP 请求的安全上限。如果达到上限（`ControllerEnd(exceeded=True)`），用户再发一条带 `controller="goal"` 的消息——新 Controller 实例读 transcript 做 judge，新预算独立生效。

**Timer 的 elapsed 不需要跨请求。** 用户发 `controller="timer" duration="30m"`——这是一次 HTTP 请求内的限时竞价。请求结束（正常/中断/超时）就结束了。真想再跑 30 分钟，用户再发一条。

**中断恢复不走 Controller 状态。** 用户按停 → `AgentEnd(interrupted=True)` → 循环型 Controller 检测到 → `ControllerEnd(interrupted=True)`。Gateway `finally` 持久化 transcript。下次打开 session，看到完整历史；用户重新发送即可继续。

### 13.2 场景覆盖

| 场景 | transcript 里有 | 跨请求恢复方式 |
|---|---|---|
| Goal 正常完成 | 完整对话 + 最终结果 | 不需要——已完成 |
| Goal 达到 max_rounds | 完整对话 + `ControllerEnd(exceeded=True)` | 用户重新发 `controller="goal"` |
| Timer 时间到 | 完整对话 + 自动 prompt 记录 | 用户重新发 `controller="timer" duration=…` |
| 用户按停 | 对话到中断点 | 同上 |
| 连接断开 / crash | 到最后一次 `add_message` | 同上 |

### 13.3 v2 扩展预留：`memory._extra["controller_state"]`

如果后续 Controller 需要保存复杂中间产物（多阶段计划、子任务队列），写到 Memory 已有的 `_extra` 槽：

```python
memory._extra["controller_state"] = {
    "round": current_round,
    "goal_text": goal_text,
    "plan": [...],
}
```

`_extra` 随 `dump_memory()` / `load_memory()` 自动序列化，不需要新接口。

## 14. 实施阶段

### Phase 1: minimal-harness 核心（6 个文件）

| # | 文件 | 动作 | 内容 |
|---|---|---|---|
| 1 | `types.py` | 改 | `ControllerStart` / `ControllerContinue` / `ControllerEnd` dataclass + `ControllerEvent` 联合 |
| 2 | `agent/controller.py` | **新** | Controller 协议 + DefaultController + `_LoopingController` + GoalController + TimerController + `_parse_duration` / `_format_duration` 工具函数 |
| 3 | `agent/runtime.py` | 改 | `ControllerRegistry` + `run(controller_type=…, controller_config=…)` → `controller.execute(…)` |
| 4 | `agent/__init__.py` | 改 | 导出 |
| 5 | `test/test_controller.py` | **新** | DefaultController 透传 + GoalController 循环 + TimerController 时间逻辑 + judge 解析 + stop_event 传播 |
| 6 | `README.md` | 改 | Controller layer 一节 |

### Phase 2: gateway 接入（4 个文件）

| # | 文件 | 动作 | 内容 |
|---|---|---|---|
| 1 | `api/chat.py` | 改 | `ChatRequest.controller` + `controller_config` 字段，透传 runtime |
| 2 | `services/runtime_service.py` | 改 | `serialize_harness_event` 加 Controller 三个事件；`create_runtime` 注册 goal/timer controller |
| 3 | `api/management.py` | 改 | `GET /api/v1/controllers` 端点 |
| 4 | `config.py` | 改 | `goal_max_rounds`、`timer_default_duration` |

### Phase 3: mh-local（0 个文件）

复用 gateway runtime_service + chat 逻辑。

### Phase 4: 前端（3 个文件）

| # | 文件 | 动作 | 内容 |
|---|---|---|---|
| 1 | `types/index.ts` | 改 | SSE_EVENTS 加 ControllerStart/Continue/End |
| 2 | `stores/chat.ts` | 改 | handleSSEEvent 加三个 case |
| 3 | 聊天输入组件 | 改 | Controller 下拉选择器 + timer 时间输入（从 `/controllers` API 加载） |

## 15. 测试设计

```python
# test_controller.py

class FakeLLMProvider:
    """可编程模拟 LLM：chat() 返回预设内容列表。"""
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls: list[dict] = []

    async def chat(self, messages, tools=None, stop_event=None, **kwargs):
        self.calls.append({"messages": messages, "stop_event": stop_event})
        ...


class TestDefaultController:
    async def test_passthrough_yields_agent_events():
        ...

    async def test_agent_start_and_end_passed_through():
        """AgentStart/AgentEnd 被 DefaultController 原样透传。"""
        ...

    async def test_agent_exception_wrapped_in_controller_end():
        ...

    async def test_cancelled_error_sets_interrupted():
        ...


class TestGoalController:
    async def test_single_round_judge_says_done():
        ...

    async def test_two_rounds_judge_next_then_done():
        ...

    async def test_max_rounds_exceeded():
        ...

    async def test_agent_interrupted_stops_immediately():
        ...

    async def test_judge_parse_done_case_variants():
        ...

    async def test_judge_parse_next_format_variants():
        ...

    async def test_judge_error_defaults_to_stop():
        ...

    async def test_judge_call_receives_stop_event():
        """验证 _call_judge 把 stop_event 传给了 llm_provider.chat()。"""
        ...


class TestTimerController:
    async def test_elapsed_under_duration_continues():
        ...

    async def test_elapsed_exceeds_duration_stops():
        ...

    async def test_judge_returns_done_but_time_not_up_uses_forced_prompt():
        """Judge 说 DONE，但时间未到——用模板 prompt 强制继续。"""
        ...

    async def test_judge_returns_next_with_time_context():
        ...

    async def test_duration_parsing():
        """_parse_duration('30m') == 1800, '1h' == 3600, '300s' == 300, etc."""
        ...

    async def test_agent_error_stops_immediately():
        ...

    async def test_max_rounds_safety_cap():
        ...


class TestControllerRegistry:
    async def test_register_and_create():
        ...

    async def test_unknown_type_falls_back_to_default():
        ...

    async def test_list_types():
        ...
```

## 16. 设计决策记录

| 决策 | 理由 |
|---|---|
| Controller 不和 AgentMetadata 耦合 | Agent 对 Controller 零感知；Controller 是运行时选择 |
| per-request `controller` + `controller_config` | 用户每次输入可以选不同 Controller；参数随请求动态调整 |
| `ControllerEvent` 独立于 `AgentEvent` | Controller 是独立层，事件不应混入 Agent 事件联合 |
| 用 `controller_type` 字段扩展，不增新事件类 | 加 RetryController 等也不用动事件定义 |
| DefaultController 永远存在 | 框架里不存"无 Controller"的代码路径 |
| v1 不持久化 Controller 状态 | 会话 transcript 本身即状态；v2 用 `memory._extra` 预留 |
| Judge 安全默认 = DONE（停止） | 解析失败 / LLM 异常 → 宁可停也不烧钱循环 |
| `_evaluate()` 合并停止判断和 prompt 生成 | GoalController 一次 judge 同时出两个结论，避免重复 LLM 调用 |
| TimerController 的 forced prompt | Judge 说 DONE 但时间未到——不能停，模板 prompt "continue with remaining time" 兜底 |
| stop_event 透传到 judge 调用 | 修复"内部 LLM 调用不可中断"；`build_summarizer` 积债另修 |
