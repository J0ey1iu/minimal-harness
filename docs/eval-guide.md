# Evaluation Guide

评测模块（`minimal_harness.eval`）用于对单 Agent 进行批量效果评测。它提供并发执行、全链路事件采集、实时落盘和可视化 HTML 报告。

## 快速开始

```python
import asyncio
from minimal_harness import AgentMetadata
from minimal_harness.agent.registry import AgentRegistry
from minimal_harness.eval import EvalTaskConfig, run_evaluation

async def main():
    agent_registry = AgentRegistry()
    await agent_registry.register(AgentMetadata(
        name="my_agent",
        display_name="My Agent",
        description="My test agent",
        system_prompt="You are a helpful assistant.",
        agent_type="simple",
        tool_names=["calculator"],
    ))

    # ... also set up tool_registry and llm_provider_factory ...

    summary = await run_evaluation(
        agent_registry=agent_registry,
        tool_registry=tool_registry,
        llm_provider_factory=lambda: llm_provider,
        config=EvalTaskConfig(
            name="my-eval",
            description="Test my agent's performance",
            agent_metadata_id="my_agent",
            inputs=[
                "What is 2+2?",
                "Calculate 15 * 37",
            ],
            max_concurrency=4,
            output_dir="./eval_results",
        ),
    )
    print(f"Report: {summary.output_path}/report.html")

asyncio.run(main())
```

完整示例见 `examples/eval_demo.py`。

## 安装

评测模块是 minimal-harness 的内置模块，无需额外安装：

```python
from minimal_harness.eval import run_evaluation, run_evaluation_simple
from minimal_harness.eval import EvalTaskConfig, EvalSummary, EvalRunRecord
```

## API

### run_evaluation（基于 Registry）

这是推荐的入口方法。它通过注册表查找 AgentMetadata 和 ToolMetadata，自动解析工具和系统提示词。

```python
async def run_evaluation(
    *,
    agent_registry: AgentRegistryProtocol,
    tool_registry: ToolRegistryProtocol,
    llm_provider_factory: Callable[[], LLMProvider],
    config: EvalTaskConfig,
    on_run_complete: Callable[[EvalRunRecord], None] | None = None,
) -> EvalSummary
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `agent_registry` | `AgentRegistryProtocol` | Agent 元数据注册表 |
| `tool_registry` | `ToolRegistryProtocol` | Tool 元数据注册表 |
| `llm_provider_factory` | `Callable[[], LLMProvider]` | LLM Provider 工厂（每次运行调用） |
| `config` | `EvalTaskConfig` | 评测任务配置 |
| `on_run_complete` | `Callable` | 每轮完成后的回调（可选），可用于实时查看进度 |

### run_evaluation_simple（直接传入组件）

如果你已经构造好 `Tool` 对象和 `LLMProvider`，可以使用此方法跳过注册表查找。

```python
async def run_evaluation_simple(
    *,
    llm_provider: LLMProvider,
    tools: Sequence[Tool],
    system_prompt: str,
    config: EvalTaskConfig,
    on_run_complete: Callable[[EvalRunRecord], None] | None = None,
) -> EvalSummary
```

### EvalTaskConfig

```python
@dataclass
class EvalTaskConfig:
    name: str                                   # 评测名称（用于目录命名）
    description: str = ""                       # 描述
    agent_metadata_id: str = ""                 # Agent 标识（显示用）
    inputs: list[str] = []                      # 测试输入列表
    max_concurrency: int = 4                    # 最大并发数
    output_dir: str = "./eval_results"          # 输出根目录
    max_iterations: int = 20                    # Agent 最大迭代次数
    cost_per_million_input_tokens: float | None = None   # 输入令牌单价（可选）
    cost_per_million_output_tokens: float | None = None  # 输出令牌单价（可选）
```

### EvalSummary

```python
@dataclass
class EvalSummary:
    task_name: str
    description: str
    agent_metadata_id: str
    total_runs: int         # 总运行数
    completed: int           # 成功完成数
    failed: int              # 失败数
    interrupted: int         # 中断数
    total_time: float        # 总耗时（秒）
    avg_time: float          # 平均耗时（秒）
    total_tokens: int        # 总 token 数
    total_cost: float | None # 总费用（配置了单价时）
    runs: list[EvalRunRecord]
    output_path: str         # 输出目录路径
```

## 输出目录结构

```
eval_results/
└── {name}_{timestamp}/
    ├── config.json              # 评测配置文件副本
    ├── summary.json             # 汇总结果（持续更新）
    ├── report.html              # HTML 可视化报告（最终生成）
    └── runs/
        ├── {run_id}.jsonl       # 事件日志（JSON Lines，实时写入）
        ├── {run_id}_summary.json # 单轮结果摘要
        └── ...
```

每条 `jsonl` 文件包含该轮 Agent 运行的所有生命周期事件，每行一个 JSON 对象：

```json
{"event_type":"agent_start","timestamp":...,"data":{"user_input":...}}
{"event_type":"llm_start",  "timestamp":...,"data":{"messages":...,"tools":...}}
{"event_type":"llm_end",    "timestamp":...,"data":{"content":...,"usage":...}}
{"event_type":"tool_start", "timestamp":...,"data":{"tool_call":...}}
{"event_type":"tool_end",   "timestamp":...,"data":{"tool_call":...,"result":...}}
{"event_type":"agent_end",  "timestamp":...,"data":{"response":...,"time_taken":...}}
```

这种设计确保：即使评测过程中进程崩溃，已采集的数据最多只丢失最后一条事件。

## HTML 报告功能

生成的 `report.html` 是一个自含式文件（无外部依赖），包含以下功能：

| 功能 | 说明 |
|------|------|
| **汇总看板** | 显示完成/失败/中断数、平均耗时、总 token 数、预估费用 |
| **搜索过滤** | 按输入文本搜索，按状态筛选 |
| **排序** | 点击表头按任意列排序 |
| **Token 条形图** | 直观展示每轮的 input/output token 配比 |
| **展开详情** | 点击行展开查看 Agent 的最终 Response |
| **状态标签** | 颜色区分 completed / failed / interrupted |

## 实时进度回调

通过 `on_run_complete` 参数可以实时追踪评测进度：

```python
def on_progress(run: EvalRunRecord):
    print(
        f"[{run.status:>12}] {run.run_id}: "
        f"{run.input_text[:40]:40s} "
        f"{run.llm_call_count} LLM calls, "
        f"{run.token_usage.total_tokens if run.token_usage else 0} tokens"
    )

summary = await run_evaluation(
    ...,
    on_run_complete=on_progress,
)
```

## 成本追踪

在 `EvalTaskConfig` 中配置按百万 token 的价格：

```python
config = EvalTaskConfig(
    ...,
    cost_per_million_input_tokens=0.15,   # $0.15 / 1M input tokens
    cost_per_million_output_tokens=0.60,  # $0.60 / 1M output tokens
)
```

配置后，summary 中的 `total_cost` 和各 Run 的 `token_usage.total_cost` 会自动计算。

## 并发控制

`max_concurrency` 参数控制同时运行的 Agent 数量。评测模块使用 `asyncio.Semaphore` 实现：

```python
config = EvalTaskConfig(
    ...,
    max_concurrency=5,  # 同时最多运行 5 个 Agent
)
```

选择建议：
- API 速率限制低的模型（如 GPT-4）：`2-4`
- 速率限制高的模型（如 GPT-4o-mini）：`5-10`
- 本地模型或高配额 API：`10-20`

## 数据采集原理

评测模块通过框架内置的 `Middleware` 机制实现无侵入式数据采集：

```
Agent 执行循环               EvalCollector (Middleware)
─────────────────            ──────────────────────────
on_agent_start(user_input)   → 记录 agent_start + 时间戳
on_llm_start(messages)       → 记录 llm_start + 时间戳
on_llm_end(event)            → 记录 llm_end + token 用量 + 时间戳
on_tool_start(tool_call)     → 记录 tool_start + 时间戳
on_tool_end(tool_call, res)  → 记录 tool_end + 时间戳
on_tool_error(tool, error)   → 记录 tool_error + 时间戳
on_agent_end(event)          → 记录 agent_end + 时间戳
```

每条记录包含 `event_type`、`timestamp` 和序列化的 `data`，实时写入独立 JSONL 文件。

## 设计原则

1. **实时落盘**：每条事件立即 `write → flush → fsync`，异常崩溃最多丢失最后一条事件
2. **零侵入**：通过 `Middleware` 模式采集，不需要修改 Agent 或 Tool 代码
3. **自包含报告**：HTML 文件内嵌所有数据和样式，无需网络或额外工具即可查看
4. **可重放**：JSONL 格式的原始事件日志可用于后续重现和分析
