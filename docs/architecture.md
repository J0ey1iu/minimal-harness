# Architecture

## Overview

minimal-harness 采用双层抽象架构，自底向上分别为�?

- **Layer 1 �?核心抽象�?(Core Abstractions)**：定�?Agent、Tool、Memory、LLMProvider 等基础概念及其事件体系。该层不依赖任何具体应用，是整个系统的基石�?
- **Layer 2 �?面向服务�?(Service Abstractions)**：在 Layer 1 之上提供 Runtime（运行编排）、Registry（注册发现）、SessionStore（持久化）等面向运行时的服务能力。该层依�?Layer 1 �?Protocol，为上层提供更高阶的抽象�?

> **Layer 3 �?应用�?(Application)** 此前�?Textual TUI 客户端占据；
> �?0.7.0 起已抽离为独立包 [`mh-tui`](https://github.com/J0ey1iu/mh-tui)�?
> 另有一个基�?FastAPI 的云端分布式应用�?[`mh-gateway`](https://github.com/J0ey1iu/mh-gateway)
> 作为另一参考实现，�?mh-tui 在同一抽象层互为补充�?

```
 ┌──────────────────────────────────────────�?
 �? Layer 3: Applications (外部仓库)        �?
 �? mh-tui · mh-gateway · �?  �?
 ├──────────────────────────────────────────�?
 �? Layer 2: Service Abstractions           �?
 �? AgentRuntime · Registry<> · SessionStore �?
 ├──────────────────────────────────────────�?
 �? Layer 1: Core Abstractions              �?
 �? Agent · Tool · Memory · LLMProvider     �?
 �? Events (AgentEvent / ToolEvent)         �?
 └──────────────────────────────────────────�?
```

理想依赖方向�?*应用�?�?Layer 2 �?Layer 1**，且每层只依赖下层定义的 **Protocol**，不应依赖具体实现�?

---

## Layer 1: 核心抽象 (Core Abstractions)

### Agent Protocol

**定义位置**: `src/minimal_harness/agent/protocol.py`

```python
class Agent(Protocol):
    def run(
        self,
        user_input: Iterable[ExtendedInputContentPart],
        stop_event: asyncio.Event | None = None,
        memory: Memory | None = None,
        tools: Sequence[Tool] | None = None,
        system_prompt: str = "",
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AgentEvent]: ...
```

Agent 是核心执行单元。其 `run()` 方法接收用户输入、停止信号、记忆、工具、系统提示、上下文（用�?locale 等运行时信息）和额外关键字参数（�?`llm_kwargs`），通过 `AsyncIterator[AgentEvent]` 对外产出事件流。事件驱动模型使�?Agent 的执行过程对调用方完全透明�?

 当前内置两种实现�?

- **`SimpleAgent`** (`agent/simple.py`)：标�?agent 循环，无消息压缩
- **`CompactionAgent`** (`agent/compacting.py`)：复�?`SimpleAgent` 的执行循�?
  （基�?`BaseAgent` 抽出共享骨架），通过覆写 `_post_llm_response` 钩子插入
  压缩逻辑；通过 `agent_type="compacting"` 启用，由 `AgentRuntime`
  通过内置的 `build_summarizer` 构建 summarizer

两者的执行循环一致：

1. 追加用户消息�?Memory
2. 调用 LLMProvider.chat() 进行流式推理
3. 流式产出 LLMChunk 事件
4. LLM 完成后，若存�?tool_calls，进入工具执行阶�?
5. 将工具执行结果写�?Memory，继续下一轮迭�?
6. 最大迭代次数通过构造函�?`max_iterations` 参数注入�?.7.0：不再由 `Settings` 提供�?

`CompactionAgent` 在第 4 步之后多一段（�?4.5 步）�?
1. 先把 LLM �?reasoning + assistant 回复写进 Memory，并 yield `MessageEvent` 通知前端
   （这是当轮的主输出，必须先于任何 housekeeping 步骤到达用户�?
2. �?`LLMEnd.usage["prompt_tokens"]` 超过 `CompactionConfig.prompt_token_threshold`�?
   调用 `Memory.compact()` 流式折叠旧消�?

设计�?*没有**任何"保护刚加�?assistant 不被折掉"的特殊逻辑：`Memory.compact()`
折叠 `msgs[0 : len-keep_recent]`，如�?assistant 落在 fold 区域就一起折。前端已经通过
`MessageEvent(assistant)` 看到原文，buffer 里的折叠只影响下一�?LLM 调用看到�?
`get_forward_messages()`。详�?§1.3 Memory �?*Frontends vs. Model View*�?

### Tool Protocol

**定义位置**: `src/minimal_harness/tool/base.py`

```python
class Tool(Protocol):
    name: str
    description: str
    parameters: dict
    display_name: str
    display_name_locale: dict[str, str] | None
    description_locale: dict[str, str] | None
    def to_schema(self) -> dict: ...
    def to_anthropic_schema(self) -> dict[str, Any]: ...
    def execute(
        self, args: dict[str, Any], tool_call: ToolCall,
        stop_event: asyncio.Event | None,
    ) -> AsyncIterator[ToolEvent]: ...
    def resolve_display_name(self, locale: str = "") -> str: ...
    def resolve_description(self, locale: str = "") -> str: ...
```

Tool 提供 LLM 可调用的外部能力。关键字段和方法�?

- `display_name` / `display_name_locale` �?人类可读名称及其国际化映�?
- `description_locale` �?描述的国际化映射
- `resolve_display_name(locale)` / `resolve_description(locale)` �?�?locale 解析展示文本
- `to_schema()` / `to_anthropic_schema()` �?将工具描述导出为 OpenAI / Anthropic �?function calling schema
- `execute()` �?执行工具，通过 `AsyncIterator[ToolEvent]` 产出 `ToolStart �?ToolProgress* �?ToolEnd` 事件序列

唯一内置实现 **`StreamingTool`** (`tool/base.py:65`) 将任�?`StreamingToolFunction` 包装�?Tool。其 `execute()` 会自动产出生命周期事件并处理异常/取消�?

内置工具�?.7.0：移�?SDK，移�?[`mh-tui`](https://github.com/J0ey1iu/mh-tui) �?`mh_tui.built_in`）：
- **`bash`** �?shell 命令执行，支持超时、工作目录、流式输�?
- **`local_file_operation`** �?本地文件读写、patch、删�?

`StreamingToolFunction` 类型�?`Callable[..., AsyncIterator[Any]]`，简单直观。工具函数只需是一个返�?`AsyncIterator` 的可调用对象�?

### Memory Protocol

**定义位置**: `src/minimal_harness/memory.py`

```python
class Memory(Protocol):
    @property
    def memory_id(self) -> str: ...
    @property
    def title(self) -> str | None: ...
    @property
    def agent_name(self) -> str: ...
    @property
    def created_at(self) -> str: ...

    async def add_message(self, message: Message) -> None: ...
    def get_all_messages(self) -> list[Message]: ...
    def get_forward_messages(self) -> list[Message]: ...
    def clear_messages(self) -> None: ...
    def set_message_usage(self, usage: TokenUsage) -> None: ...
    def get_message_usage(self) -> TokenUsage: ...
    def dump_memory(self) -> MemoryData: ...
    def load_memory(self, data: MemoryData) -> None: ...
    def get_persisted_count(self) -> int: ...
    def get_new_messages(self) -> list[Message]: ...
    def mark_all_persisted(self) -> None: ...
    def set_persisted_count(self, count: int) -> None: ...
    def compact(
        self,
        summarizer: Callable[[list[Message], str | None], AsyncIterator[str]],
        keep_recent: int,
        prompt_tokens: int = 0,
    ) -> AsyncIterator[CompactionEvent]: ...
```

Memory 维护对话历史（纯消息容器）。`compact()` 是流式异步生成器：在
`LLMEnd.usage["prompt_tokens"]` 超过阈值时�?`CompactionAgent` 调用�?
透传 `CompactionStart` / `CompactionChunk` / `CompactionEnd` 事件；折�?
后的消息被替换为 `[summary, *recent_tail]`，`_forward_offset` 保持 0
（summary �?compacted 历史的自然起点），`get_forward_messages()` �?
summary 重新投影�?`role="assistant"` 喂给 LLM。`dump_memory` /
`load_memory` 自动�?`_extra` 持久化�?

**Summarizer chat composition** (`agent/_compaction.py`): 内置
`build_summarizer(llm_provider, system_prompt)` 透传给 LLM 的 chat
payload 结构是 `[system_prompt, ...history, summary_request]` —
agent 的 system prompt 保持不变，需要折叠的消息作为对话历史传
入（prior `role="compaction"` 会被重新投影成 `assistant` turn），
末尾追加一条 user message（`SUMMARY_REQUEST`）让模型总结。这
避免了过去 `json.dumps` transcript + 单独 system prompt 的做法，
让模型在熟悉的角色里看完整对话再总结。

#### Frontends vs. Model View

`MessageEvent` 流（前端订阅）和 `get_forward_messages()`（发�?LLM �?
视图）是**解�?*的两条路�?

| 路径 | 看到什�?|
|------|---------|
| `MessageEvent` �?| 原始消息：reasoning、user、assistant、compaction (summary)。前端按发生顺序展示�?|
| `get_forward_messages()` | 紧凑视图：filter �?reasoning；把 `compaction` 重新投影�?`assistant`。下一�?LLM 调用只看�?buffer 的紧凑版本�?|

这意味着 compact 完全可能把刚加的 assistant 折进 summary（如�?`keep_recent`
不够大）。前端不受影响——用户已经通过 `MessageEvent(assistant)` 看到原文�?
buffer 的不变量�?回合结束后低于阈�?（前提是 compact 成功）�?

### Session Protocol (moved to mh-gateway)

> **0.7.0:** `Session` / `SimpleSession` / `SessionSummary` were
> moved to `mh_gateway.database._session`. The SDK
> no longer carries the `Session` abstraction �?`AgentRuntime` only
> needs a store that returns a `Memory`.

**定义位置**: `src/mh_gateway/database/_session.py`

```python
class Session(Protocol):
    @property
    def session_id(self) -> str: ...
    @property
    def memory_id(self) -> str: ...
    @property
    def agent_name(self) -> str: ...
    @property
    def display_name_locale(self) -> str | None: ...
    @property
    def user_id(self) -> str: ...
    @property
    def scenario_id(self) -> str | None: ...
    @property
    def title(self) -> str | None: ...
    @property
    def created_at(self) -> str: ...
    @property
    def memory(self) -> Memory: ...

    async def add_message(self, message: Message) -> None: ...
    def get_all_messages(self) -> list[Message]: ...
    def get_forward_messages(self) -> list[Message]: ...
    # ... 继承 Memory 的全部消息方�?
```

Session = Memory（全部消息方法） + 身份字段（user_id, scenario_id）。L2 �?Store 操作的是 Session�?
而非裸的 Memory。应用层可以直接使用 Session �?`user_id`/`scenario_id`，无需通过 `extra` 或反射。消息类型（`Message`）：

| 类型 | 角色 | 说明 |
|------|------|------|
| `SystemMessage` | `system` | 系统提示 |
| `UserMessage` | `user` | 用户输入（文�?图片/文件�?|
| `AssistantMessage` | `assistant` | LLM 回复（含 tool_calls�?|
| `ToolMessage` | `tool` | 工具执行结果 |
| `ReasoningMessage` | `reasoning` | LLM 推理链（�?DeepSeek R1�?|

`get_forward_messages()` 过滤�?`reasoning` 消息（不发送给 LLM），`dump_memory()` 输出完整的可序列化状态用于持久化�?

当前实现�?
- **`ConversationMemory`** (`memory.py`) �?纯内存实现，支持 JSON 序列�?反序列化
- **`SimpleSession`** (`mh_gateway/database/_session.py`) �?0.7.0 起从 SDK 迁出；实�?`Session` Protocol，包�?`ConversationMemory`，包含身份字�?

### LLMProvider Protocol

**定义位置**: `src/minimal_harness/llm/llm.py`

```python
class LLMProvider(Protocol):
    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool],
        stop_event: asyncio.Event | None = None,
        **kwargs: Any,
    ) -> Stream[LLMChunkDelta]: ...
```

LLMProvider 负责与外�?LLM API 交互。`chat()` 返回 `Stream[LLMChunkDelta]`，一个自定义的异步迭代器包装，在流耗尽时会持有最终的 `LLMResponse`（含 `content`、`reasoning_content`、`tool_calls`、`finish_reason`、`usage`）�?

当前实现�?
- **`OpenAILLMProvider`** (`llm/openai.py`) �?基于 `AsyncOpenAI` SDK
- **`AnthropicLLMProvider`** (`llm/anthropic.py`) �?基于 `AsyncAnthropic` SDK

两者均将原生事件统一转换�?`LLMChunkDelta`，实�?provider-agnostic 的流式输出�?

### Middleware 钩子系统

**定义位置**: `src/minimal_harness/agent/middleware.py`

`Middleware` 基类提供一组可选的异步钩子，供应用层在 Agent 生命周期中注入自定义逻辑�?

| 方法 | 触发时机 | 用�?|
|------|---------|------|
| `on_agent_start(user_input)` | Agent 开始处理输�?| 日志、审�?|
| `on_agent_end(event)` | Agent 结束运行 | 统计、追�?|
| `on_llm_start(messages, tools)` | 每次 LLM 调用�?| 成本追踪、内容过�?|
| `on_llm_end(event)` | LLM 调用结束�?| 记录 token 用量 |
| `on_compaction_start(event)` | `Memory.compact()` 触发压缩前（�?`CompactionAgent`�?| 折叠进度条、指标埋�?|
| `on_compaction_end(event)` | `Memory.compact()` 结束（仅 `CompactionAgent`�?| 折叠耗时/摘要长度埋点 |
| `on_tool_start(tool_call)` | 单工具执行前 | 权限检�?|
| `on_tool_end(tool_call, result)` | 工具成功返回�?| 结果审查 |
| `on_tool_error(tool_call, error)` | 工具抛出异常�?| 错误监控 |
| `should_allow_tool(tool_call)` | 工具执行决策�?| 返回 `bool` 或拒绝理由字符串 |
| `on_error(error)` | 未捕获异常时 | 兜底日志 |

`SimpleAgent` �?`CompactionAgent` 在关键节点调用这些钩子；`EvalCollector` �?Middleware 的典型实现，用于全链路数据采集�?

### 事件体系

**定义位置**: `src/minimal_harness/types.py`

所有事件均�?`@dataclass`，通过 `AgentEvent` 联合类型统一�?

```
AgentEvent (Union)
├── AgentStart             # 运行开始，携带 user_input
├── LLMStart               # LLM 调用开始，携带 messages & tools
├── LLMChunk               # 流式 LLM 输出�?(包装 LLMChunkDelta)
├── LLMEnd                 # LLM 调用结束，携�?content / reasoning_content / tool_calls / usage
├── CompactionStart        # Memory.compact() 触发，携�?dropped_count / existing_summary / keep_recent / prompt_tokens（仅 CompactionAgent�?
├── CompactionChunk        # summarizer 流式片段，携�?delta / accumulated（仅 CompactionAgent�?
├── CompactionEnd          # 压缩结束，携�?summary / dropped / new_offset / duration / error? 失败�?summary="" (避免把部分流式文本当成有效折�?
├── MessageEvent (助理/reasoning/compaction)  # 当轮新增消息的原始视�?(role=compaction 表示折叠摘要)
├── MemoryUpdate           # token 用量更新
├── ExecutionStart         # 工具批量执行开�?
�?  ├── ToolStart          # 单个工具开�?
�?  ├── ToolProgress       # 单个工具进度
�?  └── ToolEnd            # 单个工具结束
├── ExecutionEnd           # 工具批量执行结束，携�?results
└── AgentEnd               # 运行结束，携�?response / time_taken / exceeded / interrupted

ToolEvent (Union) = ToolStart | ToolProgress | ToolEnd
CompactionEvent (Union) = CompactionStart | CompactionChunk | CompactionEnd
```

事件流具有层级结构：Agent 包含 LLM 调用，LLM 调用可能触发工具执行，工具执行可产生进度事件。每层以 `Start/End` 括起�?

`LLMChunkDelta` 是关键的数据类，作为 provider-agnostic 的流式增量：

```python
@dataclass
class LLMChunkDelta:
    content: str | None = None          # 正文增量
    reasoning: str | None = None        # 推理链增�?(�?DeepSeek R1)
    tool_calls: list[ToolCallDelta] | None = None  # 工具调用增量
```

### 其他 Layer 1 类型

- **`TokenUsage`** (`types.py:44`) �?`{prompt_tokens, completion_tokens, total_tokens}` TypedDict
- **`ToolCall`** (`types.py:32`) �?`{id, type, function: {name, arguments}}` TypedDict
- **`ToolCallFunction`** (`types.py:25`) �?`{name, arguments}` TypedDict
- **`ToolCallDelta`** (`types.py:70`) �?流式工具调用增量 `{index, id, name, arguments}`
- **`Stream[T]`** (`llm/llm.py:49`) �?泛型异步迭代器包装，捕获最�?LLMResponse
- **`InputContentConversionFunction`** (`agent/protocol.py:8`) �?输入转换回调类型

---

## Layer 2: 面向服务抽象 (Service Abstractions)

### AgentRuntime / AgentRuntimeProtocol

**定义位置**: `src/minimal_harness/agent/runtime.py`

```python
@runtime_checkable
class AgentRuntimeProtocol(Protocol):
    agent_registry: AgentRegistryProtocol
    session_store: MemoryStoreProtocol  # 0.7.0: was SessionStoreProtocol
    tool_registry: ToolRegistryProtocol

    async def run(
        self,
        user_input: Iterable[ExtendedInputContentPart],
        agent_metadata_id: str,
        memory_id: str,
        agent_type: str | None = None,
        tool_names: list[str] | None = None,
        context: dict[str, Any] | None = None,
        llm_kwargs: dict[str, Any] | None = None,
    ) -> tuple[asyncio.Task, asyncio.Event, asyncio.Queue[AgentEvent | None]]: ...
```

`AgentRuntime` �?Layer 2 的核心编排器。它的职责是�?

1. 通过 `AgentRegistry` 查找 Agent 元数据（名称、系统提示、工具列表）
2. 通过 `MemoryStoreProtocol` 获取 Memory 实例�?.7.0：之前是 `SessionStoreProtocol`�?
3. 通过 `ToolRegistry` �?`ToolFactory` 解析并实例化工具列表
4. 通过 `LLMProviderFactory` 创建 LLM Provider
5. 创建 Agent 实例并调用其 `run()` 方法（传�?`middleware` 链、`context`、`llm_kwargs`�?
6. 返回 `(Task, Event, Queue)` 三元组供调用方驱动执�?

构造函数接受以下可选参数：

- `agent_factory` �?自定�?Agent 创建工厂（默�?`DefaultAgentFactory`�?
- `tool_factory` �?自定�?Tool 创建工厂（默�?`DefaultToolFactory`�?
- `middleware` �?Middleware 实例序列，传�?Agent
- `agent_driver_factories` �?远程 Agent 驱动工厂字典
- `tool_executor_factories` �?远程 Tool 执行器工厂字�?
- `llm_provider_resolver` �?必选，接收 `AgentMetadata` 返回 `LLMProvider` 的回�?

此外，`register_runtime_tools()` 是一个独立函数，注入 `handoff` �?`discover_agents` 两个运行时工具到 ToolRegistry，实现多 Agent 协作能力�?*�?0.7.0 起该函数已迁�?SDK，托管在 [`mh-tui`](https://github.com/J0ey1iu/mh-tui) 包中**（`mh_tui.runtime_tools.register_runtime_tools`）。应用层（TUI 或自建服务）需显式 `pip install mh-tui` 后再调用此函数�?

`handoff` 工具递归调用 `AgentRuntime.run()` 创建子任务；`discover_agents` 工具�?Registry 读取可用 Agent 列表�?

注意：`AgentRuntime.run()` 现已支持 `context` 参数（locale 等运行时上下文传播）�?`llm_kwargs` 参数（透传�?LLM SDK，如 `temperature`、`max_tokens`、`reasoning_effort` 等）�?

### Registry 体系

**`Registry[T]`** (`registry.py`) �?泛型注册基底，提供：

- `_register(name, item)` �?注册条目
- `unregister(name)` �?注销条目
- `get(name)`, `get_all()`, `names()`, `clear()` �?查询操作
- `add_listener(fn)` / `remove_listener(fn)` �?观察者模式通知

基于此派生出两种注册表：

**`ToolRegistry`** (`tool/registry.py`) �?继承 `Registry[ToolMetadata]`，存储工具元数据而非可执行对�?

```python
class ToolRegistry(Registry[ToolMetadata]):
    def register(self, metadata: ToolMetadata) -> None: ...
    def register_from_binding(self, name, description, parameters, binding, ...) -> None: ...
```

**`AgentRegistry`** (`agent/registry.py`) �?继承 `Registry[AgentMetadata]`，与 ToolRegistry 完全对称

```python
class AgentRegistry(Registry[AgentMetadata]):
    def register(self, metadata: AgentMetadata) -> AgentMetadata: ...
```

### Factory 体系

ToolRegistry 只存元数据，执行时由 **ToolFactory** 创建可执行的 `Tool` 对象�?

**`ToolFactory`** (`tool/factory.py`) �?Protocol

```python
class ToolFactory(Protocol):
    def create(self, metadata: ToolMetadata) -> Tool: ...
```

**`DefaultToolFactory`** �?处理所有内�?Binding 类型的默认实现，支持自定�?`RemoteToolExecutor` 注册�?

```python
class DefaultToolFactory:
    def __init__(self, executor_factories: dict[str, ToolExecutorFactory] | None = None): ...
    def create(self, metadata: ToolMetadata) -> Tool: ...
```

四种 Binding 类型的映射：

| Binding | 创建的可执行�?| 说明 |
|---------|---------------|------|
| `LocalToolBinding(fn=...)` | `StreamingTool(fn, ...)` | 本地函数 |
| `ExternalScriptToolBinding(script_path=...)` | `StreamingTool(fn=ExternalToolWrapper(...))` | 子进程脚�?|
| `RemoteToolBinding(url=..., driver=...)` | `RemoteTool(executor=...)` | HTTP SSE 远程调用 |
| `None` (built-in via `collect_builtin_tools`) | `StreamingTool` | 通过 `getattr(tool, "fn")` 获取 |

**`AgentMetadata.binding`** 控制 Agent 的创建方式：

| binding �?| 创建�?Agent | 说明 |
|-----------|-------------|------|
| `LocalAgentBinding()` (�?`None`) | �?`metadata.agent_type` 决定（`"simple"` �?`SimpleAgent`，`"compacting"` �?`CompactionAgent`，或自定�?AgentFactory�?| 本地执行 |
| `RemoteAgentBinding(url=..., driver=...)` | `RemoteAgent(driver=...)` | HTTP SSE 远程调用 |

`"compacting"` 派发：与 `"simple"` 共用本地 `LocalAgentBinding` 通路�?
`DefaultAgentFactory._local_agent_factories` 已预注册 `CompactingAgentFactory`�?
`CompactionConfig`（`summarizer` + `prompt_token_threshold` + `keep_recent`�?
必须由调用方在构�?`AgentRuntime` 时通过 `compaction_config=...` 注入�?
�?`agent_type="compacting"` 但未提供 config，工厂构造时直接�?`ValueError`�?

`AgentRuntime` 通过 `agent_driver_factories` 字典�?`driver` 名查�?`RemoteAgentDriverFactory`。默认实�?`SSEAgentDriver` 通过 HTTP POST + SSE 流与远程 Agent 服务通信�?

### Tool 快捷注册

**`tool/registration.py`** 提供了两种快捷注册方式：

- **`@register_tool()` 装饰�?* �?在工具函数上标注，可立即注册或延迟到 `register_decorated_tools()` 统一注册
- **`register_from_binding()`** �?`ToolRegistry` 方法，直接从参数创建 `ToolMetadata` 并注册，替换了旧�?`register_external_tool()`�?

```python
await registry.register_from_binding(
    name="my_tool",
    description="Does something",
    parameters={"type": "object", "properties": {}},
    binding=ExternalScriptToolBinding(script_path="/path/to/tool.py"),
)
```

### ToolRegistryProtocol / AgentRegistryProtocol

**定义位置**: `tool/base.py` / `agent/registry.py`

两个协议均标记为 `@runtime_checkable`，分别定义了 Tool �?Agent 的注册发现接口。它们的 CRUD 方法集高度相似（register / unregister / get / get_all / names / clear），但签名的差异导致无法用单一泛型协议统一�?

- `ToolRegistryProtocol.register(metadata: ToolMetadata)` �?接受 ToolMetadata 对象
- `AgentRegistryProtocol.register(metadata: AgentMetadata)` �?接受 AgentMetadata 对象（统一模式�?
- `ToolRegistryProtocol` 额外暴露 `register_from_binding()` 快捷方法
- `AgentRegistryProtocol` 额外暴露 `add_listener` / `remove_listener`（ToolRegistryProtocol 通过基类 `Registry[T]` 也具备监听器能力�?

### MemoryStoreProtocol

**定义位置**: `src/minimal_harness/memory.py`

```python
@runtime_checkable
class MemoryStoreProtocol(Protocol):
    """Minimal persistence contract: resolve a ``Memory`` by ID.

    The SDK's AgentRuntime only needs ``get_session()``. The richer
    Session-shaped store (with user_id, scenario_id, ...) lives in
    mh-gateway.database.
    """

    async def get_session(self, session_id: str) -> Memory | None: ...
```

> **0.7.0 change:** The previous `SessionStoreProtocol` (which
> returned identity-rich `Session` objects) has been replaced by this
> thinner `MemoryStoreProtocol`. The full Session contract
> (`user_id`, `scenario_id`, `display_name_locale`, `title`,
> `db_id`, persistence methods) lives in
> `mh_gateway.database._session` /
> `_memory_store`. mh-tui ships its own copy in
> `mh_tui._session_types`.
        display_name_locale: str | None = None,
    ) -> Session: ...
    async def get_session(self, session_id: str) -> Session | None: ...
    async def save_memory(
        self, memory: Memory, session_id: str, extra: dict[str, Any] | None = None
    ) -> None: ...
    async def delete_session(self, session_id: str) -> bool: ...
    async def list_sessions(self) -> list[SessionSummary]: ...
    async def list_user_sessions(
        self, user_id: str, scenario_id: str | None = None
    ) -> list[SessionSummary]: ...
    async def get_session_messages(self, session_id: str) -> list[dict]: ...
    def get_messages_as_items(self, session: Session) -> list[dict]: ...
```

`SessionStoreProtocol` �?Session 持久化的抽象接口。SDK 不再内建具体实现；参考实现见 [`mh-tui`](https://github.com/J0ey1iu/mh-tui) 中的 `JsonlSessionStore`（`~/.minimal_harness/sessions/`）和 [`mh-gateway`](https://github.com/J0ey1iu/mh-gateway) 中的 `SqliteSessionStore`（SQLite 后端）�?

`ConversationSession` �?L3 的运行时包装：持�?L2 `Session` 实体（含 identity 和消息）�?
叠加运行控制信息（stop_event、agent 绑定、工具列表）�?

### Settings

**定义位置**: `src/minimal_harness/settings.py`

```python
class Settings:
    @classmethod
    def model(cls) -> str: ...
    @classmethod
    def max_iterations(cls) -> int: ...
    @classmethod
    def base_url(cls) -> str: ...
    @classmethod
    def api_key(cls) -> str: ...
    @classmethod
    def theme(cls) -> str: ...
```

> **0.7.0 change:** `Settings` has been removed. The `MH_*` env vars
> are now read directly by each consumer (see
> `mh-tui.config.defaults` and `mh-service-kit.logging_setup`).
> The SDK no longer reads env vars at all.

---

## Layer 3: 应用�?�?外部仓库

> �?0.7.0 �?Layer 3 不再位于本仓库。完整数据流、组件清单与
> `ConversationSession` 等运行时会话模型详见�?
>
> - [`mh-tui`](https://github.com/J0ey1iu/mh-tui) �?Textual TUI 客户�?
> - [`mh-gateway`](https://github.com/J0ey1iu/mh-gateway) �?FastAPI 网关

---

## Protocol 全集

| Protocol | 层级 | 文件 | 标记 |
|----------|------|------|------|
| `Agent` | Layer 1 | `agent/protocol.py` | �?|
| `LLMProvider` | Layer 1 | `llm/llm.py` | �?|
| `Tool` | Layer 1 | `tool/base.py` | �?|
| `Memory` | Layer 1 | `memory.py` | �?|
| `Middleware` | Layer 1 | `agent/middleware.py` | �?|
| `Memory` | Layer 1 | `memory.py` | `@runtime_checkable` |
| `MemoryStoreProtocol` | Layer 1 | `memory.py` | `@runtime_checkable` |
| `RegistryProtocol[T]` | Layer 1 | `registry.py` | `@runtime_checkable` |
| `ToolRegistryProtocol` | Layer 1 | `tool/registry.py` | `@runtime_checkable` |
| `AgentRegistryProtocol` | Layer 1 | `agent/registry.py` | `@runtime_checkable` |
| `AgentRuntimeProtocol` | Layer 1 | `agent/runtime.py` | `@runtime_checkable` |
| `ToolFactory` | Layer 1 | `tool/factory.py` | `@runtime_checkable` |
| `ToolExecutorFactory` | Layer 1 | `tool/factory.py` | `@runtime_checkable` |
| `RemoteAgentDriver` | Layer 1 | `agent/driver.py` | `@runtime_checkable` |
| `RemoteAgentDriverFactory` | Layer 1 | `agent/driver.py` | �?|
| `RemoteToolExecutor` | Layer 1 | `tool/remote.py` | `@runtime_checkable` |
| `AgentFactory` | Layer 1 | `agent/factory.py` | `@runtime_checkable` |
| `LocalAgentFactory` | Layer 1 | `agent/factory.py` | �?|
| `Session`, `SessionSummary`, `SimpleSession`, `SessionStoreProtocol` | **moved to `mh_gateway.database`** | �?| �?|
| `RegistryProvider`, `MetadataManager`, `ToolProvider` | **moved to `mh_gateway.adapters`** | �?| �?|

### 工厂类型别名

| 类型 | 定义 | 文件 |
|------|------|------|
| `LLMProviderFactory` | `Callable[[], LLMProvider]` | `llm/llm.py` |
| `MemoryFactory` | `Callable[[], Memory]` | `memory_store.py` |
| `AgentFactory` | `Protocol: create(metadata: AgentMetadata) -> Agent` | `agent/factory.py` |
| `ToolFactory` | `Protocol: create(metadata: ToolMetadata) -> Tool` | `tool/factory.py` |
| `ToolExecutorFactory` | `Protocol: create(binding: RemoteToolBinding) -> RemoteToolExecutor` | `tool/factory.py` |
| `RemoteToolExecutor` | `Protocol: async execute(...)` | `tool/remote.py` |
| `RemoteAgentDriverFactory` | `Protocol: create(binding: RemoteAgentBinding) -> RemoteAgentDriver` | `agent/driver.py` |

### Protocol 实现关系

```
Agent ◄────────── SimpleAgent
              ◄─── CompactionAgent (structural clone + Memory.compact() hook)
              ◄─── RemoteAgent (delegates to RemoteAgentDriver)
LLMProvider ◄──── OpenAILLMProvider
         ◄──── AnthropicLLMProvider
Tool ◄─────────── StreamingTool
              ◄─── RemoteTool
Memory ◄───────── ConversationMemory
Session ◄──────── SimpleSession (with identity)
Middleware ◄───── EvalCollector
RegistryProtocol[T] ◄── Registry[T]
ToolRegistryProtocol ◄── ToolRegistry(Registry[ToolMetadata])
AgentRegistryProtocol ◄── AgentRegistry(Registry[AgentMetadata])
SessionStoreProtocol ◄── (no SDK-shipped impl �?see mh-tui's `JsonlSessionStore` or orchestration's `SqliteSessionStore`)
AgentRuntimeProtocol ◄── AgentRuntime
ToolFactory ◄──── DefaultToolFactory
ToolExecutorFactory ◄── DefaultToolExecutorFactory
RemoteToolExecutor ◄── SSEToolExecutor
RemoteAgentDriver ◄─── SSEAgentDriver
RemoteAgentDriverFactory ◄── DefaultAgentDriverFactory
AgentFactory ◄──── DefaultAgentFactory
LocalAgentFactory ◄── DefaultSimpleAgentFactory
                ◄── CompactingAgentFactory
```

---

## 跨层导入违规全景（已全部解决�?

> **0.7.0 调整**：本节为 0.5�?.6 期间的历史审计记录。表中第 7�?1�?7�?8
> 行原指向 `client/built_in/`，该目录已于 0.7.0 抽离至独立包 [`mh-tui`](https://github.com/J0ey1iu/mh-tui)�?
> 路径�?mh-tui 中仍然存在但归属已变，故下表保留为历史记录不再追加新行�?

下表汇总了已修复的所有违反分层依赖方向的导入（按严重程度排序）：

| # | 文件 (当前层级) | 导入目标 | 目标层级 | 方向 | 严重�?| 状�?|
|---|---------------|---------|---------|------|-------|------|
| 1 | `agent/simple.py` (L1) | `settings.Settings` | L2 | L1→L2 | P0 | �?已修�?|
| 2 | `llm/openai.py` (L1) | `settings.Settings` | L2 | L1→L2 | P0 | �?已修�?|
| 3 | `llm/anthropic.py` (L1) | `settings.Settings` | L2 | L1→L2 | P0 | �?已修�?|
| 4 | `agent/runtime.py` (L2) | `agent/simple.SimpleAgent` | L1 concrete | L2→L1 concrete | P0 | �?已修�?|
| 5 | `agent/runtime.py` (L2) | `tool/base.StreamingTool` | L1 concrete | L2→L1 concrete | P0 | �?已修�?|
| 6 | `memory_store.py` (L2) | `memory.ConversationMemory` | L1 concrete | L2→L1 concrete | P0 | �?已修�?|
| 7 | `client/built_in/context.py` (L3) | `llm.OpenAILLMProvider, AnthropicLLMProvider` | L1 concrete | L3→L1 concrete | P0 | �?已修�?|
| 8 | `client/built_in/app.py` (L3) | `tool.built_in.bash.get_tools` | L1 concrete | L3→L1 concrete | P0 | �?已修�?|
| 9 | `client/built_in/app.py` (L3) | `tool.built_in.local_file_operation.get_tools` | L1 concrete | L3→L1 concrete | P0 | �?已修�?|
| 10 | `client/built_in/config/tools.py` (L3) | `tool.built_in.bash.get_tools` | L1 concrete | L3→L1 concrete | P0 | �?已修�?|
| 11 | `client/built_in/config/tools.py` (L3) | `tool.built_in.local_file_operation.get_tools` | L1 concrete | L3→L1 concrete | P0 | �?已修�?|
| 12 | `agent/__init__.py` (L1 facade) | `agent/registry.AgentRegistry` | L2 | L1 re-export L2 | P1 | �?已修�?|
| 13 | `agent/__init__.py` (L1 facade) | `agent/runtime.AgentRuntime` | L2 | L1 re-export L2 | P1 | �?已修�?|
| 14 | `tool/__init__.py` (L1 facade) | `tool/registry.ToolRegistry` | L2 | L1 re-export L2 | P1 | �?已修�?|
| 15 | `client/__init__.py` (L3) | `types.*` (event types) | L1 | L3→L1 | P2 | �?保留 |
| 16 | `client/events.py` (L3) | `types.AgentEvent` | L1 | L3→L1 | P2 | �?已清�?|
| 17 | `client/built_in/display.py` (L3) | `types.*` (event types) | L1 | L3→L1 | P2 | �?保留 |
| 18 | `client/built_in/actions/sessions.py` (L3) | access `_ctrl._sessions` | L2 internal | L3→L2 internal | P2 | �?已修�?|

---

## 解决方案摘要

以下问题均已修复，保留原始问题描述和改动思路作为历史记录�?

### P0: 分层方向违规 �?已全部修�?

#### 问题 1: Layer 1 实现类反向依�?`Settings`

**状�?*: �?已修�?

**修复**: `SimpleAgent.__init__()` �?`max_iterations` 改为必选参数；`OpenAILLMProvider. __init__()` / `AnthropicLLMProvider.__init__()` �?`model` 改为必选参数。`Settings` �?0.7.0 已被移除 —�?调用方直接传 `max_iterations` �?`SimpleAgent`，`create_llm_provider()` 现在每次必须�?`model` 参数�?

#### 问题 2: `AgentRuntime._create_agent()` 硬编�?`SimpleAgent`

**状�?*: �?已修�?

**修复**: 引入 `AgentFactory` 协议。`AgentRuntime.__init__()` 接受可选的 `agent_factory` 参数和必选的 `llm_provider_resolver`。当未提�?`agent_factory` 时，`DefaultAgentFactory` 通过 `DefaultSimpleAgentFactory` 构建 `SimpleAgent` 并把 `kwargs["max_iterations"]`（缺�?100）透传�?`SimpleAgent`。`Settings` �?0.7.0 移除�?

#### 问题 3: `AgentRuntime` 直接构�?`StreamingTool` 实例

**状�?*: �?已修�?

**修复**: `make_handoff_tool()` �?`make_discover_agents_tool()` 提取�?`tool/built_in/runtime_tools.py`。`register_runtime_tools()` 现在是独立函数，由应用层�?Runtime 初始化后调用，将运行时工具注册到 `ToolRegistry`�?

#### 问题 4: `MemoryStore` 直接构�?`ConversationMemory`

**状�?*: �?已修�?

**修复**: 引入 `MemoryFactory = Callable[[], Memory]` 类型，`MemoryStore.__init__()` 接受 `memory_factory` 参数（默�?`lambda: ConversationMemory()`）。`load_memory` 已加�?`Memory` Protocol�?

#### 问题 5: Layer 3 直接导入 Layer 1 LLM Provider

**状�?*: �?已修�?

**修复**: `create_llm_provider()` 移至 `llm/factory.py`（Layer 2），通过 `llm/__init__.py` 导出。`client/built_in/context.py` �?`minimal_harness.llm` 导入并使用�?

#### 问题 6: Layer 3 直接导入 Layer 1 内置工具

**状�?*: �?已修�?

**修复**: `collect_builtin_tools()` �?`get_builtin_tool_names()` 定义�?`tool/registry.py`（Layer 2），`collect_tools()` 移至 `tool/collector.py`。`app.py` 通过 `tool.registry.get_builtin_tool_names()` 获取内置工具名�?

---

### P1: 接口/抽象层设计问�?�?已全部修�?

#### 问题 7: Protocol 物理放置不一�?

**状�?*: �?已修�?

**修复**: 采用方案 A——`ToolRegistryProtocol` �?`tool/base.py` 移至 `tool/registry.py`，与实现 `ToolRegistry` 共存�?

#### 问题 8: Registry �?`register` 签名不一�?

**状�?*: �?已修�?

**修复**: �?`registry.py` 中定�?`RegistryProtocol[T]`（泛型协议基类）。`AgentRegistryProtocol.register()` 改为接收 `AgentMetadata` 对象，实�?`register(item: T)` 统一模式。调用方负责构�?`AgentMetadata`�?

#### 问题 9: `AgentRuntimeProtocol` 私有属�?+ `Any` 类型

**状�?*: �?已修�?

**修复**: 采用方案 B——属性名去下划线前缀（`agent_registry`、`memory_store`、`tool_registry`）。`memory_store` 类型�?`Any` 改为 `MemoryStoreProtocol`（新定义的协议）�?

#### 问题 10: `Agent` Protocol 弱契�?

**状�?*: �?已修�?

**修复**: `Agent.run()` Protocol �?`memory`、`tools`、`stop_event` 移除 `= None` 默认值，保留 `| None` 类型以兼容未来实现�?

#### 问题 11: `_ManagedMemory` 暴露内部实现

**状�?*: �?已修�?

**修复**: `MemoryStore.create_memory()` �?`get_memory()` 返回类型改为 `Memory`。`Memory` Protocol 新增 `memory_id`、`title`、`agent_name`、`created_at` 属性和 `load_memory()` 方法�?

#### 问题 12: 工具注册双重机制

**状�?*: �?已修�?

**修复**: `AgentRuntime._register_runtime_tools()` 被移除，`register_runtime_tools()` 现在�?`tool/built_in/runtime_tools.py` 中的独立函数，由应用层在 Runtime 初始化后显式调用。`_inject_runtime_tools()` �?Registry 查找而非动态构造�?

#### 问题 13: `AgentRegistry` name/metadata_id 映射

**状�?*: �?已修�?

**修复**: `AgentRegistry` 新增 `_name_to_id: dict[str, str]` 映射，`get()` �?`unregister()` 均通过 name→id 映射进行查找�?

#### 问题 14: Layer 1 `__init__.py` 重导�?L2 类型

**状�?*: �?已修�?

**修复**: `agent/__init__.py` 仅导�?L1 类型（`Agent`、`SimpleAgent`、`InputContentConversionFunction`）。L2 类型通过顶层 `__init__.py` 统一导出�?

---

### P2: 语义/卫生度问�?

#### 问题 15: `LLMChunk` 事件是多余的包装

**状�?*: �?未处�?

（保留原描述，待后续迭代处理�?

#### 问题 16: `InputContentPart` 别名泄漏了上下文假设

**状�?*: �?未处�?

（保留原描述，待后续迭代处理�?

#### 问题 17: 事件别名系统冗余

**状�?*: �?已修�?

**修复**: `client/events.py` 简化为直接 re-export 事件类型（不再含�?`to_client_event` 等冗余内容）。`client/__init__.py` 清理掉旧的别名格式，统一为简洁的事件 re-export。示例文件中�?`to_client_event` 引用已移除�?

#### 问题 18: Action 文件访问私有内部状�?

**状�?*: �?已修�?

**修复**: `SessionController` 新增公开方法 `is_session_running(session_id)` �?`get_all_sessions()`。`app.py` �?`actions/sessions.py` 改用公开 API 替代直接访问 `_active_runs` �?`_sessions`�?

#### 问题 19: 弱类型化使用 `Any`

**状�?*: �?已修�?

**修复**: `LLMProviderFactory` 改为 `Callable[[], LLMProvider]`。`session_store` 参数使用 `SessionStoreProtocol`。`AgentRuntimeProtocol` 所有属性均使用具体协议类型�?

#### 问题 20: `AgentMetadata` 归属�?Layer 2

**状�?*: �?已修�?

**修复**: `AgentMetadata` 数据类定义移�?`types.py`（Layer 1）。`agent/registry.py` 通过 import 使用�?

#### 问题 21: `MemoryStore` 的层级归属模糊于物理结构

**状�?*: �?未处�?

（保留原描述，此问题需要较大规模的文件重组，待后续迭代处理�?

---

## 缺少的抽象（已全部补齐）

| 当前用法 | 缺少的抽�?| 定义位置 | 状�?|
|----------|-----------|-------------|------|
| `Callable[[], Any]` (LLMProviderFactory) | `LLMProviderFactory` | `llm/llm.py` | �?已定�?|
| `ConversationMemory()` 直接调用 | `MemoryFactory = Callable[[], Memory]` | `memory_store.py` | �?已定�?|
| `SimpleAgent(...)` 直接调用 | `AgentFactory` | `agent/runtime.py` | �?已定�?|
| `session_store: Any` | `SessionStoreProtocol` | `memory_store.py` | �?已定�?|
| `Registry[T]` 具体�?| `RegistryProtocol[T]` | `registry.py` | �?已定�?|

---

## 目录结构

```
src/minimal_harness/
├── __init__.py                 # 公开 API 聚合导出（L1+L2 类型�?
├── types.py                    # Layer 1 �?事件 dataclass、TypedDict、AgentMetadata、ToolMetadata、Binding 类型
├── memory.py                   # Layer 1 �?Memory Protocol、Message 类型、ConversationMemory
├── session.py                  # Layer 2 �?Session Protocol + SessionSummary + SimpleSession
├── registry.py                 # Layer 2 �?Registry[T] + RegistryProtocol[T] + RegistryChangeEvent
├── memory_store.py             # Layer 2 �?SessionStoreProtocol + MemoryFactory
├── sse_serialization.py        # Layer 2 �?serialize_event / deserialize_event (SSE wire format)
├── settings.py                 # Layer 2 �?环境变量配置
├── adapters.py                 # Layer 2 �?RegistryProvider, MetadataManager, ToolProvider 协议
├── database.py                 # Layer 2 �?generate_bigint_id
├── agent/
�?  ├── __init__.py             # Agent 相关公开 API（Agent, SimpleAgent, CompactionAgent, RemoteAgent, AgentRuntime, AgentFactory 等）
�?  ├── protocol.py             # Layer 1 �?Agent Protocol
�?  ├── simple.py               # Layer 1 �?SimpleAgent 实现
�?  ├── compacting.py           # Layer 1 �?CompactionAgent 实现（SimpleAgent 的结构性克�?+ Memory.compact() 钩子�?
�?  ├── middleware.py           # Layer 1 �?Middleware 基类（钩子系统；�?on_compaction_start/end�?
�?  ├── remote.py               # Layer 2 �?RemoteAgent (远程 Agent 代理)
�?  ├── driver.py               # Layer 2 �?RemoteAgentDriver Protocol + SSEAgentDriver
�?  ├── factory.py              # Layer 2 �?AgentFactory + DefaultAgentFactory + DefaultSimpleAgentFactory + CompactingAgentFactory
�?  ├── runner.py               # Layer 2 �?SSEAgentRunner (shared agent runner for SSE event stream)
�?  ├── runtime.py              # Layer 2 �?AgentRuntime + AgentRuntimeProtocol
�?  └── registry.py             # Layer 2 �?AgentRegistry + AgentRegistryProtocol
├── eval/                       # Layer 2 �?评测模块
�?  ├── __init__.py             # run_evaluation, run_evaluation_simple, EvalCollector, 类型导出
�?  ├── types.py                # EvalRunRecord, EvalSummary, EvalTaskConfig, TokenUsageRecord
�?  ├── runner.py               # EvalRunner �?并发评测编排
�?  ├── collector.py            # EvalCollector(Middleware) �?全链路事件采�?
�?  ├── persistence.py          # JSONL 实时落盘 + summary
�?  └── report.py               # 自包�?HTML 报告生成
├── llm/
�?  ├── __init__.py             # Layer 1/2 �?LLMProvider、LLMProviderFactory、LLMProviderRegistry、create_llm_provider
�?  ├── llm.py                  # Layer 1 �?LLMProvider Protocol、LLMResponse、Stream、LLMProviderRegistry
�?  ├── factory.py              # Layer 2 �?create_llm_provider 工厂实现 + register_builtin_providers
�?  ├── openai.py               # Layer 1 �?OpenAILLMProvider
�?  └── anthropic.py            # Layer 1 �?AnthropicLLMProvider
├── tool/
�?  ├── __init__.py             # Tool 相关公开 API（含类型、绑�?re-export�?
�?  ├── base.py                 # Layer 1 �?Tool Protocol、StreamingTool、create_streaming_tool
�?  ├── registry.py             # Layer 2 �?ToolRegistry(Registry[ToolMetadata]) + ToolRegistryProtocol + collect_builtin_tools
�?  ├── collector.py            # Layer 2 �?collect_tools 工具聚合
�?  ├── registration.py         # Layer 2 �?@register_tool 装饰�?+ register_decorated_tools
�?  ├── factory.py              # Layer 2 �?ToolFactory + DefaultToolFactory + ToolExecutorFactory
�?  ├── remote.py               # Layer 2 �?RemoteTool + RemoteToolExecutor Protocol + SSEToolExecutor
�?  ├── external_loader.py      # Layer 2 �?外部脚本工具加载
�?  ├── wrapper.py              # Layer 2 �?ExternalToolWrapper (子进程执�?
�?  └── built_in/
�?      ├── bash.py             # Layer 1 �?bash 工具
�?      └── local_file_operation.py  # Layer 1 �?文件操作工具
└── client/
    ├── __init__.py             # 事件类型 re-export（保留为旧路�?shim�?
    ├── events.py               # 事件类型 re-export（保留为旧路�?shim�?
    └── logging_setup.py        # setup_service_logging()（服务侧使用；TUI 模式�?mh-tui�?
```

> **Layer 3 应用层（mh-tui、mh-mh-gateway）已不在本仓�?*�?
> - 完整 TUI 组件（`app.py`、widget、modals、actions、config/、streaming/）→ [`mh-tui`](https://github.com/J0ey1iu/mh-tui)
> - `JsonlSessionStore` �?`mh-tui`
> - `handoff` / `discover_agents` 运行时工具及 `register_runtime_tools()` �?`mh_tui.runtime_tools`
