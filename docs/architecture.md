# Architecture

## Overview

minimal-harness 采用三层抽象架构，自底向上分别为：

- **Layer 1 — 核心抽象层 (Core Abstractions)**：定义 Agent、Tool、Memory、LLMProvider 等基础概念及其事件体系。该层不依赖任何具体应用，是整个系统的基石。
- **Layer 2 — 面向服务层 (Service Abstractions)**：在 Layer 1 之上提供 Runtime（运行编排）、Registry（注册发现）、MemoryStore（持久化）等面向运行时的服务能力。该层依赖 Layer 1 的 Protocol，为上层提供更高阶的抽象。
- **Layer 3 — 应用层 (Application)**：当前由 Textual TUI 客户端占据，依托 Layer 2 抽象实现用户交互。这一层包含会话管理、事件渲染、流式输出等 UI 相关逻辑。

```
 ┌──────────────────────────────────────────┐
 │  Layer 3: TUI Application                │
 │  TUIApp → SessionController → Display    │
 ├──────────────────────────────────────────┤
 │  Layer 2: Service Abstractions           │
 │  AgentRuntime · Registry<> · MemoryStore │
 ├──────────────────────────────────────────┤
 │  Layer 1: Core Abstractions              │
 │  Agent · Tool · Memory · LLMProvider     │
 │  Events (AgentEvent / ToolEvent)         │
 └──────────────────────────────────────────┘
```

理想依赖方向：**Layer 3 → Layer 2 → Layer 1**，且每层只依赖下层定义的 **Protocol**，不应依赖具体实现。

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
    ) -> AsyncIterator[AgentEvent]: ...
```

Agent 是核心执行单元。其 `run()` 方法接收用户输入、停止信号、记忆、工具和系统提示，通过 `AsyncIterator[AgentEvent]` 对外产出事件流。事件驱动模型使得 Agent 的执行过程对调用方完全透明——调用方只需消费事件即可感知 Agent 内部状态变化。

当前唯一实现为 **`SimpleAgent`** (`agent/simple.py`)，其执行循环为：

1. 追加用户消息至 Memory
2. 调用 LLMProvider.chat() 进行流式推理
3. 流式产出 LLMChunk 事件
4. LLM 完成后，若存在 tool_calls，进入工具执行阶段
5. 将工具执行结果写回 Memory，继续下一轮迭代
6. 最大迭代次数受 `Settings.max_iterations()` 控制

### Tool Protocol

**定义位置**: `src/minimal_harness/tool/base.py`

```python
class Tool(Protocol):
    name: str
    description: str
    parameters: dict
    def to_schema(self) -> dict: ...
    def to_anthropic_schema(self) -> dict[str, Any]: ...
    def execute(
        self, args: dict[str, Any], tool_call: ToolCall,
        stop_event: asyncio.Event | None,
    ) -> AsyncIterator[ToolEvent]: ...
```

Tool 提供 LLM 可调用的外部能力。关键方法：

- `to_schema()` / `to_anthropic_schema()` — 将工具描述导出为 OpenAI / Anthropic 的 function calling schema
- `execute()` — 执行工具，通过 `AsyncIterator[ToolEvent]` 产出 `ToolStart → ToolProgress* → ToolEnd` 事件序列

唯一内置实现 **`StreamingTool`** (`tool/base.py:80`) 将任意 `StreamingToolFunction` 包装为 Tool。其 `execute()` 会自动产出生命周期事件并处理异常/取消。

内置工具（`tool/built_in/`）：
- **`bash`** — shell 命令执行，支持超时、工作目录、流式输出
- **`local_file_operation`** — 本地文件读写、patch、删除

`StreamingToolFunction` 类型为 `Callable[..., AsyncIterator[Any]]`，简单直观。工具函数只需是一个返回 `AsyncIterator` 的可调用对象。

### Memory Protocol

**定义位置**: `src/minimal_harness/memory.py`

```python
class Memory(Protocol):
    def add_message(self, message: Message) -> None: ...
    def get_all_messages(self) -> list[Message]: ...
    def get_forward_messages(self) -> list[Message]: ...
    def clear_messages(self) -> None: ...
    def set_message_usage(self, usage: TokenUsage) -> None: ...
    def get_message_usage(self) -> TokenUsage: ...
    def dump_memory(self) -> MemoryData: ...
    def load_memory(self, data: MemoryData) -> None: ...
```

Memory 维护对话历史（纯消息容器）。

### Session Protocol

**定义位置**: `src/minimal_harness/session.py`

```python
class Session(Memory, Protocol):
    session_id: str
    memory_id: str
    agent_name: str
    user_id: str
    scenario_id: str | None
    title: str | None
    created_at: str
    memory: Memory
```

Session = Memory（全部消息方法） + 身份字段（user_id, scenario_id）。L2 的 Store 操作的是 Session，
而非裸的 Memory。应用层可以直接使用 Session 的 `user_id`/`scenario_id`，无需通过 `extra` 或反射。消息类型（`Message`）：

| 类型 | 角色 | 说明 |
|------|------|------|
| `SystemMessage` | `system` | 系统提示 |
| `UserMessage` | `user` | 用户输入（文本/图片/文件） |
| `AssistantMessage` | `assistant` | LLM 回复（含 tool_calls） |
| `ToolMessage` | `tool` | 工具执行结果 |
| `ReasoningMessage` | `reasoning` | LLM 推理链（如 DeepSeek R1） |

`get_forward_messages()` 过滤掉 `reasoning` 消息（不发送给 LLM），`dump_memory()` 输出完整的可序列化状态用于持久化。

当前实现：
- **`ConversationMemory`** (`memory.py:105`) — 纯内存实现，支持 JSON 序列化/反序列化
- **`_ManagedMemory`** (`memory_store.py:146`) — 代理模式，每次变更后自动持久化到磁盘

### LLMProvider Protocol

**定义位置**: `src/minimal_harness/llm/llm.py`

```python
class LLMProvider(Protocol):
    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool],
        stop_event: asyncio.Event | None = None,
    ) -> Stream[LLMChunkDelta]: ...
```

LLMProvider 负责与外部 LLM API 交互。`chat()` 返回 `Stream[LLMChunkDelta]`，一个自定义的异步迭代器包装，在流耗尽时会持有最终的 `LLMResponse`（含 `content`、`reasoning_content`、`tool_calls`、`finish_reason`、`usage`）。

当前实现：
- **`OpenAILLMProvider`** (`llm/openai.py`) — 基于 `AsyncOpenAI` SDK
- **`AnthropicLLMProvider`** (`llm/anthropic.py`) — 基于 `AsyncAnthropic` SDK

两者均将原生事件统一转换为 `LLMChunkDelta`，实现 provider-agnostic 的流式输出。

### 事件体系

**定义位置**: `src/minimal_harness/types.py`

所有事件均为 `@dataclass`，通过 `AgentEvent` 联合类型统一：

```
AgentEvent (Union)
├── AgentStart          # 运行开始，携带 user_input
├── LLMStart            # LLM 调用开始，携带 messages & tools
├── LLMChunk            # 流式 LLM 输出块 (包装 LLMChunkDelta)
├── LLMEnd              # LLM 调用结束，携带 content / reasoning_content / tool_calls / usage
├── MemoryUpdate        # token 用量更新
├── ExecutionStart      # 工具批量执行开始
│   ├── ToolStart       # 单个工具开始
│   ├── ToolProgress    # 单个工具进度
│   └── ToolEnd         # 单个工具结束
├── ExecutionEnd        # 工具批量执行结束，携带 results
└── AgentEnd            # 运行结束，携带 response / time_taken / exceeded

ToolEvent (Union) = ToolStart | ToolProgress | ToolEnd
```

事件流具有层级结构：Agent 包含 LLM 调用，LLM 调用可能触发工具执行，工具执行可产生进度事件。每层以 `Start/End` 括起。

`LLMChunkDelta` 是关键的数据类，作为 provider-agnostic 的流式增量：

```python
@dataclass
class LLMChunkDelta:
    content: str | None = None          # 正文增量
    reasoning: str | None = None        # 推理链增量 (如 DeepSeek R1)
    tool_calls: list[ToolCallDelta] | None = None  # 工具调用增量
```

### 其他 Layer 1 类型

- **`TokenUsage`** (`types.py:44`) — `{prompt_tokens, completion_tokens, total_tokens}` TypedDict
- **`ToolCall`** (`types.py:32`) — `{id, type, function: {name, arguments}}` TypedDict
- **`ToolCallFunction`** (`types.py:25`) — `{name, arguments}` TypedDict
- **`ToolCallDelta`** (`types.py:70`) — 流式工具调用增量 `{index, id, name, arguments}`
- **`Stream[T]`** (`llm/llm.py:49`) — 泛型异步迭代器包装，捕获最终 LLMResponse
- **`InputContentConversionFunction`** (`agent/protocol.py:8`) — 输入转换回调类型

---

## Layer 2: 面向服务抽象 (Service Abstractions)

### AgentRuntime / AgentRuntimeProtocol

**定义位置**: `src/minimal_harness/agent/runtime.py`

```python
@runtime_checkable
class AgentRuntimeProtocol(Protocol):
    _agent_registry: AgentRegistryProtocol
    _memory_store: Any
    _tool_registry: ToolRegistryProtocol

    def run(
        self,
        user_input: Iterable[ExtendedInputContentPart],
        agent_metadata_id: str,
        memory_id: str,
        agent_type: str | None = None,
    ) -> tuple[asyncio.Task, asyncio.Event, asyncio.Queue[AgentEvent | None]]: ...
```

`AgentRuntime` 是 Layer 2 的核心编排器。它的职责是：

1. 通过 `AgentRegistry` 查找 Agent 元数据（名称、系统提示、工具列表）
2. 通过 `MemoryStore` 获取/创建 Memory 实例
3. 通过 `ToolRegistry` 解析工具列表
4. 通过 `LLMProviderFactory` 创建 LLM Provider
5. 创建 `SimpleAgent` 实例并调用其 `run()` 方法
6. 返回 `(Task, Event, Queue)` 三元组供调用方驱动执行

此外，Runtime 会在每次运行前注入 `handoff` 和 `discover_agents` 两个运行时工具（`_inject_runtime_tools()`），实现多 Agent 协作能力。

`handoff` 工具递归调用 `AgentRuntime.run()` 创建子任务；`discover_agents` 工具从 Registry 读取可用 Agent 列表。

### Registry 体系

**`Registry[T]`** (`registry.py`) — 泛型注册基底，提供：

- `_register(name, item)` — 注册条目
- `unregister(name)` — 注销条目
- `get(name)`, `get_all()`, `names()`, `clear()` — 查询操作
- `add_listener(fn)` / `remove_listener(fn)` — 观察者模式通知

基于此派生出两种注册表：

**`ToolRegistry`** (`tool/registry.py`) — 继承 `Registry[ToolMetadata]`，存储工具元数据而非可执行对象

```python
class ToolRegistry(Registry[ToolMetadata]):
    def register(self, metadata: ToolMetadata) -> None: ...
    def register_from_binding(self, name, description, parameters, binding, ...) -> None: ...
```

**`AgentRegistry`** (`agent/registry.py`) — 继承 `Registry[AgentMetadata]`，与 ToolRegistry 完全对称

```python
class AgentRegistry(Registry[AgentMetadata]):
    def register(self, metadata: AgentMetadata) -> AgentMetadata: ...
```

### Factory 体系

ToolRegistry 只存元数据，执行时由 **ToolFactory** 创建可执行的 `Tool` 对象：

**`ToolFactory`** (`tool/factory.py`) — Protocol

```python
class ToolFactory(Protocol):
    def create(self, metadata: ToolMetadata) -> Tool: ...
```

**`DefaultToolFactory`** — 处理所有内置 Binding 类型的默认实现，支持自定义 `RemoteToolExecutor` 注册：

```python
class DefaultToolFactory:
    def __init__(self, executor_factories: dict[str, ToolExecutorFactory] | None = None): ...
    def create(self, metadata: ToolMetadata) -> Tool: ...
```

三种 Binding 类型的映射：

| Binding | 创建的可执行体 | 说明 |
|---------|---------------|------|
| `LocalToolBinding(fn=...)` | `StreamingTool(fn, ...)` | 本地函数 |
| `ExternalScriptToolBinding(script_path=...)` | `StreamingTool(fn=ExternalToolWrapper(...))` | 子进程脚本 |
| `RemoteToolBinding(url=..., driver=...)` | `RemoteTool(executor=...)` | HTTP 远程调用 |

**`AgentMetadata.binding`** 控制 Agent 的创建方式：

| binding 值 | 创建的 Agent | 说明 |
|-----------|-------------|------|
| `None` | `SimpleAgent` (或自定义 AgentFactory) | 本地执行 |
| `RemoteAgentBinding(url=..., driver=...)` | `RemoteAgent(driver=...)` | HTTP 远程调用 |

`AgentRuntime` 通过 `agent_driver_factories` 字典按 `driver` 名查找 `RemoteAgentDriverFactory`。这些扩展点使得框架完全掌握在用户手中。

### ToolRegistryProtocol / AgentRegistryProtocol

**定义位置**: `tool/base.py` / `agent/registry.py`

两个协议均标记为 `@runtime_checkable`，分别定义了 Tool 和 Agent 的注册发现接口。它们的 CRUD 方法集高度相似（register / unregister / get / get_all / names / clear），但签名的差异导致无法用单一泛型协议统一：

- `ToolRegistryProtocol.register(tool: Tool)` — 接受已构造的 Tool 对象
- `AgentRegistryProtocol.register(*, name, description, ...)` — 接受关键字段并在内部构造 AgentMetadata

AgentRegistryProtocol 额外暴露 `add_listener` / `remove_listener`，而 ToolRegistryProtocol 没有（虽然基类 `Registry[T]` 也提供了监听器能力）。

### MemoryStore

**定义位置**: `src/minimal_harness/memory_store.py`

```python
class MemoryStore:
    def __init__(self, storage_dir: Path | None = None) -> None: ...
    def create_memory(self, memory_id=None, agent_name="") -> _ManagedMemory: ...
    def get_memory(self, memory_id: str) -> _ManagedMemory | None: ...
    def save_memory(self, memory, memory_id, extra=None) -> None: ...
    def delete_memory(self, memory_id: str) -> bool: ...
    def list_sessions(self) -> list[dict[str, Any]]: ...
```

MemoryStore 提供 Memory 的持久化能力，文件存储在 `~/.minimal_harness/memories/` 下。

`_ManagedMemory` 是内部代理类，包装 `ConversationMemory` 并在每次变更后自动调用 `MemoryStore._persist()` 写入磁盘，实现"变化即持久化"的模式。

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
    def api_key(cls) -> str | None: ...
    @classmethod
    def theme(cls) -> str: ...
```

从环境变量读取配置的静态工具类。归属于 Layer 2，因为它处理环境相关的运行时配置。

---

## Layer 3: 应用层 — TUI 客户端

### 整体数据流

```
User Input (ChatInput widget)
    │
    ▼
TUIApp.action_submit()
    │
    ▼
SessionController.start_run()
    │
    ▼
AgentRuntime.run() ──► asyncio.Task ──► asyncio.Queue[AgentEvent]
    │
    ▼
TUIApp._tick() ──► SessionController.drain_session_events()
    │
    ▼
ChatDisplay.handle_event() ──► StreamBuffer ──► StreamingController
    │                                              │
    ▼                                              ▼
ExportTracker ──► ExportPresenter.export_svg()    Widget (live)
```

### 关键组件

| 组件 | 文件 | 职责 |
|------|------|------|
| `TUIApp` | `app.py` | 顶层 Textual App，组合所有 widget，驱动事件循环 |
| `AppContext` | `context.py` | 聚合 TUIConfig、ToolRegistry、MemoryStore |
| `SessionController` | `session_controller.py` | 会话生命周期管理 + 运行协调 |
| `SessionFactory` | `session_factory.py` | 创建/加载 `ConversationSession` 实例 |
| `AgentManager` | `agent_manager.py` | Agent 预设注册、默认会话创建 |
| `SessionReplayer` | `session_replayer.py` | 从 Memory 回放历史对话到 ChatDisplay |
| `ChatDisplay` | `display.py` | 事件分发 → 渲染、流式输出、导出追踪 |
| `StreamBuffer` | `buffer.py` | 流式内容累积缓冲区 |
| `StreamingController` | `streaming_controller.py` | 管理流式过程中的实时 widget 更新 |
| `SlashCommandHandler` | `slash_handler.py` | `/` 命令系统（config、tools、new、sessions、share） |

### 运行时会话模型

```python
@dataclass
class ConversationSession:
    session: Session          # L2 Session 实体（身份 + 消息）
    agent_metadata_id: str
    tool_names: list[str]
    stop_event: asyncio.Event
    def interrupt(self) -> None: ...
    def reset(self) -> None: ...
```

`ConversationSession` 是 L3 的运行时包装：持有 L2 `Session` 实体（含 identity 和消息），
叠加运行控制信息（stop_event、agent 绑定、工具列表）。
TUI 通过 `SessionController` 管理这些会话，支持多会话并行运行（后台 handoff 任务）。

---

## Protocol 全集

| Protocol | 层级 | 文件 | 标记 |
|----------|------|------|------|
| `Agent` | Layer 1 | `agent/protocol.py` | — |
| `LLMProvider` | Layer 1 | `llm/llm.py` | — |
| `Tool` | Layer 1 | `tool/base.py` | — |
| `Memory` | Layer 1 | `memory.py` | — |
| `Session` | Layer 2 | `session.py` | — |
| `RegistryProtocol[T]` | Layer 2 | `registry.py` | `@runtime_checkable` |
| `MemoryStoreProtocol` | Layer 2 | `memory_store.py` | `@runtime_checkable` |
| `ToolRegistryProtocol` | Layer 2 | `tool/registry.py` | `@runtime_checkable` |
| `AgentRegistryProtocol` | Layer 2 | `agent/registry.py` | `@runtime_checkable` |
| `AgentRuntimeProtocol` | Layer 2 | `agent/runtime.py` | `@runtime_checkable` |

### 工厂类型别名

| 类型 | 定义 | 文件 |
|------|------|------|
| `LLMProviderFactory` | `Callable[[], LLMProvider]` | `llm/llm.py` |
| `MemoryFactory` | `Callable[[], Memory]` | `memory_store.py` |
| `AgentFactory` | `Callable[..., Agent]` | `agent/runtime.py` |

### Protocol 实现关系

```
Agent ◄────────── SimpleAgent
LLMProvider ◄──── OpenAILLMProvider
         ◄──── AnthropicLLMProvider
Tool ◄─────────── StreamingTool
               ◄─── RemoteTool (NEW — HTTP 远程调用)
Memory ◄───────── ConversationMemory
Session ◄──────── ManagedSession (proxy with identity)
RegistryProtocol[T] ◄── Registry[T]
ToolRegistryProtocol ◄── ToolRegistry(Registry[ToolMetadata])
AgentRegistryProtocol ◄── AgentRegistry(Registry[AgentMetadata])
MemoryStoreProtocol ◄── DiskMemoryStore
AgentRuntimeProtocol ◄── AgentRuntime
ToolFactory ◄──── DefaultToolFactory (NEW)
RemoteToolExecutor ◄── SSEToolExecutor (NEW)
RemoteAgentDriver ◄─── SSEAgentDriver (NEW)
```

---

## 跨层导入违规全景（已全部解决）

下表汇总了已修复的所有违反分层依赖方向的导入（按严重程度排序）：

| # | 文件 (当前层级) | 导入目标 | 目标层级 | 方向 | 严重度 | 状态 |
|---|---------------|---------|---------|------|-------|------|
| 1 | `agent/simple.py` (L1) | `settings.Settings` | L2 | L1→L2 | P0 | ✅ 已修复 |
| 2 | `llm/openai.py` (L1) | `settings.Settings` | L2 | L1→L2 | P0 | ✅ 已修复 |
| 3 | `llm/anthropic.py` (L1) | `settings.Settings` | L2 | L1→L2 | P0 | ✅ 已修复 |
| 4 | `agent/runtime.py` (L2) | `agent/simple.SimpleAgent` | L1 concrete | L2→L1 concrete | P0 | ✅ 已修复 |
| 5 | `agent/runtime.py` (L2) | `tool/base.StreamingTool` | L1 concrete | L2→L1 concrete | P0 | ✅ 已修复 |
| 6 | `memory_store.py` (L2) | `memory.ConversationMemory` | L1 concrete | L2→L1 concrete | P0 | ✅ 已修复 |
| 7 | `client/built_in/context.py` (L3) | `llm.OpenAILLMProvider, AnthropicLLMProvider` | L1 concrete | L3→L1 concrete | P0 | ✅ 已修复 |
| 8 | `client/built_in/app.py` (L3) | `tool.built_in.bash.get_tools` | L1 concrete | L3→L1 concrete | P0 | ✅ 已修复 |
| 9 | `client/built_in/app.py` (L3) | `tool.built_in.local_file_operation.get_tools` | L1 concrete | L3→L1 concrete | P0 | ✅ 已修复 |
| 10 | `client/built_in/config/tools.py` (L3) | `tool.built_in.bash.get_tools` | L1 concrete | L3→L1 concrete | P0 | ✅ 已修复 |
| 11 | `client/built_in/config/tools.py` (L3) | `tool.built_in.local_file_operation.get_tools` | L1 concrete | L3→L1 concrete | P0 | ✅ 已修复 |
| 12 | `agent/__init__.py` (L1 facade) | `agent/registry.AgentRegistry` | L2 | L1 re-export L2 | P1 | ✅ 已修复 |
| 13 | `agent/__init__.py` (L1 facade) | `agent/runtime.AgentRuntime` | L2 | L1 re-export L2 | P1 | ✅ 已修复 |
| 14 | `tool/__init__.py` (L1 facade) | `tool/registry.ToolRegistry` | L2 | L1 re-export L2 | P1 | ✅ 已修复 |
| 15 | `client/__init__.py` (L3) | `types.*` (event types) | L1 | L3→L1 | P2 | ⚪ 保留 |
| 16 | `client/events.py` (L3) | `types.AgentEvent` | L1 | L3→L1 | P2 | ✅ 已清理 |
| 17 | `client/built_in/display.py` (L3) | `types.*` (event types) | L1 | L3→L1 | P2 | ⚪ 保留 |
| 18 | `client/built_in/actions/sessions.py` (L3) | access `_ctrl._sessions` | L2 internal | L3→L2 internal | P2 | ✅ 已修复 |

---

## 解决方案摘要

以下问题均已修复，保留原始问题描述和改动思路作为历史记录。

### P0: 分层方向违规 — 已全部修复

#### 问题 1: Layer 1 实现类反向依赖 `Settings`

**状态**: ✅ 已修复

**修复**: `SimpleAgent.__init__()` 的 `max_iterations` 改为必选参数；`OpenAILLMProvider. __init__()` / `AnthropicLLMProvider.__init__()` 的 `model` 改为必选参数。`Settings` 的调用点集中在 `AgentRuntime._create_agent()` 和 `llm/factory.py:create_llm_provider()`。

#### 问题 2: `AgentRuntime._create_agent()` 硬编码 `SimpleAgent`

**状态**: ✅ 已修复

**修复**: 引入 `AgentFactory = Callable[..., Agent]`。`AgentRuntime.__init__()` 接受可选的 `agent_factory` 和 `llm_provider_factory` 参数。当未提供 `agent_factory` 时，fallback 逻辑仍在 `_create_agent()` 中构建 `SimpleAgent` 并通过 `Settings.max_iterations()` 注入配置。

#### 问题 3: `AgentRuntime` 直接构造 `StreamingTool` 实例

**状态**: ✅ 已修复

**修复**: `make_handoff_tool()` 和 `make_discover_agents_tool()` 提取到 `tool/built_in/runtime_tools.py`。`AgentRuntime` 通过懒导入调用这些工厂函数，且运行时工具在 `__init__` 时通过 `_register_runtime_tools()` 注册到 `ToolRegistry`。

#### 问题 4: `MemoryStore` 直接构造 `ConversationMemory`

**状态**: ✅ 已修复

**修复**: 引入 `MemoryFactory = Callable[[], Memory]` 类型，`MemoryStore.__init__()` 接受 `memory_factory` 参数（默认 `lambda: ConversationMemory()`）。`load_memory` 已加入 `Memory` Protocol。

#### 问题 5: Layer 3 直接导入 Layer 1 LLM Provider

**状态**: ✅ 已修复

**修复**: `create_llm_provider()` 移至 `llm/factory.py`（Layer 2），通过 `llm/__init__.py` 导出。`client/built_in/context.py` 从 `minimal_harness.llm` 导入并使用。

#### 问题 6: Layer 3 直接导入 Layer 1 内置工具

**状态**: ✅ 已修复

**修复**: `collect_builtin_tools()` 和 `get_builtin_tool_names()` 定义在 `tool/registry.py`（Layer 2），`collect_tools()` 移至 `tool/collector.py`。`app.py` 通过 `tool.registry.get_builtin_tool_names()` 获取内置工具名。

---

### P1: 接口/抽象层设计问题 — 已全部修复

#### 问题 7: Protocol 物理放置不一致

**状态**: ✅ 已修复

**修复**: 采用方案 A——`ToolRegistryProtocol` 从 `tool/base.py` 移至 `tool/registry.py`，与实现 `ToolRegistry` 共存。

#### 问题 8: Registry 的 `register` 签名不一致

**状态**: ✅ 已修复

**修复**: 在 `registry.py` 中定义 `RegistryProtocol[T]`（泛型协议基类）。`AgentRegistryProtocol.register()` 改为接收 `AgentMetadata` 对象，实现 `register(item: T)` 统一模式。调用方负责构造 `AgentMetadata`。

#### 问题 9: `AgentRuntimeProtocol` 私有属性 + `Any` 类型

**状态**: ✅ 已修复

**修复**: 采用方案 B——属性名去下划线前缀（`agent_registry`、`memory_store`、`tool_registry`）。`memory_store` 类型从 `Any` 改为 `MemoryStoreProtocol`（新定义的协议）。

#### 问题 10: `Agent` Protocol 弱契约

**状态**: ✅ 已修复

**修复**: `Agent.run()` Protocol 中 `memory`、`tools`、`stop_event` 移除 `= None` 默认值，保留 `| None` 类型以兼容未来实现。

#### 问题 11: `_ManagedMemory` 暴露内部实现

**状态**: ✅ 已修复

**修复**: `MemoryStore.create_memory()` 和 `get_memory()` 返回类型改为 `Memory`。`Memory` Protocol 新增 `memory_id`、`title`、`agent_name`、`created_at` 属性和 `load_memory()` 方法。

#### 问题 12: 工具注册双重机制

**状态**: ✅ 已修复

**修复**: `AgentRuntime._register_runtime_tools()` 在 `__init__` 时将 `handoff` 和 `discover_agents` 注册到 `ToolRegistry`。`_inject_runtime_tools()` 从 Registry 查找而非动态构造。

#### 问题 13: `AgentRegistry` name/metadata_id 映射

**状态**: ✅ 已修复

**修复**: `AgentRegistry` 新增 `_name_to_id: dict[str, str]` 映射，`get()` 和 `unregister()` 均通过 name→id 映射进行查找。

#### 问题 14: Layer 1 `__init__.py` 重导出 L2 类型

**状态**: ✅ 已修复

**修复**: `agent/__init__.py` 仅导出 L1 类型（`Agent`、`SimpleAgent`、`InputContentConversionFunction`）。L2 类型通过顶层 `__init__.py` 统一导出。

---

### P2: 语义/卫生度问题

#### 问题 15: `LLMChunk` 事件是多余的包装

**状态**: ⚪ 未处理

（保留原描述，待后续迭代处理）

#### 问题 16: `InputContentPart` 别名泄漏了上下文假设

**状态**: ⚪ 未处理

（保留原描述，待后续迭代处理）

#### 问题 17: 事件别名系统冗余

**状态**: ✅ 已修复

**修复**: `client/events.py` 简化为直接 re-export 事件类型（不再含有 `to_client_event` 等冗余内容）。`client/__init__.py` 清理掉旧的别名格式，统一为简洁的事件 re-export。示例文件中的 `to_client_event` 引用已移除。

#### 问题 18: Action 文件访问私有内部状态

**状态**: ✅ 已修复

**修复**: `SessionController` 新增公开方法 `is_session_running(session_id)` 和 `get_all_sessions()`。`app.py` 和 `actions/sessions.py` 改用公开 API 替代直接访问 `_active_runs` 和 `_sessions`。

#### 问题 19: 弱类型化使用 `Any`

**状态**: ✅ 已修复

**修复**: `LLMProviderFactory` 改为 `Callable[[], LLMProvider]`。`memory_store` 参数使用新定义的 `MemoryStoreProtocol`。`AgentRuntimeProtocol` 所有属性均使用具体协议类型。

#### 问题 20: `AgentMetadata` 归属在 Layer 2

**状态**: ✅ 已修复

**修复**: `AgentMetadata` 数据类定义移至 `types.py`（Layer 1）。`agent/registry.py` 通过 import 使用。

#### 问题 21: `MemoryStore` 的层级归属模糊于物理结构

**状态**: ⚪ 未处理

（保留原描述，此问题需要较大规模的文件重组，待后续迭代处理）

---

## 缺少的抽象（已全部补齐）

| 当前用法 | 缺少的抽象 | 定义位置 | 状态 |
|----------|-----------|-------------|------|
| `Callable[[], Any]` (LLMProviderFactory) | `LLMProviderFactory` | `llm/llm.py` | ✅ 已定义 |
| `ConversationMemory()` 直接调用 | `MemoryFactory = Callable[[], Memory]` | `memory_store.py` | ✅ 已定义 |
| `SimpleAgent(...)` 直接调用 | `AgentFactory` | `agent/runtime.py` | ✅ 已定义 |
| `memory_store: Any` | `MemoryStoreProtocol` | `memory_store.py` | ✅ 已定义 |
| `Registry[T]` 具体类 | `RegistryProtocol[T]` | `registry.py` | ✅ 已定义 |

---

## 目录结构

```
src/minimal_harness/
├── __init__.py                 # 公开 API 聚合导出（L1+L2 类型）
├── types.py                    # Layer 1 — 事件 dataclass、TypedDict、AgentMetadata
├── memory.py                   # Layer 1 — Memory Protocol、Message 类型、ConversationMemory
├── registry.py                 # Layer 2 — Registry[T] + RegistryProtocol[T] 泛型基底
├── memory_store.py             # Layer 2 — MemoryStore + MemoryStoreProtocol + _ManagedMemory
├── settings.py                 # Layer 2 — 环境变量配置
├── agent/
│   ├── __init__.py             # Layer 1 — 仅导出 L1 类型 (Agent, SimpleAgent)
│   ├── protocol.py             # Layer 1 — Agent Protocol
│   ├── simple.py               # Layer 1 — SimpleAgent 实现
│   ├── remote.py               # Layer 2 — RemoteAgent (远程 Agent 代理)
│   ├── driver.py               # Layer 2 — RemoteAgentDriver Protocol + SSEAgentDriver
│   ├── runtime.py              # Layer 2 — AgentRuntime + AgentRuntimeProtocol + AgentFactory
│   └── registry.py             # Layer 2 — AgentRegistry + AgentRegistryProtocol
├── llm/
│   ├── __init__.py             # Layer 1/2 — LLMProvider、LLMProviderFactory、create_llm_provider
│   ├── llm.py                  # Layer 1 — LLMProvider Protocol、LLMResponse、Stream
│   ├── factory.py              # Layer 2 — create_llm_provider 工厂实现
│   ├── openai.py               # Layer 1 — OpenAILLMProvider
│   └── anthropic.py            # Layer 1 — AnthropicLLMProvider
├── tool/
│   ├── __init__.py             # Tool 相关公开 API
│   ├── base.py                 # Layer 1 — Tool Protocol、StreamingTool
│   ├── registry.py             # Layer 2 — ToolRegistry(Registry[ToolMetadata]) + ToolRegistryProtocol + collect_builtin_tools
│   ├── collector.py            # Layer 2 — collect_tools 工具聚合
│   ├── registration.py         # Layer 2 — @register_tool 装饰器
│   ├── factory.py              # Layer 2 — ToolFactory + DefaultToolFactory + ToolExecutorFactory
│   ├── remote.py               # Layer 2 — RemoteTool + RemoteToolExecutor Protocol + SSEToolExecutor
│   ├── external_loader.py      # Layer 2 — 外部脚本工具加载
│   ├── wrapper.py              # Layer 2 — ExternalToolWrapper (子进程执行)
│   └── built_in/
│       ├── bash.py             # Layer 1 — bash 工具
│       ├── local_file_operation.py  # Layer 1 — 文件操作工具
│       └── runtime_tools.py    # Layer 2 — handoff / discover_agents 运行时工具
└── client/
    ├── __init__.py             # 向后兼容事件 re-export（已清理）
    ├── events.py               # 向后兼容事件 shim（已清理）
    └── built_in/               # Layer 3 — TUI 客户端
        ├── app.py              # TUIApp (入口)
        ├── context.py          # AppContext + TUIConfig
        ├── session.py          # Session Protocol + ConversationSession
        ├── session_controller.py
        ├── session_factory.py
        ├── session_replayer.py
        ├── agent_manager.py
        ├── display.py          # ChatDisplay
        ├── buffer.py           # StreamBuffer
        ├── streaming_controller.py
        ├── chat_widgets.py
        ├── markdown_styles.py
        ├── renderer.py
        ├── slash_handler.py
        ├── export_tracker.py
        ├── export_presenter.py
        ├── widgets.py
        ├── messages.py
        ├── modals.py
        ├── constants.py
        ├── app.tcss
        ├── config/
        └── actions/
```
