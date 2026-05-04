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
```

Memory 维护对话历史。消息类型（`Message`）：

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

**`ToolRegistry`** (`tool/registry.py`) — 继承 `Registry[Tool]`

```python
class ToolRegistry(Registry[Tool]):
    def register(self, tool: Tool) -> None: ...
    def register_external_tool(self, name, description, parameters, fn, uri=None, **kwargs) -> None: ...
```

**`AgentRegistry`** (`agent/registry.py`) — 继承 `Registry[AgentMetadata]`

```python
class AgentRegistry(Registry[AgentMetadata]):
    def register(self, *, name, description, system_prompt,
                  agent_type, tool_names, metadata_id) -> AgentMetadata: ...
```

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

### 会话模型

```python
class Session(Protocol):
    session_id: str
    agent_metadata_id: str
    memory_id: str
    tool_names: list[str]
    stop_event: asyncio.Event
    def interrupt(self) -> None: ...
    def reset(self) -> None: ...

class ConversationSession  # dataclass, Session 的具体实现
```

Session 绑定了一个 Agent 元数据 ID、一个 Memory ID 和一组工具名。`stop_event` 提供取消能力。TUI 通过 `SessionController` 管理这些会话，支持多会话并行运行（后台 handoff 任务）。

---

## Protocol 全集

| Protocol | 层级 | 文件 | 标记 |
|----------|------|------|------|
| `Agent` | Layer 1 | `agent/protocol.py` | — |
| `LLMProvider` | Layer 1 | `llm/llm.py` | — |
| `Tool` | Layer 1 | `tool/base.py` | — |
| `Memory` | Layer 1 | `memory.py` | — |
| `ToolRegistryProtocol` | Layer 2 | `tool/base.py` | `@runtime_checkable` |
| `AgentRegistryProtocol` | Layer 2 | `agent/registry.py` | `@runtime_checkable` |
| `AgentRuntimeProtocol` | Layer 2 | `agent/runtime.py` | `@runtime_checkable` |
| `Session` | Layer 3 | `client/built_in/session.py` | — |

### Protocol 实现关系

```
Agent ◄────────── SimpleAgent
LLMProvider ◄──── OpenAILLMProvider
         ◄──── AnthropicLLMProvider
Tool ◄─────────── StreamingTool
Memory ◄───────── ConversationMemory
        ◄──────── _ManagedMemory (proxy)
ToolRegistryProtocol ◄── ToolRegistry(Registry[Tool])
AgentRegistryProtocol ◄── AgentRegistry(Registry[AgentMetadata])
AgentRuntimeProtocol ◄── AgentRuntime
Session ◄──────── ConversationSession
```

---

## 跨层导入违规全景

下表汇总了当前代码中所有违反分层依赖方向的实际导入（按严重程度排序）：

| # | 文件 (当前层级) | 导入目标 | 目标层级 | 方向 | 严重度 |
|---|---------------|---------|---------|------|-------|
| 1 | `agent/simple.py` (L1) | `settings.Settings` | L2 | L1→L2 | **P0** |
| 2 | `llm/openai.py` (L1) | `settings.Settings` | L2 | L1→L2 | **P0** |
| 3 | `llm/anthropic.py` (L1) | `settings.Settings` | L2 | L1→L2 | **P0** |
| 4 | `agent/runtime.py` (L2) | `agent/simple.SimpleAgent` | L1 concrete | L2→L1 concrete | **P0** |
| 5 | `agent/runtime.py` (L2) | `tool/base.StreamingTool` | L1 concrete | L2→L1 concrete | **P0** |
| 6 | `memory_store.py` (L2) | `memory.ConversationMemory` | L1 concrete | L2→L1 concrete | **P0** |
| 7 | `client/built_in/context.py` (L3) | `llm.OpenAILLMProvider, AnthropicLLMProvider` | L1 concrete | L3→L1 concrete | **P0** |
| 8 | `client/built_in/app.py` (L3) | `tool.built_in.bash.get_tools` | L1 concrete | L3→L1 concrete | **P0** |
| 9 | `client/built_in/app.py` (L3) | `tool.built_in.local_file_operation.get_tools` | L1 concrete | L3→L1 concrete | **P0** |
| 10 | `client/built_in/config/tools.py` (L3) | `tool.built_in.bash.get_tools` | L1 concrete | L3→L1 concrete | **P0** |
| 11 | `client/built_in/config/tools.py` (L3) | `tool.built_in.local_file_operation.get_tools` | L1 concrete | L3→L1 concrete | **P0** |
| 12 | `agent/__init__.py` (L1 facade) | `agent/registry.AgentRegistry` | L2 | L1 re-export L2 | **P1** |
| 13 | `agent/__init__.py` (L1 facade) | `agent/runtime.AgentRuntime` | L2 | L1 re-export L2 | **P1** |
| 14 | `tool/__init__.py` (L1 facade) | `tool/registry.ToolRegistry` | L2 | L1 re-export L2 | **P1** |
| 15 | `client/__init__.py` (L3) | `types.*` (event types) | L1 | L3→L1 | **P2** |
| 16 | `client/events.py` (L3) | `types.AgentEvent` | L1 | L3→L1 | **P2** |
| 17 | `client/built_in/display.py` (L3) | `types.*` (event types) | L1 | L3→L1 | **P2** |
| 18 | `client/built_in/actions/sessions.py` (L3) | access `_ctrl._sessions` | L2 internal | L3→L2 internal | **P2** |

---

## 当前架构存在的问题

以下问题按优先级分组，P0 为结构性缺陷（违反分层方向），P1 为接口/抽象层设计问题，P2 为卫生度/一致性改进。

---

### P0: 分层方向违规

#### 问题 1: Layer 1 实现类反向依赖 `Settings`（L2 → L1 反向）

**涉及文件**: `agent/simple.py`, `llm/openai.py`, `llm/anthropic.py`

三个 Layer 1 的核心实现类直接调用 `Settings.model()` 和 `Settings.max_iterations()` 等静态方法来获取配置：

```python
# agent/simple.py:41
self._max_iterations = max_iterations if max_iterations is not None else Settings.max_iterations()

# llm/openai.py, llm/anthropic.py
model = model or Settings.model()
```

`Settings` 是 Layer 2 的服务组件（读取环境变量），Layer 1 的核心抽象不应依赖它。这违背了依赖倒置原则——底层实现不应该知道配置从何而来。

**改动思路**：
- 将 `max_iterations` 作为 `SimpleAgent.__init__()` 的必选参数，移除 `Settings` 依赖，由调用方（`AgentRuntime`）负责读取配置并注入
- 将 `model` 参数在 `OpenAILLMProvider.__init__()` / `AnthropicLLMProvider.__init__()` 中去掉默认值 `= ""` 的 fallback 到 `Settings.model()`，改为由调用方显式传入
- `Settings` 的调用点集中到 Layer 2 的 `AgentRuntime` 和 Layer 3 的 `create_llm_provider()`

```python
# 改后示例
class SimpleAgent:
    def __init__(self, llm_provider: LLMProvider, max_iterations: int): ...

class OpenAILLMProvider:
    def __init__(self, client: AsyncOpenAI, model: str): ...  # model 必选
```

#### 问题 2: `AgentRuntime._create_agent()` 硬编码 `SimpleAgent`

**涉及文件**: `agent/runtime.py:90-98`

Layer 2 的编排器通过 lazy import 直接实例化 Layer 1 的具体类：

```python
def _create_agent(self, agent_type: str) -> Agent:
    from minimal_harness.agent.simple import SimpleAgent
    if agent_type == "simple":
        llm_provider = self._llm_provider_factory()
        return SimpleAgent(llm_provider=llm_provider)
    raise ValueError(f"Unknown agent type: {agent_type}")
```

这导致新增 Agent 类型必须修改 Layer 2 代码，且 `"simple"` 字符串硬编码在 `AgentMetadata` 的默认值中 (`agent/registry.py:17`)。

**改动思路**：
- 引入 `AgentFactory` Protocol：

```python
class AgentFactory(Protocol):
    def create(self, llm_provider: LLMProvider, **kwargs: Any) -> Agent: ...
```

- `AgentRuntime.__init__()` 接受 `agent_factories: dict[str, AgentFactory]` 或 `agent_factory: AgentFactory`
- 运行时通过工厂创建 Agent，而非硬编码类型判断

#### 问题 3: `AgentRuntime` 直接构造 `StreamingTool` 实例

**涉及文件**: `agent/runtime.py:175, 325`

`_make_handoff_tool()` 和 `_make_discover_agents_tool()` 内部直接 `import StreamingTool` 并实例化。Layer 2 不应了解 Layer 1 的具体工具类。

**改动思路**：
- 将这两个工具函数提取到独立的模块（如 `tool/built_in/handoff.py`）
- `AgentRuntime` 通过 `ToolRegistry` 或通过注入的 `ToolFactory` 获取这些运行时工具，而非直接构造 `StreamingTool`

#### 问题 4: `MemoryStore` 直接构造 `ConversationMemory`

**涉及文件**: `memory_store.py:41, 63`

```python
inner = ConversationMemory()  # 直接实例化 Layer 1 具体类
```

Layer 2 的 MemoryStore 硬编码了对 Layer 1 具体实现的依赖。如果未来有另一种 Memory 实现（如数据库后端），MemoryStore 无法切换。

**改动思路**：
- 引入 `MemoryFactory` 类型：`Callable[[], Memory]`
- `MemoryStore.__init__()` 接受 `memory_factory` 参数，默认值为 `ConversationMemory`
- 持久化逻辑使用 `Memory.dump_memory()` / `Memory.load_memory()`（已在协议中定义），与具体实现解耦

#### 问题 5: Layer 3 直接导入 Layer 1 LLM Provider

**涉及文件**: `client/built_in/context.py:37-51`

```python
def create_llm_provider(cfg: dict[str, Any]) -> LLMProvider:
    if provider == "anthropic":
        return AnthropicLLMProvider(...)
    return OpenAILLMProvider(...)
```

TUI 直接依赖具体的 LLM Provider 实现类，且 provider 创建逻辑分散在 Layer 3。

**改动思路**：
- 将 `create_llm_provider()` 提升至 Layer 2，作为 `LLMProviderFactory` 的标准实现
- Layer 3 只通过 `LLMProvider` Protocol 使用 provider，不关心具体实现

#### 问题 6: Layer 3 直接导入 Layer 1 内置工具

**涉及文件**: `app.py:74-83`, `config/tools.py`

```python
from minimal_harness.tool.built_in.bash import get_tools as get_bash_tools
from minimal_harness.tool.built_in.local_file_operation import get_tools as get_local_file_operation_tools
```

TUI 层直接感知内置工具列表并手动注册。Layer 3 不应知道有哪些 Layer 1 工具存在。

**改动思路**：
- 在 Layer 2 的 `ToolRegistry` 或 `collect_tools()` 中提供 `collect_builtin_tools()` 方法，负责加载内置工具
- Layer 3 调用 `ctx.rebuild()` 即可自动加载所有工具（内置 + 外部），无需自己枚举
- 或者将 `collect_tools()` 函数下移至 Layer 2 的 `tool/` 目录

---

### P1: 接口/抽象层设计问题

#### 问题 7: Protocol 物理放置不一致

`ToolRegistryProtocol` 定义在 `tool/base.py`，与 `Tool` 协议共处一个文件；而 `AgentRegistryProtocol` 定义在 `agent/registry.py`，与其实现 `AgentRegistry` 共存。两个 Registry Protocol 的放置策略不统一。

**改动思路**：
- 方案 A：将 `ToolRegistryProtocol` 移至 `tool/registry.py` 与实现共存（当前 Agent 的模式）
- 方案 B：将 `AgentRegistryProtocol` 从 `registry.py` 中抽出独立文件，保持与 Tool 一致
- 建议采用方案 A，Protocol 与实现共处一个模块可以减少文件碎片

#### 问题 8: Registry 的 `register` 签名不一致

- `ToolRegistryProtocol.register(tool: Tool)` — 接受已构造的对象
- `AgentRegistryProtocol.register(*, name, description, ...)` — 接受关键字段并在内部构造

这导致无法用泛型 `RegistryProtocol[T]` 统一两者的接口。同时 `AgentRegistryProtocol` 暴露了 `add_listener` / `remove_listener`，而 `ToolRegistryProtocol` 没有（尽管基类有能力）。

**改动思路**：
- 将两者统一为 `register(item: T)` 模式
- `AgentRegistry` 的调用方负责构造 `AgentMetadata`，而非由 Registry 内部构造
- 在 `Registry[T]` 基类上提取 `RegistryProtocol[T]`，让两个子协议继承：

```python
class RegistryProtocol[T](Protocol):
    def register(self, name: str, item: T) -> None: ...
    def unregister(self, name: str) -> bool: ...
    def get(self, name: str) -> T | None: ...
    def get_all(self) -> list[T]: ...
    def names(self) -> list[str]: ...
    def clear(self) -> None: ...
    def add_listener(self, listener: Callable[[], None]) -> None: ...
    def remove_listener(self, listener: Callable[[], None]) -> None: ...
```

#### 问题 9: `AgentRuntimeProtocol` 以私有属性暴露依赖

```python
_agent_registry: AgentRegistryProtocol
_memory_store: Any
_tool_registry: ToolRegistryProtocol
```

以下划线前缀的"私有"属性出现在 Protocol 契约中，语义矛盾。`_memory_store` 使用 `Any` 类型，完全失去类型安全性。

**改动思路**：
- 方案 A：将这些依赖作为 `run()` 方法的参数传入
- 方案 B：将下划线前缀去掉，改为公开属性名（`agent_registry`、`memory_store`、`tool_registry`），并将 `_memory_store: Any` 改为 `MemoryStore` Protocol 的引入

#### 问题 10: `Agent` Protocol 的弱契约

`Agent` Protocol 将 `memory`、`tools`、`stop_event` 声明为可选（`| None = None`），注释解释为 "structural compatibility"。但 `SimpleAgent.run()` 在运行时用 `assert` 强制要求它们非 None。Protocol 定义了过于宽松的契约，实际运行时却是严格约束。

**改动思路**：
- 从 Protocol 签名中移除 `= None` 默认值，改为非 Optional 类型
- 如果确实需要可选性（用于兼容未来其他 Agent 实现），应明确定义默认行为而非用 `assert` 兜底

#### 问题 11: `_ManagedMemory` 暴露了内部实现

`MemoryStore.create_memory()` 和 `get_memory()` 返回类型标注为 `_ManagedMemory`（下划线前缀），而 `_ManagedMemory` 实现了 `Memory` 协议。调用方不应知道返回的是代理类——它们只需要 `Memory` 接口。

**改动思路**：
- 将返回类型标注改为 `Memory`，隐藏 `_ManagedMemory` 实现细节
- 如果需要暴露 `_ManagedMemory` 的扩展属性（如 `title`、`agent_name`），可以在 `Memory` Protocol 中添加这些属性，或通过 `MemoryStore` 的方法间接访问

#### 问题 12: 工具注册的双重机制

系统同时支持两种工具注入方式：
1. `ToolRegistry.register(tool)` / `register_external_tool(...)` — 静态注册，持久存在于 Registry 中
2. `AgentRuntime._inject_runtime_tools()` — 每次运行动态注入 `handoff` / `discover_agents`

运行时工具不在 ToolRegistry 中出现，但在 Agent 执行时会被注入。这意味着 ToolRegistry 不是"所有可用工具"的完全集合，存在隐式的工具来源。

**改动思路**：
- 将 `handoff` / `discover_agents` 也注册到 `ToolRegistry` 中（在 `AgentRuntime.__init__()` 时注册）
- 或者明确引入 "系统工具层" 概念，在架构中显式区分 "用户工具"（Registry 管理）和 "系统工具"（Runtime 注入）

#### 问题 13: `AgentRegistry` 注册 key 与查找 key 不一致

`AgentRegistry.register()` 默认以 `name` 作为 `metadata_id` 进行 `_register()`。但协议方法 `get(name)` 传入的也是 `name`，实际查找的是 `metadata_id`。如果用户显式指定了不同的 `metadata_id`，`get(name)` 将无法找到该条目——因为底层 `Registry._data` 的 key 是 `metadata_id`。

**改动思路**：
- 明确区分对外的 `name`（显示名）和内部的 `metadata_id`（存储 key）
- 建立 `name → metadata_id` 的映射字典，或直接在 `get()` 中同时支持按 name 和 metadata_id 查找
- 或者在文档中明确约束 `name` 必须等于 `metadata_id`

#### 问题 14: Layer 1 `__init__.py` 重新导出 Layer 2 类型

**涉及文件**: `agent/__init__.py`, `tool/__init__.py`

```python
# agent/__init__.py
from .registry import AgentMetadata, AgentRegistry, AgentRegistryProtocol
from .runtime import AgentRuntime, AgentRuntimeProtocol

# tool/__init__.py
from .registry import ToolRegistry
```

`agent` 包本应是 Layer 1 概念（Agent Protocol），但其 `__init__.py` 同时导出了 Layer 2 的服务组件（AgentRegistry、AgentRuntime）。`tool` 包同理。这导致用户可以通过 `from minimal_harness.agent import AgentRuntime` 直接访问，模糊了层级边界。

**改动思路**：
- Layer 1 的 `__init__.py` 只导出 Layer 1 类型：`Agent`、`SimpleAgent`、`Tool`、`StreamingTool`、`ToolRegistryProtocol`（虽然是 L2，但协议允许在 L1 定义）
- 将 Layer 2 的具体实现导出统一通过 `src/minimal_harness/__init__.py` 管理
- 或为 Layer 2 建立独立的包（如 `src/minimal_harness/service/`）

---

### P2: 语义/卫生度问题

#### 问题 15: `LLMChunk` 事件是多余的包装

```python
@dataclass
class LLMChunk:
    chunk: LLMChunkDelta | None
```

`LLMChunkDelta` 本身已是数据类，再套一层 `LLMChunk` 没有增加任何信息。`Agent.run()` 可以直接产出 `LLMChunkDelta` 作为事件。

**改动思路**：
- 在 `AgentEvent` union 中直接用 `LLMChunkDelta` 替换 `LLMChunk`
- `ChatDisplay.handle_event()` 中直接匹配 `isinstance(event, LLMChunkDelta)` 而非先匹配 `LLMChunk` 再解包

#### 问题 16: `InputContentPart` 别名泄漏了上下文假设

```python
# memory.py:29
InputContentPart = TextContentPart
ExtendedInputContentPart = FileContentPart | ImageContentPart | TextContentPart
```

`InputContentPart = TextContentPart` 将多模态输入窄化为纯文本，命名暗示这是"输入"范畴的概念。这个别名出现在 Layer 1 的 `memory.py`中，但文本/图像/文件的区分是下游消费者的关切。Layer 1 应只定义消息结构，不应引入"输入就是文本"的隐含假设。

**改动思路**：
- 消除 `InputContentPart` 别名，统一使用 `TextContentPart`
- 如果确实需要表达"输入只支持文本"的语义限制，应在使用处做校验而非在类型层静默

#### 问题 17: 事件别名系统冗余

**涉及文件**: `client/__init__.py`, `client/events.py`

```python
# client/__init__.py
AgentEndEvent = AgentEnd  # 重命名
Event = AgentEvent        # 重命名
```

`client/events.py` 和 `client/__init__.py` 均提供了 Layer 1 事件的别名系统，且 `to_client_event()` 是恒等函数。这似乎是某次重构的遗留产物——可能曾计划有客户端专用事件层级，但最终未实现。

**改动思路**：
- 删除 `client/events.py`，统一通过 `client/__init__.py` 或直接使用 `minimal_harness.types` 的事件类型
- 如果客户端确实需要独立的事件命名空间，应定义客户端专用事件而非简单做类型别名

#### 问题 18: Action 文件访问私有内部状态

**涉及文件**: `app.py:449`, `actions/sessions.py`

```python
# app.py
if session_id in self._ctrl._active_runs:  # 直接访问 _active_runs

# actions/sessions.py
sessions = self._ctrl._sessions  # 直接访问 _sessions
```

Layer 3 的 Action 绕过 `SessionController` 的公开 API，直接访问其内部 `_active_runs` 和 `_sessions` 字典。

**改动思路**：
- 在 `SessionController` 上添加公开方法：
  - `is_session_running(session_id: str) -> bool`
  - `get_all_sessions() -> dict[str, ConversationSession]`
- 删除外部对 `_active_runs` 和 `_sessions` 的直接访问

#### 问题 19: 弱类型化使用 `Any`

```python
# agent/runtime.py:33
LLMProviderFactory = Callable[[], Any]  # 应为 LLMProvider

# agent/runtime.py:45
_memory_store: Any  # 应使用 IMemoryStore Protocol

# agent/runtime.py:81
memory_store: Any  # 同上
```

使用 `Any` 失去了类型安全的价值。

**改动思路**：
- `LLMProviderFactory` 改为 `Callable[[], LLMProvider]`
- 引入 `MemoryStoreProtocol`（参考 `tool/base.py` 中 `ToolRegistryProtocol` 的模式），替换 `Any`

#### 问题 20: `AgentMetadata` 归属在 Layer 2

**涉及文件**: `agent/registry.py:13`

`AgentMetadata` 是纯数据类（`@dataclass`），定义在 Layer 2 的 `agent/registry.py` 中。但 `AgentRegistryProtocol.get()` 返回 `AgentMetadata | None`，这意味着 Layer 1 的 Protocol 引用了 Layer 2 的数据结构。

**改动思路**：
- 将 `AgentMetadata` 移至 `types.py`（Layer 1 共有类型）或 `agent/protocol.py`
- Layer 2 的 `AgentRegistry` 使用它，但不应定义它

#### 问题 21: `MemoryStore` 的层级归属模糊于物理结构

`MemoryStore` 自述为 "Layer 2 service abstraction"，但文件位于 `src/minimal_harness/memory_store.py`，与 Layer 1 的 `memory.py` 并列。`settings.py`（Layer 2）、`registry.py`（Layer 2）、`memory_store.py`（Layer 2）全部平铺在根目录下，与 Layer 1 的 `memory.py`、`types.py` 混在一起。物理结构无法体现逻辑分层。

**改动思路**（与问题 14 协同考虑）：
```
src/minimal_harness/
├── core/                  # Layer 1 — 核心抽象
│   ├── agent/
│   ├── tool/
│   ├── llm/
│   ├── memory.py
│   └── types.py
├── service/               # Layer 2 — 服务抽象
│   ├── registry.py
│   ├── memory_store.py
│   ├── settings.py
│   ├── agent_runtime.py
│   ├── agent_registry.py
│   └── tool_registry.py
└── client/                # Layer 3 — 应用
```

---

## 缺少的抽象

当前项目中以下概念没有对应的 Protocol 定义，但在代码中频繁使用：

| 当前用法 | 缺少的抽象 | 建议定义位置 |
|----------|-----------|-------------|
| `Callable[[], Any]` (LLMProviderFactory) | `LLMProviderFactory` Protocol | `llm/llm.py` 或 `agent/runtime.py` |
| `ConversationMemory()` 直接调用 | `MemoryFactory = Callable[[], Memory]` | `memory_store.py` |
| `SimpleAgent(...)` 直接调用 | `AgentFactory` Protocol | `agent/runtime.py` |
| `memory_store: Any` | `IMemoryStore` Protocol | `memory_store.py` |
| `Registry[T]` 具体类 | `RegistryProtocol[T]` | `registry.py` |

所有缺少的 Protocol 都应遵循已有的模式：定义在相关功能的模块中，与实现解耦。

---

## 目录结构

```
src/minimal_harness/
├── __init__.py                 # 公开 API 聚合导出
├── types.py                    # Layer 1 — 事件 dataclass、TypedDict、类型别名
├── memory.py                   # Layer 1 — Memory Protocol、Message 类型、ConversationMemory
├── registry.py                 # Layer 2 — Registry[T] 泛型基底
├── memory_store.py             # Layer 2 — MemoryStore + _ManagedMemory
├── settings.py                 # Layer 2 — 环境变量配置
├── agent/
│   ├── protocol.py             # Layer 1 — Agent Protocol
│   ├── simple.py               # Layer 1 — SimpleAgent 实现
│   ├── runtime.py              # Layer 2 — AgentRuntime + AgentRuntimeProtocol
│   └── registry.py             # Layer 2 — AgentRegistry + AgentRegistryProtocol + AgentMetadata
├── llm/
│   ├── llm.py                  # Layer 1 — LLMProvider Protocol、LLMResponse、Stream
│   ├── openai.py               # Layer 1 — OpenAILLMProvider
│   └── anthropic.py            # Layer 1 — AnthropicLLMProvider
├── tool/
│   ├── base.py                 # Layer 1 — Tool Protocol、StreamingTool、ToolRegistryProtocol
│   ├── registry.py             # Layer 2 — ToolRegistry
│   ├── registration.py         # Layer 2 — @register_tool 装饰器
│   ├── external_loader.py      # Layer 2 — 外部脚本工具加载
│   ├── wrapper.py              # Layer 2 — ExternalToolWrapper (子进程执行)
│   └── built_in/               # Layer 1 — bash、local_file_operation
└── client/
    ├── __init__.py             # 向后兼容事件别名导出
    ├── events.py               # 向后兼容 shim（待移除）
    └── built_in/               # Layer 3 — TUI 客户端
        ├── app.py              # TUIApp (入口)
        ├── context.py          # AppContext + TUIConfig + create_llm_provider
        ├── session.py          # Session Protocol + ConversationSession
        ├── session_controller.py
        ├── session_factory.py
        ├── session_replayer.py
        ├── agent_manager.py
        ├── display.py          # ChatDisplay
        ├── buffer.py           # StreamBuffer
        ├── streaming_controller.py
        ├── chat_widgets.py     # 消息组件 (UserMsg, AssistantMsg, ...)
        ├── markdown_styles.py  # 自定义 markdown 渲染
        ├── renderer.py         # 静态格式化函数
        ├── slash_handler.py    # 斜杠命令处理
        ├── export_tracker.py   # 导出追踪
        ├── export_presenter.py # SVG 导出
        ├── widgets.py          # Banner, ChatInput, SessionNotification
        ├── messages.py         # Textual Message 类型
        ├── modals.py           # 模态对话框
        ├── constants.py        # 主题、常量
        ├── app.tcss            # Textual CSS
        ├── config/             # 配置持久化
        └── actions/            # 用户动作实现
```
