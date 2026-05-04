# Architecture

## Overview

minimal-harness 采用三层抽象架构，自底向上分别为：

- **Layer 1 — 核心抽象层 (Core Abstractions)**：定义 Agent、Tool、Memory、LLMProvider 等基础概念及其事件体系。该层不依赖任何具体应用，是整个系统的基石。
- **Layer 2 — 面向服务层 (Service Abstractions)**：在 Layer 1 之上提供 Runtime（运行编排）、Registry（注册发现）、MemoryStore（持久化）等面向运行时的服务能力。该层可以独立于具体客户端使用。
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

## 当前架构存在的问题

### 1. Protocol 放置不一致

`ToolRegistryProtocol` 定义在 `tool/base.py`，与 `Tool` 协议共处一个文件；而 `AgentRegistryProtocol` 定义在 `agent/registry.py`，与其实现 `AgentRegistry` 共存。两个 Registry Protocol 的放置策略不统一。

**建议**: 将所有 Protocol 统一放置在 `protocol.py` 或独立的 `_protocol.py` 文件中，与实现分离。

### 2. LLMChunk 事件是多余的包装

`LLMChunk` 事件（`types.py:89`）仅包装一个 `LLMChunkDelta | None`：

```python
@dataclass
class LLMChunk:
    chunk: LLMChunkDelta | None
```

`LLMChunkDelta` 本身已是数据类，再套一层 `LLMChunk` 没有增加任何信息。`Agent.run()` 可以直接产出 `LLMChunkDelta` 作为事件，避免不必要的嵌套。

**建议**: 移除 `LLMChunk`，在 `AgentEvent` union 中直接使用 `LLMChunkDelta`。

### 3. AgentRuntime 反向依赖 SimpleAgent

`AgentRuntime._create_agent()` 通过 lazy import 直接实例化 `SimpleAgent`（`agent/runtime.py:94-97`）。Layer 2 的服务组件直接依赖 Layer 1 的具体实现，破坏了分层原则。

```python
def _create_agent(self, agent_type: str) -> Agent:
    from minimal_harness.agent.simple import SimpleAgent
    if agent_type == "simple":
        llm_provider = self._llm_provider_factory()
        return SimpleAgent(llm_provider=llm_provider)
    raise ValueError(f"Unknown agent type: {agent_type}")
```

**建议**: 引入 AgentFactory 抽象或依赖注入，让 Runtime 只依赖 `Agent` Protocol。

### 4. MemoryStore 的层级归属模糊

`MemoryStore` 的文档字符串自述为 "Layer 2 service abstraction"，但文件位于 `src/minimal_harness/memory_store.py`，与 Layer 1 的 `memory.py` 并列。项目没有为 Layer 2 开辟独立的目录空间（如 `service/`），导致层级边界在物理结构上不清晰。

**建议**: 将 Layer 2 组件（Runtime、Registry、MemoryStore）移至独立目录，如 `src/minimal_harness/service/`。

### 5. Registry 签名字段不一致

两个 Registry Protocol 的 `register` 方法采用不同签名：
- `ToolRegistryProtocol.register(tool: Tool)` — 按对象注册
- `AgentRegistryProtocol.register(*, name, description, ...)` — 按字段注册

这导致无法用 `RegistryProtocol[T]` 泛型协议抽象两者的共性（都是 name-keyed CRUD）。同时，`AgentRegistryProtocol` 暴露了 `add_listener` / `remove_listener`，而 `ToolRegistryProtocol` 没有（尽管基类 `Registry` 有此能力）。

**建议**: 统一注册签名为 `register(item: T)` 模式，或抽象出 `RegistryProtocol[T]`。

### 6. AgentRuntimeProtocol 的私有属性暴露

```python
@runtime_checkable
class AgentRuntimeProtocol(Protocol):
    _agent_registry: AgentRegistryProtocol
    _memory_store: Any
    _tool_registry: ToolRegistryProtocol
```

协议要求实现类拥有以下划线开头的"私有"属性，语义上矛盾。这些属性是 `run()` 方法的运行时依赖，应作为方法参数传入或通过 Protocol 提供 getter 方法。

**建议**: 将这些依赖作为 `run()` 的参数，或通过 `AgentRuntimeProtocol` 的公共属性/方法暴露。

### 7. Agent Protocol 的弱契约

`Agent` Protocol 将 `memory`、`tools`、`stop_event` 声明为可选（`| None = None`），注释解释为 "structural compatibility"。但 `SimpleAgent.run()` 在运行时用 `assert` 强制要求它们非 None。Protocol 定义了过于宽松的契约，实际运行时却是严格约束。

**建议**: 从 Protocol 签名中移除 `= None` 默认值，或改为非 Optional 类型。不应为"结构兼容性"削弱 Protocol 的正确性定义。

### 8. Layer 1 类型别名泄漏了上下文假设

```python
# memory.py:29
InputContentPart = TextContentPart
ExtendedInputContentPart = FileContentPart | ImageContentPart | TextContentPart
```

`InputContentPart = TextContentPart` 将多模态输入窄化为纯文本，且命名暗示这是"输入"范畴的概念。这个别名出现在 Layer 1 的 `memory.py` 中，但文本/图像/文件的区分是下游消费者（LLMProvider、客户端）的关切。Layer 1 应只定义消息结构，不应引入"输入就是文本"这样的假设。

**建议**: 消除 `InputContentPart` 别名，统一使用明确的 `TextContentPart` 或 `ContentPart` union。

### 9. _ManagedMemory 暴露了内部实现

`MemoryStore.create_memory()` 和 `get_memory()` 的返回类型是 `_ManagedMemory`（下划线前缀），而 `_ManagedMemory` 实现了 `Memory` 协议。调用方不应知道返回的是代理类——它们只需要 `Memory` 接口。

**建议**: 将返回类型标注为 `Memory`，隐藏 `_ManagedMemory` 实现细节。

### 10. 工具注册的双重机制

系统同时支持两种工具注入方式：
- `ToolRegistry.register(tool)` / `register_external_tool(...)` — 静态注册
- `AgentRuntime._inject_runtime_tools()` — 每次运行动态注入 `handoff` / `discover_agents`

运行时工具 (`handoff`, `discover_agents`) 不在 ToolRegistry 中出现，但在 Agent 执行时会被注入到工具列表中。这意味着 ToolRegistry 并不是"所有可用工具"的完全集合，存在隐式的工具来源。

**建议**: 将运行时工具也纳入 ToolRegistry 管理，或在架构文档中明确标注"系统工具不在 Registry 中"。

### 11. AgentRegistry 的注册 key 与查找 key 不一致

`AgentRegistry.register()` 默认以 `name` 作为 `metadata_id` 进行 `_register()`。但协议方法 `get(name)` 传入的也是 `name`，实际查找的是 `metadata_id`。这两个概念在默认情况下等价，但如果用户显式指定了不同的 `metadata_id`，`get(name)` 将无法找到该条目——因为底层 `Registry._data` 的 key 是 `metadata_id`。

**建议**: 明确区分对外标识（name）和内部 key（metadata_id），在 lookup 时做正确的键映射。

---

## 目录结构

```
src/minimal_harness/
├── __init__.py                 # 公开 API 聚合导出
├── types.py                    # Layer 1 — 事件 dataclass、TypedDict、类型别名
├── memory.py                   # Layer 1 — Memory Protocol、Message 类型、ConversationMemory
├── registry.py                 # Layer 2 — Registry[T] 泛型基底
├── memory_store.py             # Layer 2 — MemoryStore + _ManagedMemory
├── settings.py                 # Layer 1 — 环境变量配置
├── agent/
│   ├── protocol.py             # Layer 1 — Agent Protocol
│   ├── simple.py               # Layer 1 — SimpleAgent 实现
│   ├── runtime.py              # Layer 2 — AgentRuntime + AgentRuntimeProtocol
│   └── registry.py             # Layer 2 — AgentRegistry + AgentRegistryProtocol
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
    ├── __init__.py             # 向后兼容导出
    ├── events.py               # 向后兼容 shim
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
