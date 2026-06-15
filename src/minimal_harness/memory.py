import json
import time
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Literal,
    NotRequired,
    Protocol,
    TypedDict,
    runtime_checkable,
)

from minimal_harness.types import CompactionEvent, TokenUsage


class TextContentPart(TypedDict):
    type: Literal["text"]
    text: str


class ImageContentPart(TypedDict):
    type: Literal["image"]
    url: str
    data: NotRequired[str]
    media_type: NotRequired[str]


class FileMetadata(TypedDict):
    file_id: str
    file_name: str
    file_size: int
    backend_type: str


class FileContentPart(TypedDict):
    type: Literal["file"]
    file: FileMetadata


InputContentPart = TextContentPart
ExtendedInputContentPart = FileContentPart | ImageContentPart | TextContentPart


class SystemMessage(TypedDict):
    role: Literal["system"]
    content: str


class UserMessage(TypedDict):
    role: Literal["user"]
    content: list[InputContentPart] | list[ExtendedInputContentPart]


class AssistantMessage(TypedDict):
    role: Literal["assistant"]
    content: str | None
    tool_calls: list[Any] | None


class ToolMessage(TypedDict):
    role: Literal["tool"]
    tool_call_id: str
    content: str
    progress: NotRequired[list[str]]
    meta: NotRequired[dict]


class ReasoningMessage(TypedDict):
    role: Literal["reasoning"]
    content: str


class CompactionMessage(TypedDict):
    """Synthetic message produced by ``Memory.compact()``.

    Stored on disk exactly as-is (so session replay can render the
    original compaction metadata), but stripped from LLM-visible context
    and remapped to an ``AssistantMessage`` carrying the same content —
    just like ``ReasoningMessage``, except that the LLM *does* see
    compactions (as assistant turns), not the reasoning chain.
    """

    role: Literal["compaction"]
    content: str
    meta: NotRequired[dict[str, Any]]


Message = (
    SystemMessage
    | UserMessage
    | AssistantMessage
    | ToolMessage
    | ReasoningMessage
    | CompactionMessage
)


def system_message(content: str) -> SystemMessage:
    return {"role": "system", "content": content}


def user_message(
    content: list[InputContentPart] | list[ExtendedInputContentPart],
) -> UserMessage:
    return {"role": "user", "content": content}


def assistant_message(
    content: str | None, tool_calls: list[Any] | None = None
) -> AssistantMessage:
    return {"role": "assistant", "content": content, "tool_calls": tool_calls}


def tool_message(
    tool_call_id: str,
    content: str,
    progress: list[str] | None = None,
    meta: dict | None = None,
) -> ToolMessage:
    msg: ToolMessage = {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    }
    if progress:
        msg["progress"] = progress
    if meta:
        msg["meta"] = meta
    return msg


def reasoning_message(content: str) -> ReasoningMessage:
    return {"role": "reasoning", "content": content}


class MemoryData(TypedDict):
    messages: list[Message]
    usage: TokenUsage
    extra: dict[str, Any]


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


@runtime_checkable
class MemoryStoreProtocol(Protocol):
    """Minimal persistence contract: resolve a ``Memory`` by ID.

    The SDK's :class:`AgentRuntime` only needs ``get_session()`` — the
    richer ``Session``-shaped store lives in downstream packages
    (mh-orchestration-service, mh-tui) where session identity
    (user_id, scenario_id, …) is meaningful.
    """

    async def get_session(self, session_id: str) -> Memory | None: ...


class ConversationMemory:
    def __init__(self) -> None:
        self._messages: list[Message] = []
        self._total_usage: TokenUsage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._extra: dict[str, Any] = {}
        self._persisted_count: int = 0

    @property
    def memory_id(self) -> str:
        return self._extra.get("memory_id", "")

    @property
    def title(self) -> str | None:
        return self._extra.get("title")

    @property
    def agent_name(self) -> str:
        return self._extra.get("agent_name", "")

    @property
    def created_at(self) -> str:
        return self._extra.get("created_at", "")

    def flush(self) -> None:
        pass

    async def add_message(self, message: Message) -> None:
        self._messages.append(message)

    def get_all_messages(self) -> list[Message]:
        return self._messages.copy()

    def get_persisted_count(self) -> int:
        return self._persisted_count

    def get_new_messages(self) -> list[Message]:
        return self._messages[self._persisted_count :]

    def mark_all_persisted(self) -> None:
        self._persisted_count = len(self._messages)

    def set_persisted_count(self, count: int) -> None:
        self._persisted_count = count

    @property
    def _forward_offset(self) -> int:
        return self._extra.get("compact_offset", 0)

    @_forward_offset.setter
    def _forward_offset(self, value: int) -> None:
        self._extra["compact_offset"] = value

    def get_forward_messages(self) -> list[Message]:
        """Return the messages visible to the LLM.

        - ``reasoning`` messages are stripped (they're internal chain-of-
          thought, never sent to the LLM).
        - ``compaction`` messages are re-projected to ``role="assistant"``
          so the LLM sees the prior summary as part of the assistant's
          historical turns rather than as system-injected context.
        - Other roles (``user``, ``tool``, ``assistant``, ``system``) pass
          through unchanged.
        """
        transformed: list[Message] = []
        for m in self._messages:
            role = m.get("role")
            if role == "reasoning":
                continue
            if role == "compaction":
                # The CompactionMessage's content is always a string
                # (it's a summary text by construction). The runtime
                # type of m.get("content") is widened by the Message
                # union; narrow it explicitly here.
                _raw_content = m.get("content")
                assistant_view: AssistantMessage = {
                    "role": "assistant",
                    "content": _raw_content
                    if isinstance(_raw_content, str)
                    else "",
                    "tool_calls": None,
                }
                transformed.append(assistant_view)
                continue
            transformed.append(m)
        return transformed[self._forward_offset:]

    def clear_messages(self) -> None:
        self._messages.clear()

    def set_message_usage(self, usage: TokenUsage) -> None:
        self._total_usage["prompt_tokens"] += usage["prompt_tokens"]
        self._total_usage["completion_tokens"] += usage["completion_tokens"]
        self._total_usage["total_tokens"] += usage["total_tokens"]

    def get_message_usage(self) -> TokenUsage:
        return self._total_usage.copy()

    def dump_memory(self) -> MemoryData:
        return {
            "messages": self._messages.copy(),
            "usage": self._total_usage.copy(),
            "extra": self._extra.copy(),
        }

    def dump_memory_json(self, indent: int | None = 2) -> str:
        return json.dumps(
            self.dump_memory(), indent=indent, ensure_ascii=False, default=str
        )

    def load_memory(self, data: MemoryData) -> None:
        self._messages = data["messages"].copy()
        self._total_usage = data["usage"].copy()
        self._extra = data.get("extra", {}).copy()
        self._persisted_count = len(self._messages)

    def load_memory_json(self, data: str) -> None:
        parsed: MemoryData = json.loads(data)
        self.load_memory(parsed)

    async def compact(
        self,
        summarizer: Callable[[list[Message], str | None], AsyncIterator[str]],
        keep_recent: int,
        prompt_tokens: int = 0,
    ) -> AsyncIterator[CompactionEvent]:
        """Stream-compact: fold older messages into a summary, keep the tail.

        Yields ``CompactionStart`` once, zero or more ``CompactionChunk``s, and
        exactly one ``CompactionEnd`` (with ``error`` set on failure). On
        failure the buffer is left untouched and ``dropped_message_count`` is
        reported as 0 in the end event.

        The synthetic ``CompactionMessage`` (role="compaction") is inserted
        at index 0 with a ``meta`` field carrying dropped count, previous
        summary length, etc. ``_forward_offset`` stays at 0 — the compaction
        message is the natural start of the compacted conversation, and
        ``get_forward_messages()`` re-projects it to ``role="assistant"``
        before handing the buffer to the LLM.
        """
        from minimal_harness.types import (
            CompactionChunk,
            CompactionEnd,
            CompactionStart,
        )

        msgs = self._messages
        offset = self._forward_offset
        end = len(msgs) - keep_recent
        if end <= offset:
            return

        existing_summary: str | None = None
        start = offset
        if msgs and msgs[0].get("role") == "compaction":
            _content = msgs[0].get("content")
            if isinstance(_content, str):
                existing_summary = _content
                start = 1

        to_summarize = msgs[start:end]
        if not to_summarize:
            return

        yield CompactionStart(
            dropped_message_count=len(to_summarize),
            existing_summary=existing_summary,
            keep_recent=keep_recent,
            prompt_tokens=prompt_tokens,
        )

        accumulated = ""
        start_time = time.time()
        error_msg: str | None = None

        try:
            async for delta in summarizer(list(to_summarize), existing_summary):
                if not delta:
                    continue
                accumulated += delta
                yield CompactionChunk(delta=delta, accumulated=accumulated)
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"

        if error_msg is None and accumulated:
            compaction_meta: dict[str, Any] = {
                "dropped_count": len(to_summarize),
                "keep_recent": keep_recent,
                "previous_summary_chars": (
                    len(existing_summary) if existing_summary else 0
                ),
                "timestamp": time.time(),
            }
            summary_message: Message = {
                "role": "compaction",
                "content": accumulated,
                "meta": compaction_meta,
            }
            new_messages: list[Message] = [summary_message]
            new_messages.extend(msgs[end:])
            self._messages = new_messages
            self._forward_offset = 0
            dropped = len(to_summarize)
            new_offset = 0
            final_summary = accumulated
        else:
            # Failure: do NOT report the partial accumulated text as a
            # ``summary`` — downstream consumers (and the CompactionAgent)
            # use a non-empty ``CompactionEnd.summary`` to decide whether
            # to emit a ``MessageEvent(role="compaction")`` to the
            # frontend. A truncated partial summary is not a valid fold.
            dropped = 0
            new_offset = self._forward_offset
            final_summary = ""

        yield CompactionEnd(
            summary=final_summary,
            dropped_message_count=dropped,
            new_offset=new_offset,
            duration=time.time() - start_time,
            error=error_msg,
        )
