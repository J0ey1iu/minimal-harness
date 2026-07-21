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
    meta: NotRequired[dict[str, Any]]


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
    meta: dict[str, Any] | None = None,
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
    replay_messages: NotRequired[list[Message]]
    persisted_count: NotRequired[int]
    max_persisted_sort_order: NotRequired[int]
    forward_offset: NotRequired[int]


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
    def get_replay_messages(self) -> list[Message]: ...
    def clear_messages(self) -> None: ...
    def set_message_usage(self, usage: TokenUsage) -> None: ...
    def reset_message_usage(self) -> None: ...
    def get_message_usage(self) -> TokenUsage: ...
    def dump_memory(self) -> MemoryData: ...
    def load_memory(self, data: MemoryData) -> None: ...
    def get_persisted_count(self) -> int: ...
    def get_new_messages(self) -> list[Message]: ...
    def mark_all_persisted(self) -> None: ...
    def set_persisted_count(self, count: int) -> None: ...
    def get_forward_offset(self) -> int: ...
    def set_forward_offset(self, offset: int) -> None: ...
    def compact(
        self,
        summarizer: Callable[[list[Message], str | None], AsyncIterator[str]],
        keep_recent: int,
        total_tokens: int = 0,
    ) -> AsyncIterator[CompactionEvent]: ...

    def compress_tool_messages(
        self,
        summarizer: Callable[[list[Message], str | None], AsyncIterator[str]],
        tool_token_threshold: int,
    ) -> AsyncIterator[CompactionEvent]: ...


class BaseMemory:
    """Default-implementation base for :class:`Memory` implementors.

    The SDK defines :class:`Memory` as a structural ``Protocol``, so
    duck-typed classes that implement the surface area satisfy the
    type checker. But structural matching has a real cost: when the
    protocol gains a new method (e.g. ``compact()`` in 0.7.0), every
    downstream implementor must add it or the agent loop crashes at
    runtime with ``AttributeError``. The downstream ``JsonlManagedSession``
    in mh-tui hit exactly this — see commit ``766f13c``.

    Subclassing :class:`BaseMemory` instead gives implementors a
    concrete contract: any abstract method that is left unimplemented
    will raise ``NotImplementedError`` at instantiation time, not
    buried in a streaming generator. The base also provides default
    no-op implementations for bookkeeping methods
    (``mark_all_persisted``, ``set_persisted_count``,
    ``get_persisted_count``, ``get_new_messages``,
    ``get_message_usage``, ``set_message_usage``) so subclasses only
    need to override the message-storage surface and persistence
    surface.
    """

    @property
    def memory_id(self) -> str:
        raise NotImplementedError

    @property
    def title(self) -> str | None:
        raise NotImplementedError

    @title.setter
    def title(self, value: str | None) -> None:
        raise NotImplementedError

    @property
    def agent_name(self) -> str:
        raise NotImplementedError

    @property
    def created_at(self) -> str:
        raise NotImplementedError

    async def add_message(self, message: Message) -> None:
        raise NotImplementedError

    def get_all_messages(self) -> list[Message]:
        raise NotImplementedError

    def get_forward_messages(self) -> list[Message]:
        raise NotImplementedError

    def get_replay_messages(self) -> list[Message]:
        raise NotImplementedError

    def clear_messages(self) -> None:
        raise NotImplementedError

    def set_message_usage(self, usage: TokenUsage) -> None:
        raise NotImplementedError

    def reset_message_usage(self) -> None:
        raise NotImplementedError

    def get_message_usage(self) -> TokenUsage:
        raise NotImplementedError

    def dump_memory(self) -> MemoryData:
        raise NotImplementedError

    def load_memory(self, data: MemoryData) -> None:
        raise NotImplementedError

    def get_persisted_count(self) -> int:
        return 0

    def get_new_messages(self) -> list[Message]:
        return []

    def mark_all_persisted(self) -> None:
        return None

    def set_persisted_count(self, count: int) -> None:
        return None

    def compact(
        self,
        summarizer: Callable[[list[Message], str | None], AsyncIterator[str]],
        keep_recent: int,
        prompt_tokens: int = 0,
    ) -> AsyncIterator[CompactionEvent]:
        """Default ``compact()`` that yields a single ``CompactionEnd`` reporting
        "not implemented". Implementors that store messages on disk MUST
        override this — :class:`CompactionAgent` will treat a
        ``NotImplementedError`` propagated from here as a soft failure
        (the assistant turn is preserved, the run continues), but the
        buffer will never be folded and the conversation will grow
        unbounded.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement Memory.compact()"
        )

    def compress_tool_messages(
        self,
        summarizer: Callable[[list[Message], str | None], AsyncIterator[str]],
        tool_token_threshold: int,
    ) -> AsyncIterator[CompactionEvent]:
        """Default ``compress_tool_messages()`` that reports "not implemented".
        Implementors that store messages on disk MUST override this —
        :class:`ToolCompactionAgent` will treat a ``NotImplementedError``
        propagated from here as a soft failure (the assistant turn is
        preserved, the run continues).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement Memory.compress_tool_messages()"
        )


@runtime_checkable
class MemoryStoreProtocol(Protocol):
    """Minimal persistence contract: resolve a ``Memory`` by ID.

    The SDK's :class:`AgentRuntime` only needs ``get_session()`` — the
    richer ``Session``-shaped store lives in downstream packages
    (mh-gateway, mh-tui) where session identity
    (user_id, scenario_id, …) is meaningful.
    """

    async def get_session(self, session_id: str) -> Memory | None: ...


class ConversationMemory:
    def __init__(self) -> None:
        self._messages: list[Message] = []
        # Monotonically-growing list of every message that was *ever* added,
        # including CompactionMessages inserted by compact(). This is never
        # mutated by compaction operations — it preserves the full raw history
        # for session replay, even when keep_recent=0 folds everything from
        # the live buffer.
        self._replay_history: list[Message] = []
        self._total_usage: TokenUsage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._extra: dict[str, Any] = {}
        self._persisted_count: int = 0
        # Separate counter for the maximum sort_order already written to the
        # backing store.  Normally it matches ``_persisted_count``, but
        # after a compaction the live buffer is rebuilt while the DB still
        # holds old rows; this counter ensures ``get_persisted_count()``
        # returns the correct next sort_order for ``save_memory()`` even
        # when ``_persisted_count`` has been reset to 0.
        self._max_persisted_sort_order: int = 0
        # Index into ``_messages`` where ``get_forward_messages()`` starts
        # slicing. After a compaction the summary is inserted at this
        # position, so the LLM sees the compacted view while ``_messages``
        # retains the full history for display and persistence.
        self._forward_offset: int = 0

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
        self._replay_history.append(message)

    def get_all_messages(self) -> list[Message]:
        """Return all messages (original + compaction summary + recent).

        After a compaction the compaction summary is inserted at the
        ``_forward_offset`` boundary; all prior messages are preserved.
        Reasoning messages are still present here (they are stripped
        only in :meth:`get_forward_messages`).

        .. deprecated::
            Prefer :meth:`get_replay_messages` for the full raw
            history (including messages that have been folded into
            a summary) and :meth:`get_forward_messages` for the
            LLM-visible projection. ``get_all_messages`` is kept for
            backwards compatibility — its semantic is "live buffer",
            which is sometimes useful (e.g. the
            :class:`~minimal_harness.agent.simple.SimpleAgent` uses
            it to look up the most recent assistant turn for
            ``response_text`` fallback).
        """
        return self._messages.copy()

    def get_persisted_count(self) -> int:
        # After compaction the live buffer is shorter than the number
        # of rows already written to the DB.  Return whichever is larger
        # so ``save_memory()`` always uses a monotonic sort_order.
        return max(self._persisted_count, self._max_persisted_sort_order)

    def get_new_messages(self) -> list[Message]:
        return self._messages[self._persisted_count :]

    def mark_all_persisted(self) -> None:
        self._persisted_count = len(self._messages)
        self._max_persisted_sort_order = max(
            self._max_persisted_sort_order, self._persisted_count
        )

    def set_persisted_count(self, count: int) -> None:
        self._persisted_count = count
        if count > self._max_persisted_sort_order:
            self._max_persisted_sort_order = count

    def get_forward_offset(self) -> int:
        return self._forward_offset

    def set_forward_offset(self, offset: int) -> None:
        self._forward_offset = offset

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
        for m in self._messages[self._forward_offset :]:
            role = m.get("role")
            if role == "reasoning":
                continue
            if role == "compaction":
                _raw_content = m.get("content")
                assistant_view: AssistantMessage = {
                    "role": "assistant",
                    "content": _raw_content if isinstance(_raw_content, str) else "",
                    "tool_calls": None,
                }
                transformed.append(assistant_view)
                continue
            transformed.append(m)
        return transformed

    def clear_messages(self) -> None:
        self._messages.clear()
        self._replay_history.clear()

    def get_replay_messages(self) -> list[Message]:
        return self._replay_history.copy()

    def set_message_usage(self, usage: TokenUsage) -> None:
        self._total_usage["prompt_tokens"] += usage["prompt_tokens"]
        self._total_usage["completion_tokens"] += usage["completion_tokens"]
        self._total_usage["total_tokens"] += usage["total_tokens"]

    def reset_message_usage(self) -> None:
        self._total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def get_message_usage(self) -> TokenUsage:
        return self._total_usage.copy()

    def dump_memory(self) -> MemoryData:
        return {
            "messages": self._messages.copy(),
            "usage": self._total_usage.copy(),
            "extra": self._extra.copy(),
            "replay_messages": self._replay_history.copy(),
            "persisted_count": self._persisted_count,
            "max_persisted_sort_order": self._max_persisted_sort_order,
            "forward_offset": self._forward_offset,
        }

    def dump_memory_json(self, indent: int | None = 2) -> str:
        return json.dumps(
            self.dump_memory(), indent=indent, ensure_ascii=False, default=str
        )

    def load_memory(self, data: MemoryData) -> None:
        self._messages.clear()
        self._replay_history.clear()
        for msg in data.get("messages", []):
            self._messages.append(msg)
        for msg in data.get("replay_messages", []):
            self._replay_history.append(msg)
        if not self._replay_history:
            # Backward-compat: old dumps without replay_messages.
            # Copy messages so at least the compacted buffer is visible.
            self._replay_history = [m for m in self._messages]
        u = data.get("usage")
        if u:
            self._total_usage = u.copy()
        self._extra = data.get("extra", {}).copy()
        self._persisted_count = len(self._messages)
        self._max_persisted_sort_order = max(
            self._persisted_count, data.get("max_persisted_sort_order", 0)
        )
        self._forward_offset = data.get("forward_offset", 0)

    def load_memory_json(self, data: str) -> None:
        parsed: MemoryData = json.loads(data)
        self.load_memory(parsed)

    async def compact(
        self,
        summarizer: Callable[[list[Message], str | None], AsyncIterator[str]],
        keep_recent: int,
        total_tokens: int = 0,
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
        if len(msgs) > offset and msgs[offset].get("role") == "compaction":
            _content = msgs[offset].get("content")
            if isinstance(_content, str):
                existing_summary = _content
                start = offset + 1

        to_summarize = msgs[start:end]
        if not to_summarize:
            return

        yield CompactionStart(
            dropped_message_count=len(to_summarize),
            existing_summary=existing_summary,
            keep_recent=keep_recent,
            total_tokens=total_tokens,
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
            msgs.insert(end, summary_message)
            self._forward_offset = end
            self._replay_history.append(summary_message)
            # The live buffer has been rebuilt — reset the prefix counter
            # to 0 so that get_new_messages() returns the full buffer.
            # But remember the old persisted count so get_persisted_count()
            # still returns a monotonic sort_order for save_memory().
            self._max_persisted_sort_order = max(
                self._max_persisted_sort_order, self._persisted_count
            )
            self._persisted_count = 0
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

    async def compress_tool_messages(
        self,
        summarizer: Callable[[list[Message], str | None], AsyncIterator[str]],
        tool_token_threshold: int,
    ) -> AsyncIterator[CompactionEvent]:
        """Compress ``role="tool"`` message content **in place**.

        Scans the forward buffer for ``role="tool"`` messages. If their
        estimated token count (content char count / 2) exceeds
        *tool_token_threshold*, calls *summarizer* to produce a summary
        **and replaces each tool message's content with a portion of the
        summary**. The original ``tool_call_id`` and other fields are
        preserved so the LLM API's structural requirement (every
        ``tool_call_id`` must have a matching tool response) is satisfied.

        Original messages are preserved in the replay history.

        Yields ``CompactionStart``, zero or more ``CompactionChunk``
        (streaming summary deltas), and one ``CompactionEnd``. On failure
        (summarizer raised) the buffer is left untouched and
        ``dropped_message_count`` is 0 in the end event.
        """
        from minimal_harness.types import (
            CompactionChunk,
            CompactionEnd,
            CompactionStart,
        )

        msgs = self._messages
        offset = self._forward_offset

        # Collect tool messages from forward buffer
        tool_indices: list[int] = []
        tool_msgs: list[Message] = []
        for i in range(offset, len(msgs)):
            if msgs[i].get("role") == "tool":
                tool_indices.append(i)
                tool_msgs.append(msgs[i])

        if not tool_msgs:
            return

        # If there is only one tool message and it is already a compressed
        # summary, skip to avoid re-compressing an already compressed result.
        if len(tool_msgs) == 1 and tool_msgs[0].get("meta", {}).get("compressed"):
            return

        # Estimate token count from content length
        total_chars = sum(len(str(m.get("content", ""))) for m in tool_msgs)
        estimated_tokens = total_chars // 2

        if estimated_tokens <= tool_token_threshold:
            return

        yield CompactionStart(
            dropped_message_count=len(tool_msgs),
            existing_summary=None,
            keep_recent=0,
            total_tokens=estimated_tokens,
        )

        accumulated = ""
        start_time = time.time()
        error_msg: str | None = None

        try:
            async for delta in summarizer(tool_msgs, None):
                if not delta:
                    continue
                accumulated += delta
                yield CompactionChunk(delta=delta, accumulated=accumulated)
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"

        if error_msg is None and accumulated:
            # Instead of merging into one message (which breaks the LLM
            # API's requirement that every tool_call_id has a matching
            # tool response), replace each tool message's content with a
            # slice of the summary. This preserves tool_call_id and other
            # structural fields.
            n = len(tool_msgs)
            base = len(accumulated) // n
            remainder = len(accumulated) % n
            pos = 0
            for i, idx in enumerate(tool_indices):
                chunk_size = base + (1 if i < remainder else 0)
                chunk = accumulated[pos : pos + chunk_size]
                pos += chunk_size

                old_content = msgs[idx].get("content", "")
                msgs[idx]["content"] = chunk  # type: ignore[typeddict-item]
                msgs[idx]["meta"] = {  # type: ignore[typeddict-item]
                    "compressed": True,
                    "dropped_count": n,
                    "original_chars": len(str(old_content)),
                    "timestamp": time.time(),
                }

                # Record original in replay history
                original = dict(msgs[idx])
                original["content"] = old_content
                original["meta"] = {"pre_compression": True}
                self._replay_history.append(original)  # type: ignore[arg-type]

            dropped = len(tool_msgs)
            new_offset = self._forward_offset
            final_summary = accumulated
        else:
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
