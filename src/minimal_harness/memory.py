from typing import Any, Literal, Protocol, TypedDict

from minimal_harness.types import TokenUsage


class TextContentPart(TypedDict):
    type: Literal["text"]
    text: str


class ImageContentPart(TypedDict):
    type: Literal["image"]
    url: str


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


Message = SystemMessage | UserMessage | AssistantMessage | ToolMessage


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


def tool_message(tool_call_id: str, content: str) -> ToolMessage:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


class Memory(Protocol):
    def add_message(self, message: Message) -> None: ...
    def get_all_messages(self) -> list[Message]: ...
    def clear_messages(self) -> None: ...
    def add_usage(self, usage: TokenUsage) -> None: ...
    def get_total_usage(self) -> TokenUsage: ...


class ConversationMemory:
    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        self._messages: list[Message] = [{"role": "system", "content": system_prompt}]
        self._total_usage: TokenUsage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def add_message(self, message: Message) -> None:
        self._messages.append(message)

    def get_all_messages(self) -> list[Message]:
        return self._messages.copy()

    def clear_messages(self) -> None:
        system_message = self._messages[0]
        self._messages.clear()
        self._messages.append(system_message)

    def add_usage(self, usage: TokenUsage) -> None:
        self._total_usage["prompt_tokens"] = usage["prompt_tokens"]
        self._total_usage["completion_tokens"] = usage["completion_tokens"]
        self._total_usage["total_tokens"] = usage["total_tokens"]

    def get_total_usage(self) -> TokenUsage:
        return self._total_usage.copy()
