from typing import Any, Literal, Protocol, TypedDict


class TextContentPart(TypedDict):
    type: Literal["text"]
    text: str


class FileMetadata(TypedDict):
    file_id: str
    file_name: str
    file_size: int
    backend_type: str


class FileContentPart(TypedDict):
    type: Literal["file"]
    file: FileMetadata


InputContentPart = TextContentPart
ExtendedInputContentPart = FileContentPart | TextContentPart


class SystemMessage(TypedDict):
    role: Literal["system"]
    content: str


class UserMessage(TypedDict):
    role: Literal["user"]
    content: list[InputContentPart]


class AssistantMessage(TypedDict):
    role: Literal["assistant"]
    content: str | None
    tool_calls: list[Any] | None


class ToolMessage(TypedDict):
    role: Literal["tool"]
    tool_call_id: str
    content: str


Message = SystemMessage | UserMessage | AssistantMessage | ToolMessage


class TokenUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


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
        self._messages.clear()

    def add_usage(self, usage: TokenUsage) -> None:
        self._total_usage["prompt_tokens"] = usage["prompt_tokens"]
        self._total_usage["completion_tokens"] = usage["completion_tokens"]
        self._total_usage["total_tokens"] = usage["total_tokens"]

    def get_total_usage(self) -> TokenUsage:
        return self._total_usage.copy()
