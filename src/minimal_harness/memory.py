from typing import Protocol, TypedDict, Literal, Any


class SystemMessage(TypedDict):
    role: Literal["system"]
    content: str


class UserMessage(TypedDict):
    role: Literal["user"]
    content: str


class AssistantMessage(TypedDict):
    role: Literal["assistant"]
    content: str | None
    tool_calls: list[Any] | None


class ToolMessage(TypedDict):
    role: Literal["tool"]
    tool_call_id: str
    content: str


Message = SystemMessage | UserMessage | AssistantMessage | ToolMessage


class Memory(Protocol):
    def add_message(self, message: Message) -> None: ...
    def get_all_messages(self) -> list[Message]: ...
    def clear_messages(self) -> None: ...


class ConversationMemory:
    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        self._messages: list[Message] = [{"role": "system", "content": system_prompt}]

    def add_message(self, message: Message) -> None:
        self._messages.append(message)

    def get_all_messages(self) -> list[Message]:
        return self._messages.copy()

    def clear_messages(self) -> None:
        self._messages.clear()
