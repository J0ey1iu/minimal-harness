from openai.types.chat import ChatCompletionToolUnionParam
from typing import Any, Callable, Awaitable

ToolFunction = Callable[..., Awaitable[Any]]


class Tool:
    def __init__(self, name: str, description: str, parameters: dict, fn: ToolFunction):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.fn = fn

    def to_schema(self) -> ChatCompletionToolUnionParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
