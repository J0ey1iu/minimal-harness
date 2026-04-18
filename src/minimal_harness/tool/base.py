from typing import Any, AsyncIterator, Awaitable, Callable

from openai.types.chat import ChatCompletionToolUnionParam

ToolFunction = Callable[..., Awaitable[Any]]
UserInputCallback = Callable[[str], Awaitable[Any]]
StreamingToolFunction = Callable[..., AsyncIterator[Any]]
ProgressCallback = Callable[[Any], Awaitable[None]]


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


class InteractiveTool:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict,
        fn_first: ToolFunction,
        fn_final: ToolFunction,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.fn_first = fn_first
        self.fn_final = fn_final

    async def execute_first(self, **kwargs: Any) -> Any:
        return await self.fn_first(**kwargs)

    async def execute_final(self, user_input: str, **kwargs: Any) -> Any:
        return await self.fn_final(user_input, **kwargs)

    def to_schema(self) -> ChatCompletionToolUnionParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class StreamingTool:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict,
        fn: StreamingToolFunction,
    ):
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
