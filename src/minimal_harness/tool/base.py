from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, AsyncIterator

from openai.types.chat import ChatCompletionToolUnionParam

from minimal_harness.types import (
    ChunkCallback,
    ExecutionStartCallback,
    ProgressCallback,
    StreamingToolFunction,
    ToolEndCallback,
    ToolFunction,
    ToolStartCallback,
)

if TYPE_CHECKING:
    from minimal_harness.agent.protocol import Agent


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


class AgenticTool(StreamingTool):
    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict,
        agent: Agent,
        fn: StreamingToolFunction | None = None,
    ):
        super().__init__(name, description, parameters, fn or self._execute)
        self.agent = agent
        self._on_tool_start: ToolStartCallback | None = None
        self._on_tool_end: ToolEndCallback | None = None
        self._on_execution_start: ExecutionStartCallback | None = None
        self._on_tool_progress: ProgressCallback | None = None
        self._on_chunk: ChunkCallback[Any] | None = None

    async def _execute(self, **kwargs: Any) -> AsyncIterator[str]:
        user_message = json.dumps(kwargs, ensure_ascii=False)
        result = await self.agent.run(
            user_input=[{"type": "text", "text": user_message}],
            on_tool_start=self._on_tool_start,
            on_tool_end=self._on_tool_end,
            on_execution_start=self._on_execution_start,
            on_tool_progress=self._on_tool_progress,
            on_chunk=self._on_chunk,
        )
        yield result


BaseTool = Tool | StreamingTool | InteractiveTool
