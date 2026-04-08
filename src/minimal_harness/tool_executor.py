import json
import asyncio
from typing import Any

from minimal_harness.llm import ToolCall
from minimal_harness.memory import Message
from minimal_harness.tool import Tool


class ToolExecutor:
    def __init__(self, tools: dict[str, Tool]):
        self._tools = tools

    async def execute(self, tool_calls: list[ToolCall]) -> list[Message]:
        tasks = [self._execute_single(tc) for tc in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        messages: list[Message] = []
        for tc, result in zip(tool_calls, results):
            if isinstance(result, Exception):
                content = f"[Tool Error] {tc['function']['name']}: {result}"
            else:
                content = (
                    json.dumps(result, ensure_ascii=False)
                    if not isinstance(result, str)
                    else result
                )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": content,
                }
            )

        return messages

    async def _execute_single(self, tc: ToolCall) -> Any:
        name = tc["function"]["name"]
        raw_args = tc["function"]["arguments"]

        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")

        args = json.loads(raw_args) if raw_args else {}
        print(f"[Tool Call] {name}({args})")
        return await self._tools[name].fn(**args)
