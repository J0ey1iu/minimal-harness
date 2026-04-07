import json
import asyncio
from typing import Any, Callable, Awaitable, TypedDict, cast
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionChunk

from minimal_harness.tool import Tool
from minimal_harness.memory import ConversationMemory, Memory, Message


ChunkCallback = Callable[[ChatCompletionChunk | None, bool], Awaitable[None]]


class ToolCallFunction(TypedDict):
    name: str
    arguments: str


class ToolCall(TypedDict):
    id: str
    type: str
    function: ToolCallFunction


class Agent:
    def __init__(
        self,
        model: str = "minimax-m2.1",
        system_prompt: str = "You are a helpful assistant.",
        tools: list[Tool] | None = None,
        max_iterations: int = 10,
        client: AsyncOpenAI | None = None,
        memory: Memory | None = None,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.tools: dict[str, Tool] = {t.name: t for t in (tools or [])}
        self.max_iterations = max_iterations
        self.client = client or AsyncOpenAI()
        self.memory = memory or ConversationMemory(system_prompt=system_prompt)

    async def run(
        self,
        user_input: str,
        on_chunk: ChunkCallback | None = None,
    ) -> str:
        self.memory.add_message({"role": "user", "content": user_input})

        for iteration in range(self.max_iterations):
            response_message, tool_calls = await self._chat_stream(on_chunk)

            self.memory.add_message(cast(Message, response_message))

            if not tool_calls:
                return str(response_message.get("content")) or ""

            await self._execute_tool_calls(tool_calls)

        raise RuntimeError(f"Agent exceeded maximum iterations ({self.max_iterations})")

    async def _chat_stream(
        self,
        on_chunk: ChunkCallback | None,
    ) -> tuple[ChatCompletionMessageParam, list[ToolCall]]:
        """
        Initiate streaming request, return (assistant_message_dict, tool_calls_list)
        """
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=self.memory.get_all_messages(),  # type: ignore[arg-type]
            tools=[t.to_schema() for t in self.tools.values()],
            tool_choice="auto" if self.tools else "none",
            stream=True,
        )

        content_parts = []
        tool_calls_acc = {}

        async for chunk in stream:
            if on_chunk:
                await on_chunk(chunk, False)

            delta = chunk.choices[0].delta if chunk.choices else None

            if delta is None:
                continue

            if delta.content:
                content_parts.append(delta.content)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = ToolCall(
                            id="",
                            type="function",
                            function=ToolCallFunction(name="", arguments=""),
                        )
                    acc = tool_calls_acc[idx]
                    if tc_delta.id:
                        acc["id"] += tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            acc["function"]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            acc["function"]["arguments"] += tc_delta.function.arguments

        if on_chunk:
            await on_chunk(None, True)

        content = "".join(content_parts) or None
        tool_calls = list(tool_calls_acc.values()) if tool_calls_acc else []

        assistant_message: ChatCompletionMessageParam = {
            "role": "assistant",
            "content": content,
        }
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls

        return assistant_message, tool_calls

    async def _execute_tool_calls(self, tool_calls: list[ToolCall]):
        """Concurrently execute all tool calls, append results to messages"""
        tasks = [self._execute_single_tool(tc) for tc in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for tc, result in zip(tool_calls, results):
            if isinstance(result, Exception):
                content = f"[Tool Error] {tc['function']['name']}: {result}"
            else:
                content = (
                    json.dumps(result, ensure_ascii=False)
                    if not isinstance(result, str)
                    else result
                )

            self.memory.add_message(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": content,
                }
            )

    async def _execute_single_tool(self, tc: ToolCall) -> Any:
        name = tc["function"]["name"]
        raw_args = tc["function"]["arguments"]

        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")

        args = json.loads(raw_args) if raw_args else {}
        print(f"[Tool Call] {name}({args})")
        return await self.tools[name].fn(**args)
