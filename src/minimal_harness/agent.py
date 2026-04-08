from typing import cast

from minimal_harness.llm import ChunkCallback, LLMProvider, ToolExecutor
from minimal_harness.memory import ConversationMemory, Memory, Message
from minimal_harness.tool import Tool


class Agent:
    def __init__(
        self,
        llm_provider: LLMProvider,
        tools: list[Tool] | None = None,
        max_iterations: int = 10,
        memory: Memory | None = None,
        tool_executor: ToolExecutor | None = None,
    ):
        self._llm_provider = llm_provider
        self._tools: dict[str, Tool] = {t.name: t for t in (tools or [])}
        self._tool_executor = tool_executor or ToolExecutor(self._tools)
        self._max_iterations = max_iterations
        self._memory = memory or ConversationMemory()

    async def run(
        self,
        user_input: str,
        on_chunk: ChunkCallback | None = None,
    ) -> str:
        self._memory.add_message({"role": "user", "content": user_input})

        for _ in range(self._max_iterations):
            response = await self._llm_provider.chat(
                messages=self._memory.get_all_messages(),
                tools=list(self._tools.values()),
                on_chunk=on_chunk,
            )

            async for _ in response:
                pass

            llm_response = response.response
            self._memory.add_message(
                cast(
                    Message,
                    {
                        "role": "assistant",
                        "content": llm_response.content,
                        "tool_calls": llm_response.tool_calls or None,
                    },
                )
            )

            if not llm_response.tool_calls:
                return str(llm_response.content) or ""

            results = await self._tool_executor.execute(llm_response.tool_calls)
            for msg in results:
                self._memory.add_message(msg)

        raise RuntimeError(
            f"Agent exceeded maximum iterations ({self._max_iterations})"
        )
