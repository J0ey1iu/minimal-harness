import asyncio
import json
import platform
import sys
from pathlib import Path
from typing import Any, cast

from openai import AsyncOpenAI

from minimal_harness.agent import OpenAIAgent
from minimal_harness.llm.openai import OpenAILLMProvider
from minimal_harness.memory import (
    ConversationMemory,
    ExtendedInputContentPart,
)
from minimal_harness.tool.registry import ToolRegistry


def extract_thinking(delta: Any) -> str | None:
    for attr in ("reasoning_content", "reasoning"):
        value = getattr(delta, attr, None)
        if value:
            return value

    provider_fields = getattr(delta, "provider_specific_fields", None)
    if provider_fields and isinstance(provider_fields, dict):
        value = provider_fields.get("reasoning_content")
        if value:
            return value

    model_extra = getattr(delta, "model_extra", None)
    if model_extra and isinstance(model_extra, dict):
        for key in ("reasoning_content", "reasoning"):
            value = model_extra.get(key)
            if value:
                return value

    return None


class SimpleStreamHandler:
    def __init__(self) -> None:
        self.response_text = ""
        self.thinking_text = ""
        self.tool_calls: list[dict[str, Any]] = []
        self.tool_call_args: dict[int, str] = {}
        self.response_started = False
        self.thinking_started = False

    async def on_chunk(self, chunk: Any | None, is_done: bool) -> None:
        if is_done or not chunk:
            return

        choices = getattr(chunk, "choices", None)
        if not choices:
            return

        delta = choices[0].delta
        if not delta:
            return

        reasoning = extract_thinking(delta)
        if reasoning:
            self._handle_thinking(reasoning)

        if delta.tool_calls:
            self._handle_tool_call_deltas(delta.tool_calls)

        if delta.content:
            self._handle_content(delta.content)

    def _handle_thinking(self, reasoning: str) -> None:
        if not self.thinking_started:
            sys.stdout.write("\n\x1b[90m[Thinking]\n")
            sys.stdout.flush()
            self.thinking_started = True

        if len(reasoning) > len(self.thinking_text) and reasoning.startswith(
            self.thinking_text
        ):
            new_part = reasoning[len(self.thinking_text) :]
            self.thinking_text = reasoning
        else:
            new_part = reasoning
            self.thinking_text += reasoning

        sys.stdout.write(new_part)
        sys.stdout.flush()

    def _end_thinking_block(self) -> None:
        if self.thinking_started:
            sys.stdout.write("\x1b[0m\n")
            sys.stdout.flush()

    def _handle_tool_call_deltas(self, tool_calls: Any) -> None:
        for tc in tool_calls:
            idx = tc.index
            if idx not in self.tool_call_args:
                self.tool_call_args[idx] = ""
                name = tc.function.name if tc.function and tc.function.name else "?"
                self._end_thinking_block()
                print(f"\x1b[93m[Tool call: {name}]\x1b[0m", flush=True)
                self.tool_calls.append({"index": idx, "name": name})
            if tc.function and tc.function.arguments:
                self.tool_call_args[idx] += tc.function.arguments

    def _handle_content(self, content: str) -> None:
        if not self.response_started:
            self._end_thinking_block()
            sys.stdout.write("\n\x1b[96mAssistant:\x1b[0m ")
            sys.stdout.flush()
            self.response_started = True

        self.response_text += content
        sys.stdout.write(content)
        sys.stdout.flush()

    async def on_tool_start(self, tool_call: Any, _: Any) -> None:
        name = tool_call["function"]["name"]
        print(f"\n\x1b[93m⚡ Executing: {name}\x1b[0m", flush=True)
        try:
            args_raw = tool_call["function"].get("arguments", "")
            args_obj = json.loads(args_raw) if args_raw else {}
            preview = json.dumps(args_obj, ensure_ascii=False)
            if len(preview) > 200:
                preview = preview[:200] + "…"
            print(f"\x1b[90m   args: {preview}\x1b[0m", flush=True)
        except (json.JSONDecodeError, TypeError):
            pass

    async def on_tool_end(self, tool_call: Any, result: Any) -> None:
        name = tool_call["function"]["name"]
        if isinstance(result, Exception):
            print(f"\x1b[91m✗ {name} failed: {result}\x1b[0m", flush=True)
        else:
            result_str = str(result)
            if len(result_str) > 300:
                result_str = result_str[:300] + "…"
            print(f"\x1b[92m✓ {name} → {result_str}\x1b[0m", flush=True)

    async def on_execution_start(self, tool_calls: Any) -> None:
        self._end_thinking_block()
        n = len(tool_calls)
        label = "tool" if n == 1 else "tools"
        print(f"\n\x1b[95m[Running {n} {label}…]\x1b[0m", flush=True)

    def finish(self) -> None:
        self._end_thinking_block()
        if self.response_started:
            print(flush=True)


class SimpleCli:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "qwen3.5-27b",
        system_prompt: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.system_prompt = system_prompt

    def run(self) -> None:
        try:
            asyncio.run(self._run_async())
        except KeyboardInterrupt:
            print("\nGoodbye!")
            sys.exit(0)

    async def _run_async(self) -> None:
        async def wait_for_user_input(first_result: str) -> str:
            print(f"\n\x1b[91m[User Input Required]\x1b[0m {first_result}")
            return input("\x1b[94mYour answer:\x1b[0m ").strip()

        client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key or None)
        llm_provider = OpenAILLMProvider(client=client, model=self.model)

        default_prompt = self.system_prompt or "You are a helpful assistant."
        default_prompt += (
            f"\n\nYou are working on a {platform.system()} machine"
            f" and your current working directory is `{Path.cwd()}`"
        )

        memory = ConversationMemory(system_prompt=default_prompt)
        tools = ToolRegistry.get_instance().get_all()
        agent = OpenAIAgent(
            llm_provider=llm_provider,
            tools=tools,
            memory=memory,
        )

        print("=" * 50)
        print("  simple-cli — minimal-harness chat")
        print("=" * 50)
        print(f"  Model : {self.model}")
        print(f"  URL   : {self.base_url}")
        print("=" * 50)
        print("  Type 'exit' or 'quit' to stop")
        print("=" * 50)
        print()

        while True:
            try:
                user_input = input("\x1b[94mYou:\x1b[0m ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            print()
            handler = SimpleStreamHandler()

            try:
                llm_provider._on_chunk = handler.on_chunk  # type: ignore[attr-defined]

                await agent.run(
                    user_input=cast(
                        list[ExtendedInputContentPart],
                        [{"type": "text", "text": user_input}],
                    ),
                    on_tool_start=handler.on_tool_start,
                    on_tool_end=handler.on_tool_end,
                    on_execution_start=handler.on_execution_start,
                    wait_for_user_input=wait_for_user_input,
                )
            except Exception as e:
                print(f"\n\x1b[91mError: {e}\x1b[0m")
                import traceback

                traceback.print_exc()

            handler.finish()
            print()

            usage = memory.get_total_usage()
            tokens = usage.get("total_tokens", 0)
            print(f"\x1b[90m[Tokens used: {tokens:,}]\x1b[0m")
            print()
