"""SimpleCli — interactive chat loop backed by an OpenAI-compatible API."""

import asyncio
import platform
import sys
import threading
from pathlib import Path
from typing import cast

from openai import AsyncOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from minimal_harness.agent import OpenAIAgent
from minimal_harness.llm.openai import OpenAILLMProvider
from minimal_harness.memory import (
    ConversationMemory,
    ExtendedInputContentPart,
)
from minimal_harness.tool.registry import ToolRegistry

from .stream_handler import SimpleStreamHandler
from .terminal import CbreakMode, monitor_esc_key

console = Console()


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
            console.print("\nGoodbye!", style="bold")
            sys.exit(0)

    async def _run_async(self) -> None:
        cbreak = CbreakMode()
        esc_pause = threading.Event()

        async def wait_for_user_input(first_result: str) -> str:
            console.print(f"\n[User Input Required] {first_result}", style="red")
            esc_pause.set()
            result = cbreak.canonical_input("Your answer: ")
            esc_pause.clear()
            return result

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

        self._print_banner()

        while True:
            user_input = self._read_user_input()
            if user_input is None:
                break

            console.print()
            handler = SimpleStreamHandler()
            stop_event = asyncio.Event()

            with cbreak:
                esc_thread = threading.Thread(
                    target=monitor_esc_key,
                    args=(stop_event, esc_pause),
                    daemon=True,
                )
                esc_thread.start()

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
                        on_tool_progress=handler.on_tool_progress,
                        stop_event=stop_event,
                    )

                    if stop_event.is_set():
                        console.print(Text("\n[Stopped by user]", style="red"))
                except Exception as e:
                    handler._stop_live()
                    console.print(Text(f"\nError: {e}", style="bold red"))
                    console.print_exception()
                finally:
                    stop_event.set()
                    esc_thread.join(timeout=0.1)

            handler.finish()
            console.print()

            usage = memory.get_total_usage()
            tokens = usage.get("total_tokens", 0)
            console.print(Text(f"[Tokens used: {tokens:,}]", style="dim"))
            console.print()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _print_banner(self) -> None:
        info = (
            f"[bold]Model[/bold] : {self.model}\n"
            f"[bold]URL[/bold]   : {self.base_url}\n\n"
            "Type [bold]'exit'[/bold] or [bold]'quit'[/bold] to stop\n"
            "Press [bold]ESC[/bold] to stop generation"
        )
        console.print(
            Panel(info, title="simple-cli \u2014 minimal-harness chat", expand=False)
        )
        console.print()

    @staticmethod
    def _read_user_input() -> str | None:
        """Read a line from the user. Returns ``None`` on exit/EOF."""
        try:
            console.print(Text("You:", style="blue"), end=" ")
            user_input = input().strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye!", style="bold")
            return None

        if not user_input:
            return SimpleCli._read_user_input()

        if user_input.lower() in ("exit", "quit"):
            console.print("Goodbye!", style="bold")
            return None

        return user_input
