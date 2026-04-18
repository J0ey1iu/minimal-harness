"""SimpleCli — interactive chat loop backed by an OpenAI-compatible API."""

import asyncio
import os
import platform
import sys
import termios
import tty
from pathlib import Path
from typing import Any, cast

from openai import AsyncOpenAI
from prompt_toolkit import PromptSession
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

console = Console()

_original_term_settings: Any = None


def _enter_cbreak() -> None:
    global _original_term_settings
    fd = sys.stdin.fileno()
    _original_term_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)


def _leave_cbreak() -> None:
    global _original_term_settings
    if _original_term_settings is not None:
        fd = sys.stdin.fileno()
        termios.tcsetattr(fd, termios.TCSADRAIN, _original_term_settings)
        termios.tcflush(fd, termios.TCIFLUSH)
        _original_term_settings = None


def _esc_reader(stop_event: asyncio.Event) -> None:
    ch = os.read(sys.stdin.fileno(), 1)
    if ch == b"\x1b":
        stop_event.set()


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
        session = PromptSession()

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
        is_tty = platform.system() != "Windows" and sys.stdin.isatty()

        while True:
            user_input = await session.prompt_async("You: ")
            if user_input is None:
                break

            if not user_input.strip():
                continue

            lower_input = user_input.lower().strip()
            if lower_input in ("exit", "quit"):
                console.print("Goodbye!", style="bold")
                break

            console.print()
            handler = SimpleStreamHandler()
            stop_event = asyncio.Event()
            reader_active = False

            async def wait_for_user_input(first_result: str) -> str:
                nonlocal reader_active
                console.print(f"\n[User Input Required] {first_result}", style="red")
                if is_tty and reader_active:
                    loop.remove_reader(sys.stdin.fileno())
                    reader_active = False
                    _leave_cbreak()
                handler._pause_live()
                result = await session.prompt_async("Your answer: ")
                handler._resume_live()
                if is_tty:
                    _enter_cbreak()
                    loop.add_reader(sys.stdin.fileno(), _esc_reader, stop_event)
                    reader_active = True
                return result

            loop = asyncio.get_event_loop()

            if is_tty:
                _enter_cbreak()
                loop.add_reader(sys.stdin.fileno(), _esc_reader, stop_event)
                reader_active = True

            try:
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
                    on_chunk=handler.on_chunk,
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
                if reader_active:
                    try:
                        loop.remove_reader(sys.stdin.fileno())
                    except (ValueError, OSError):
                        pass
                    reader_active = False
                if is_tty:
                    _leave_cbreak()

            handler.finish()
            console.print()

            usage = memory.get_total_usage()
            tokens = usage.get("total_tokens", 0)
            console.print(Text(f"[Tokens used: {tokens:,}]", style="dim"))
            console.print()

    def _print_banner(self) -> None:
        info = (
            f"[bold]Model[/bold] : {self.model}\n"
            f"[bold]URL[/bold]   : {self.base_url}\n\n"
            "Type [bold]'exit'[/bold] or [bold]'quit'[/bold] to stop\n"
            "Press [bold]ESC[/bold] to stop generation"
        )
        console.print(
            Panel(
                info,
                title="simple-cli \u2014 minimal-harness chat",
                expand=False,
            )
        )
        console.print()
