"""Streaming buffer for LLM output."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO

from rich.console import Console
from rich.text import Text

from minimal_harness.client.built_in.markdown_styles import BorderedMarkdown


@dataclass
class StreamBuffer:
    """Holds the current streaming LLM output."""

    content: str = ""
    reasoning: str = ""
    tool_calls: dict[int, dict[str, str]] = field(default_factory=dict)
    _flushed: bool = False

    def render(self, render_markdown: bool = True, width: int = 80) -> Text:
        out = Text()
        if self.reasoning:
            out.append("╭─ thinking\n", "dim bright_blue")
            out.append(self.reasoning, "dim bright_blue")
            out.append("\n╰─", "dim bright_blue")
        if self.content:
            if self.reasoning:
                out.append("\n\n")
            if render_markdown:
                with StringIO() as buf:
                    with Console(
                        file=buf, force_terminal=True, width=width
                    ) as console:
                        console.print(BorderedMarkdown(self.content))
                    out.append(Text.from_ansi(buf.getvalue()))
            else:
                out.append(self.content)
        return out

    def clear(self) -> None:
        self.content = ""
        self.reasoning = ""
        self.tool_calls.clear()
        self._flushed = False
