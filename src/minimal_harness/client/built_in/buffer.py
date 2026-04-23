"""Streaming buffer for LLM output."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO

from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text


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
            out.append("▼ thinking\n", "dim italic #89b4fa")
            out.append(self.reasoning, "dim italic #89b4fa")
        if self.content:
            if self.reasoning:
                out.append("\n\n")
            if render_markdown:
                buf = StringIO()
                console = Console(file=buf, force_terminal=True, width=width)
                console.print(Markdown(self.content))
                out.append(Text.from_ansi(buf.getvalue()))
            else:
                out.append(self.content)
        return out

    def clear(self) -> None:
        self.content = ""
        self.reasoning = ""
        self.tool_calls.clear()
        self._flushed = False