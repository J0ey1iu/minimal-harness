"""Streaming buffer for LLM output."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StreamBuffer:
    """Holds the current streaming LLM output."""

    content: str = ""
    reasoning: str = ""
    tool_calls: dict[int, dict[str, str]] = field(default_factory=dict)
    _flushed: bool = False

    def clear(self) -> None:
        self.content = ""
        self.reasoning = ""
        self.tool_calls.clear()
        self._flushed = False
