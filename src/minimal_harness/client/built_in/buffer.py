"""Streaming buffer for LLM output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamBuffer:
    """Holds the current streaming LLM output."""

    content: str = ""
    reasoning: str = ""
    tool_calls: dict[int, dict[str, str]] = field(default_factory=dict)
    _flushed: bool = False

    def add_chunk(self, delta: Any) -> None:
        if delta is None:
            return
        if delta.reasoning:
            self.reasoning += delta.reasoning
        if delta.content:
            self.content += delta.content
        if delta.tool_calls:
            for tc in delta.tool_calls:
                call = self.tool_calls.setdefault(
                    tc.index, {"id": "", "name": "", "arguments": ""}
                )
                if tc.id:
                    call["id"] += tc.id
                if tc.name:
                    call["name"] += tc.name
                if tc.arguments:
                    call["arguments"] += tc.arguments

    def clear(self) -> None:
        self.content = ""
        self.reasoning = ""
        self.tool_calls.clear()
        self._flushed = False
