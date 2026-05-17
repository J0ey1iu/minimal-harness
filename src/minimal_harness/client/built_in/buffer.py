"""Streaming buffer for LLM output."""

from __future__ import annotations

from typing import Any


class StreamBuffer:
    """Holds the current streaming LLM output.

    Uses list-based string accumulation internally for efficient
    concatenation of many small streaming chunks.
    """

    def __init__(self) -> None:
        self._content_parts: list[str] = []
        self._reasoning_parts: list[str] = []
        self.tool_calls: dict[int, dict[str, str]] = {}
        self._flushed: bool = False

    @property
    def content(self) -> str:
        return "".join(self._content_parts)

    @content.setter
    def content(self, value: str) -> None:
        self._content_parts = [value] if value else []

    @property
    def reasoning(self) -> str:
        return "".join(self._reasoning_parts)

    @reasoning.setter
    def reasoning(self, value: str) -> None:
        self._reasoning_parts = [value] if value else []

    def add_chunk(self, delta: Any) -> None:
        if delta is None:
            return
        if delta.reasoning:
            self._reasoning_parts.append(delta.reasoning)
        if delta.content:
            self._content_parts.append(delta.content)
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

    @property
    def flushed(self) -> bool:
        return self._flushed

    def mark_flushed(self) -> None:
        self._flushed = True

    def clear(self) -> None:
        self._content_parts.clear()
        self._reasoning_parts.clear()
        self.tool_calls.clear()
        self._flushed = False
