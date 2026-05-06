"""Export history tracking for chat content."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExportEntry:
    text: str
    style: str | None = None
    is_markdown: bool = False


class ExportTracker:
    def __init__(self) -> None:
        self._history: list[ExportEntry] = []

    @property
    def history(self) -> list[ExportEntry]:
        return self._history

    def add(self, entry: ExportEntry) -> None:
        self._history.append(entry)

    def clear(self) -> None:
        self._history.clear()
