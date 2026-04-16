from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from minimal_harness.tool.base import Tool


class ToolRegistry:
    _instance: "ToolRegistry | None" = None

    def __init__(self) -> None:
        self._tools: dict[str, "Tool"] = {}
        self._listeners: list[Callable[[], None]] = []

    @classmethod
    def get_instance(cls) -> "ToolRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, tool: "Tool") -> None:
        self._tools[tool.name] = tool
        self._notify()

    def unregister(self, name: str) -> bool:
        if name in self._tools:
            del self._tools[name]
            self._notify()
            return True
        return False

    def get(self, name: str) -> "Tool | None":
        return self._tools.get(name)

    def get_all(self) -> list["Tool"]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def clear(self) -> None:
        self._tools.clear()
        self._notify()

    def add_listener(self, listener: Callable[[], None]) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[], None]) -> None:
        self._listeners.remove(listener)

    def _notify(self) -> None:
        for listener in self._listeners:
            listener()
