from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self) -> None:
        self._data: dict[str, T] = {}
        self._listeners: list[Callable[[], None]] = []

    def _register(self, name: str, item: T) -> None:
        self._data[name] = item
        self._notify()

    def unregister(self, name: str) -> bool:
        if name in self._data:
            del self._data[name]
            self._notify()
            return True
        return False

    def get(self, name: str) -> T | None:
        return self._data.get(name)

    def get_all(self) -> list[T]:
        return list(self._data.values())

    def names(self) -> list[str]:
        return list(self._data.keys())

    def clear(self) -> None:
        self._data.clear()
        self._notify()

    def add_listener(self, listener: Callable[[], None]) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[], None]) -> None:
        self._listeners.remove(listener)

    def _notify(self) -> None:
        for listener in self._listeners:
            listener()
