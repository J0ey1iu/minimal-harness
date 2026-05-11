from __future__ import annotations

from typing import Awaitable, Callable, Generic, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class RegistryProtocol(Generic[T], Protocol):
    """Generic registry interface for managing named items of type T."""

    async def register(self, name: str, item: T) -> None: ...

    async def unregister(self, name: str) -> bool: ...

    async def get(self, name: str) -> T | None: ...

    async def get_all(self, exclude: str | None = None) -> list[T]: ...

    async def names(self) -> list[str]: ...

    async def clear(self) -> None: ...

    async def add_listener(self, listener: Callable[[], Awaitable[None]]) -> None: ...

    async def remove_listener(
        self, listener: Callable[[], Awaitable[None]]
    ) -> None: ...


class Registry(Generic[T]):
    def __init__(self) -> None:
        self._data: dict[str, T] = {}
        self._listeners: list[Callable[[], Awaitable[None]]] = []

    async def _register(self, name: str, item: T) -> None:
        self._data[name] = item
        await self._notify()

    async def unregister(self, name: str) -> bool:
        if name in self._data:
            del self._data[name]
            await self._notify()
            return True
        return False

    async def get(self, name: str) -> T | None:
        return self._data.get(name)

    async def get_all(self, exclude: str | None = None) -> list[T]:
        if exclude is None:
            return list(self._data.values())
        return [v for k, v in self._data.items() if k != exclude]

    async def names(self) -> list[str]:
        return list(self._data.keys())

    async def clear(self) -> None:
        self._data.clear()
        await self._notify()

    async def add_listener(self, listener: Callable[[], Awaitable[None]]) -> None:
        self._listeners.append(listener)

    async def remove_listener(self, listener: Callable[[], Awaitable[None]]) -> None:
        self._listeners.remove(listener)

    async def _notify(self) -> None:
        for listener in self._listeners:
            await listener()
