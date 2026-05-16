from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Generic,
    Literal,
    Protocol,
    TypeVar,
    runtime_checkable,
)

T = TypeVar("T")


@dataclass
class RegistryChangeEvent:
    """Payload passed to registry listeners on every mutation."""

    action: Literal["register", "unregister", "clear"]
    name: str | None = None
    item: Any | None = None


@runtime_checkable
class RegistryProtocol(Generic[T], Protocol):
    """Generic registry interface for managing named items of type T."""

    async def register(self, name: str, item: T) -> None: ...

    async def unregister(self, name: str) -> bool: ...

    async def get(self, name: str) -> T | None: ...

    async def get_all(self, exclude: str | None = None) -> list[T]: ...

    async def names(self) -> list[str]: ...

    async def clear(self) -> None: ...

    async def add_listener(
        self, listener: Callable[[RegistryChangeEvent], Awaitable[None]]
    ) -> None: ...

    async def remove_listener(
        self, listener: Callable[[RegistryChangeEvent], Awaitable[None]]
    ) -> None: ...


class Registry(Generic[T]):
    def __init__(self) -> None:
        self._data: dict[str, T] = {}
        self._listeners: list[Callable[[RegistryChangeEvent], Awaitable[None]]] = []

    async def _register(self, name: str, item: T) -> None:
        self._data[name] = item
        await self._notify(RegistryChangeEvent("register", name, item))

    async def unregister(self, name: str) -> bool:
        if name in self._data:
            item = self._data.pop(name)
            await self._notify(RegistryChangeEvent("unregister", name, item))
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
        items = dict(self._data)
        self._data.clear()
        await self._notify(RegistryChangeEvent("clear", item=items))

    async def add_listener(
        self, listener: Callable[[RegistryChangeEvent], Awaitable[None]]
    ) -> None:
        self._listeners.append(listener)

    async def remove_listener(
        self, listener: Callable[[RegistryChangeEvent], Awaitable[None]]
    ) -> None:
        self._listeners.remove(listener)

    async def _notify(self, event: RegistryChangeEvent) -> None:
        for listener in self._listeners:
            await listener(event)
