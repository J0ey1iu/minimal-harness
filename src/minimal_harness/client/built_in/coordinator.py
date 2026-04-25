"""Application coordinator - owns core state and orchestrates components."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text

from minimal_harness.client.built_in.buffer import StreamBuffer
from minimal_harness.client.built_in.context import AppContext

if TYPE_CHECKING:
    from minimal_harness.client.built_in.memory import PersistentMemory


class AppCoordinator:
    def __init__(
        self,
        ctx: AppContext,
        buf: StreamBuffer,
        committed: list[Text],
    ) -> None:
        self._ctx = ctx
        self._buf = buf
        self._committed = committed
        self._first = True

    @property
    def ctx(self) -> AppContext:
        return self._ctx

    @property
    def buf(self) -> StreamBuffer:
        return self._buf

    @property
    def committed(self) -> list[Text]:
        return self._committed

    @property
    def first(self) -> bool:
        return self._first

    @first.setter
    def first(self, value: bool) -> None:
        self._first = value

    def clear_all(self) -> None:
        self._committed.clear()
        self._buf.clear()
        self._first = True

    def reset_for_new_chat(self) -> None:
        self.clear_all()
        self._ctx.reset_memory()
        self._ctx.rebuild()

    def reset_for_session(self, memory: "PersistentMemory") -> None:
        self.clear_all()
        self._ctx.memory = memory
        self._ctx.rebuild()