"""Global error handler for the TUI — captures unhandled exceptions and displays them."""

from __future__ import annotations

import asyncio
import sys
import traceback as tb
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class CapturedError:
    timestamp: str
    formatted: str
    brief: str
    source: str = ""

    @classmethod
    def from_exc_info(
        cls,
        exc_type: type[BaseException],
        exc_val: BaseException,
        tb_obj,
        source: str = "",
    ) -> CapturedError:
        formatted = "".join(tb.format_exception(exc_type, exc_val, tb_obj))
        brief = f"{exc_type.__name__}: {exc_val}"
        return cls(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            formatted=formatted,
            brief=brief,
            source=source,
        )


class ErrorHandler:
    """Singleton that captures unhandled exceptions and provides them to the TUI."""

    _instance: ErrorHandler | None = None

    def __init__(self) -> None:
        self._errors: list[CapturedError] = []
        self._listeners: list[Callable[[CapturedError], None]] = []
        self._enabled: bool = False

    def __new__(cls) -> ErrorHandler:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def errors(self) -> list[CapturedError]:
        return list(self._errors)

    @property
    def latest(self) -> CapturedError | None:
        return self._errors[-1] if self._errors else None

    def add_listener(self, listener: Callable[[CapturedError], None]) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[CapturedError], None]) -> None:
        try:
            self._listeners.remove(listener)
        except ValueError:
            pass

    def capture(self, error: CapturedError) -> None:
        self._errors.append(error)
        for listener in self._listeners:
            try:
                listener(error)
            except Exception:
                pass

    def clear(self) -> None:
        self._errors.clear()

    def install(self) -> None:
        if self._enabled:
            return
        self._enabled = True

        handler = self

        def _hook(
            exc_type: type[BaseException], exc_val: BaseException, _tb_obj
        ) -> None:
            err = CapturedError.from_exc_info(
                exc_type, exc_val, _tb_obj, source="global"
            )
            handler.capture(err)
            sys.__excepthook__(exc_type, exc_val, _tb_obj)

        sys.excepthook = _hook

        def _asyncio_handler(loop: object, context: dict[str, object]) -> None:
            exc = context.get("exception")
            if exc is not None and isinstance(exc, Exception):
                err = CapturedError.from_exc_info(
                    type(exc), exc, exc.__traceback__, source="asyncio"
                )
                handler.capture(err)

        loop = asyncio.get_event_loop()
        loop.set_exception_handler(_asyncio_handler)

    def uninstall(self) -> None:
        sys.excepthook = sys.__excepthook__
        self._enabled = False
