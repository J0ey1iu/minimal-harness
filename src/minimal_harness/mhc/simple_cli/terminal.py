"""Terminal utilities for cbreak mode and ESC key monitoring."""

import asyncio
import platform
import sys
import threading
from typing import Any


class CbreakMode:
    """Context manager that switches the terminal into cbreak mode.

    In cbreak mode individual key-presses can be read without waiting for
    a newline.  On Windows (or when stdin is not a TTY) the context manager
    is a no-op.
    """

    def __init__(self) -> None:
        self._original_settings: Any = None
        self._fd: int | None = None
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def __enter__(self) -> "CbreakMode":
        if platform.system() == "Windows" or not sys.stdin.isatty():
            return self
        import termios
        import tty

        fd = sys.stdin.fileno()
        self._fd = fd
        self._original_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        self._active = True
        return self

    def __exit__(self, *_: object) -> None:
        self._restore()

    def _restore(self) -> None:
        if (
            self._active
            and self._fd is not None
            and self._original_settings is not None
        ):
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._original_settings)
            self._active = False

    def _set_cbreak(self) -> None:
        if self._fd is not None:
            import tty

            tty.setcbreak(self._fd)
            self._active = True

    def canonical_input(self, prompt: str) -> str:
        """Temporarily restore canonical mode to read a full line of input."""
        if not self._active or platform.system() == "Windows":
            return input(prompt)

        import termios

        assert self._fd is not None and self._original_settings is not None
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._original_settings)
        self._active = False

        try:
            result = input(prompt).strip()
        finally:
            self._set_cbreak()

        return result


def monitor_esc_key(stop_event: asyncio.Event, pause_event: threading.Event) -> None:
    """Block until ESC is pressed or *stop_event* is set.

    While *pause_event* is set the monitor sleeps without reading input,
    allowing canonical-mode prompts to work uninterrupted.
    """
    import time

    if platform.system() == "Windows":
        import msvcrt

        while not stop_event.is_set():
            if pause_event.is_set():
                time.sleep(0.05)
                continue
            if msvcrt.kbhit():  # type: ignore[attr-defined]
                ch = msvcrt.getch()  # type: ignore[attr-defined]
                if ch == b"\x1b":
                    stop_event.set()
                    return
            time.sleep(0.05)
    else:
        import select

        while not stop_event.is_set():
            if pause_event.is_set():
                time.sleep(0.05)
                continue
            if select.select([sys.stdin], [], [], 0.05)[0]:
                ch = sys.stdin.read(1)
                if ch == "\x1b":
                    stop_event.set()
                    return
