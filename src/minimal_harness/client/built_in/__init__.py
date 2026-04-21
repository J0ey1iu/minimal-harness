from __future__ import annotations


def __getattr__(name: str):
    if name == "TUIApp":
        from .tui import TUIApp
        return TUIApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")