"""Simple agent loop — deprecated alias for :class:`BaseAgent`.

``SimpleAgent`` was a zero-code subclass of ``BaseAgent`` with an identical
constructor. It is now deprecated — use ``BaseAgent`` directly.
"""

from __future__ import annotations

import warnings
from typing import Sequence

from minimal_harness.llm.llm import LLMProvider

from .base import BaseAgent
from .middleware import Middleware
from .protocol import InputContentConversionFunction


class SimpleAgent(BaseAgent):
    """Deprecated alias for :class:`BaseAgent`.

    .. deprecated:: 0.8.0
        Use :class:`BaseAgent` directly. This subclass will be removed in a
        future version.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        max_iterations: int = 100,
        custom_input_conversion: InputContentConversionFunction | None = None,
        middleware: Sequence[Middleware] = (),
        emit_message_events: bool = True,
    ):
        warnings.warn(
            "SimpleAgent is deprecated, use BaseAgent directly. "
            "SimpleAgent will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            llm_provider=llm_provider,
            max_iterations=max_iterations,
            custom_input_conversion=custom_input_conversion,
            middleware=middleware,
            emit_message_events=emit_message_events,
        )


__all__ = ["SimpleAgent"]
