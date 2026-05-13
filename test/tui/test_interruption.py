from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from minimal_harness.agent.registry import AgentRegistry
from minimal_harness.client.built_in.buffer import StreamBuffer
from minimal_harness.client.built_in.context import AppContext
from minimal_harness.client.built_in.session_controller import SessionController
from minimal_harness.types import LLMChunkDelta

# -- SessionController interruption ------------------------------------------


class TestSessionControllerInterruption:
    def test_interrupt_calls_stop_on_current_session(self):
        ctx = MagicMock(spec=AppContext)
        ctrl = SessionController(MagicMock(), AgentRegistry(), ctx)
        session = MagicMock()
        ctrl._sessions["s1"] = session
        ctrl._current_session_id = "s1"

        ctrl.interrupt()
        session.interrupt.assert_called_once()

    def test_interrupt_sets_stop_event_for_active_run(self):
        ctx = MagicMock(spec=AppContext)
        ctrl = SessionController(MagicMock(), AgentRegistry(), ctx)
        ctrl._sessions["s1"] = MagicMock()
        ctrl._current_session_id = "s1"

        stop_event = asyncio.Event()
        ctrl._active_runs["s1"] = (MagicMock(), stop_event, MagicMock())

        ctrl.interrupt()
        assert stop_event.is_set()


# -- Buffer state after stop (no textual app needed) ---------------------------


def test_buffer_clear_after_streaming():
    buf = StreamBuffer()
    buf.add_chunk(LLMChunkDelta(content="partial"))
    assert buf.content == "partial"
    buf.clear()
    assert buf.content == ""
    assert buf._flushed is False
