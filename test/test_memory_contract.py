"""Memory contract check: duck-typed implementors must fail fast at run start.

Regression guard for mh-incubator #58 — a memory missing a protocol
member used to crash with AttributeError deep inside the streaming
generator on an edge path. The check in ``verify_memory_contract``
(and the one in ``Agent.run``) must name the missing members loudly.
"""

import pytest

from minimal_harness.memory import ConversationMemory, verify_memory_contract


def test_complete_memory_passes() -> None:
    verify_memory_contract(ConversationMemory())  # must not raise


def test_partial_memory_raises_naming_missing_member() -> None:
    class PartialMemory:
        """Deliberately missing the whole replay/persistence surface."""

        async def add_message(self, message: object) -> None: ...

    with pytest.raises(TypeError) as exc:
        verify_memory_contract(PartialMemory())
    assert "get_replay_messages" in str(exc.value)
    assert "PartialMemory" in str(exc.value)
