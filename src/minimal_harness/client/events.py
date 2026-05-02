"""Backward-compat re-exports - use types.py directly.

All event types previously defined here have been consolidated into
minimal_harness.types to eliminate the parallel event hierarchy.
"""

from minimal_harness.types import (
    AgentEvent as Event,
)


def to_client_event(event: Event) -> Event:
    return event
