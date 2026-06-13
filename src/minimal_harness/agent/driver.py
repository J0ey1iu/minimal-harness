from __future__ import annotations

from typing import Protocol, runtime_checkable

from minimal_harness.types import RemoteAgentBinding


@runtime_checkable
class RemoteAgentDriver(Protocol):
    """Protocol for executing an agent remotely.

    Users implement this protocol to bridge framework-internal
    ``Agent.run()`` calls to an external agent service over any
    transport protocol. The default SSE-over-HTTP implementation
    lives in :mod:`mh_service_kit.sse`.
    """

    async def run(self, *args: object, **kwargs: object) -> object: ...


@runtime_checkable
class RemoteAgentDriverFactory(Protocol):
    """Factory that creates a ``RemoteAgentDriver`` from a binding."""

    def create(self, binding: RemoteAgentBinding) -> RemoteAgentDriver: ...


__all__ = [
    "RemoteAgentDriver",
    "RemoteAgentDriverFactory",
]
