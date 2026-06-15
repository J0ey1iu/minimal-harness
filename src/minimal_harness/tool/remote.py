"""Remote tool abstractions.

Keeps the ``RemoteTool`` wrapper, the ``RemoteToolExecutor`` Protocol,
and the ``make_remote_tool`` factory in the SDK. The concrete
SSE-over-HTTP executors (``SSEToolExecutor``, ``ToolServiceExecutor``)
live in the :mod:`mh_service_kit` package as service-infrastructure
code, not framework primitives.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Protocol,
    runtime_checkable,
)

from minimal_harness.types import (
    RemoteToolBinding,
    ToolCall,
    ToolEvent,
    ToolResult,
)

if TYPE_CHECKING:
    import httpx
else:
    try:
        import httpx
    except ImportError:  # pragma: no cover
        httpx = None  # type: ignore[assignment]


def _unwrap_tool_result(data: Any) -> Any:
    if (
        isinstance(data, dict)
        and "content" in data
        and ("__meta" in data or "__stop" in data)
    ):
        return ToolResult(
            content=data["content"],
            meta=data.get("__meta"),
            stop=data.get("__stop", False),
        )
    return data


@runtime_checkable
class RemoteToolExecutor(Protocol):
    """Protocol for executing a tool remotely. [...]

    Users implement this protocol to bridge framework-internal
    ``Tool.execute()`` calls to an external service over any
    transport protocol (SSE, gRPC, message queue, …).
    """

    def execute(
        self,
        args: dict[str, Any],
        tool_call: ToolCall,
        stop_event: Any,
    ) -> AsyncIterator[ToolEvent]: ...


class RemoteTool:
    """A ``Tool`` that delegates execution to a ``RemoteToolExecutor``.

    Usage::

        tool = RemoteTool(
            name="weather",
            description="Get weather for a city",
            parameters={...},
            executor=SSEToolExecutor(binding),
        )
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict,
        executor: RemoteToolExecutor,
        display_name: str | None = None,
        display_name_locale: dict[str, str] | None = None,
        description_locale: dict[str, str] | None = None,
        endpoint_url: str = "",
    ) -> None:
        self.name = name
        self.display_name = display_name or name
        self.description = description
        self.parameters = parameters
        self._executor = executor
        self._endpoint_url = endpoint_url
        self.display_name_locale = display_name_locale
        self.description_locale = description_locale

    @property
    def endpoint_url(self) -> str:
        return self._endpoint_url

    def resolve_display_name(self, locale: str = "") -> str:
        if locale and self.display_name_locale and locale in self.display_name_locale:
            return self.display_name_locale[locale]
        return self.display_name

    def resolve_description(self, locale: str = "") -> str:
        if locale and self.description_locale and locale in self.description_locale:
            return self.description_locale[locale]
        return self.description

    def to_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    async def execute(
        self,
        args: dict[str, Any],
        tool_call: ToolCall,
        stop_event: Any,
    ) -> AsyncIterator[ToolEvent]:
        async for event in self._executor.execute(args, tool_call, stop_event):
            yield event


def make_remote_tool(schema: dict) -> RemoteTool:
    """Create a ``RemoteTool`` from a tool schema with ``endpoint_url``.

    The schema can be either the outer OpenAI format::

        {"type": "function", "function": {"name": "...", ...}, "endpoint_url": "http://..."}

    or the inner function dict directly.
    """
    from mh_service_kit.sse.tool_executor import SSEToolExecutor

    func = schema if "function" not in schema else schema["function"]
    endpoint_url = schema.get("endpoint_url") or func.get("endpoint_url", "")

    if not endpoint_url:
        raise ValueError(
            f"Tool '{func.get('name', '?')}' requires endpoint_url. "
            f"Omit the tool from the schema instead of sending it without a URL."
        )

    executor = SSEToolExecutor(RemoteToolBinding(url=endpoint_url))
    return RemoteTool(
        name=func["name"],
        description=func.get("description", ""),
        parameters=func.get("parameters", {}),
        executor=executor,
        endpoint_url=endpoint_url,
    )
