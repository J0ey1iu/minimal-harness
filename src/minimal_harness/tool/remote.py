from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, Protocol, runtime_checkable

from minimal_harness.types import (
    ExtraHeadersProvider,
    RemoteToolBinding,
    ToolCall,
    ToolEnd,
    ToolEvent,
    ToolProgress,
    ToolResult,
    ToolStart,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import httpx
else:
    try:
        import httpx
    except ImportError:  # pragma: no cover
        httpx = None  # type: ignore[assignment]


def _unwrap_tool_result(data: Any) -> Any:
    if isinstance(data, dict) and "content" in data and "__meta" in data:
        return ToolResult(content=data["content"], meta=data["__meta"])
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
        stop_event: asyncio.Event | None,
    ) -> AsyncIterator[ToolEvent]: ...


class SSEToolExecutor:
    """Default remote-tool executor based on SSE over HTTP.

    API contract (framework-defined)::

        POST <url>
        Request:  { "args": {...}, "tool_call": {...} }

        Response (SSE stream)::

          data: {"type": "tool_progress", "data": <chunk_data>}
          data: {"type": "tool_end",    "data": <result_data>}
          data: {"type": "error",       "data": {"message": "..."}}

    The ``type`` field is framework metadata (event discriminator).
    The ``data`` field is the tool implementer's actual output, passed
    through transparently as ``ToolEnd.result`` / ``ToolProgress.chunk``.
    """

    def __init__(self, binding: RemoteToolBinding) -> None:
        if httpx is None:
            raise ImportError(
                "httpx is required for SSEToolExecutor. "
                "Install it via `pip install httpx`."
            )
        self._url = binding.url
        self._headers = dict(binding.headers)
        self._extra_headers_provider: ExtraHeadersProvider | None = (
            binding.extra_headers_provider
        )
        self._timeout = binding.timeout

    async def _resolve_headers(self) -> dict[str, str]:
        headers = dict(self._headers)
        if self._extra_headers_provider is not None:
            extra = await self._extra_headers_provider()
            headers.update(extra)
        return headers

    async def execute(
        self,
        args: dict[str, Any],
        tool_call: ToolCall,
        stop_event: asyncio.Event | None,
    ) -> AsyncIterator[ToolEvent]:
        yield ToolStart(tool_call)
        tool_name = tool_call.get("function", {}).get("name", "?")
        logger.info(
            "tool.start name=%s url=%s args=%s",
            tool_name,
            self._url,
            list(args.keys()),
        )

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream(
                    "POST",
                    self._url,
                    json={"args": args, "tool_call": tool_call},
                    headers=await self._resolve_headers(),
                ) as resp:
                    logger.debug(
                        "tool.response url=%s status=%d",
                        self._url,
                        resp.status_code,
                    )
                    resp.raise_for_status()
                    final_result: Any = None
                    async for line in resp.aiter_lines():
                        if stop_event and stop_event.is_set():
                            break
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        payload = json.loads(line[6:])
                        event_type = payload.get("type")
                        tool_data = payload.get("data")

                        if event_type is None:
                            logger.debug(
                                "tool.sse.missing_type url=%s payload=%s",
                                self._url,
                                payload,
                            )
                            continue

                        if "data" not in payload:
                            if event_type == "error":
                                error_msg = payload.get("message", str(payload))
                                yield ToolEnd(tool_call, Exception(error_msg))
                                return
                            logger.debug(
                                "tool.sse.missing_data url=%s type=%s payload=%s",
                                self._url,
                                event_type,
                                payload,
                            )
                            continue

                        if event_type == "error":
                            error_msg = (
                                tool_data.get("message", str(tool_data))
                                if isinstance(tool_data, dict)
                                else str(tool_data)
                            )
                            yield ToolEnd(tool_call, Exception(error_msg))
                            return

                        if event_type == "tool_end":
                            final_result = _unwrap_tool_result(tool_data)
                        elif event_type == "tool_progress":
                            yield ToolProgress(tool_call, tool_data)

                    if final_result is not None:
                        yield ToolEnd(tool_call, final_result)
                    else:
                        yield ToolEnd(tool_call, "ok")

        except httpx.HTTPStatusError as e:
            logger.warning(
                "tool.http.error name=%s url=%s status=%d",
                tool_name,
                self._url,
                e.response.status_code,
            )
            yield ToolEnd(
                tool_call,
                Exception(f"Remote tool HTTP error: {e.response.status_code}"),
            )
        except httpx.RequestError as e:
            logger.warning(
                "tool.request.error name=%s url=%s error=%s",
                tool_name,
                self._url,
                e,
            )
            yield ToolEnd(
                tool_call,
                Exception(f"Remote tool request failed: {e}"),
            )
        except Exception as e:
            logger.exception(
                "tool.unexpected.error name=%s url=%s",
                tool_name,
                self._url,
            )
            yield ToolEnd(
                tool_call,
                Exception(f"Unexpected tool error: {e}"),
            )


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
        stop_event: asyncio.Event | None,
    ) -> AsyncIterator[ToolEvent]:
        async for event in self._executor.execute(args, tool_call, stop_event):
            yield event


class ToolServiceExecutor:
    """RemoteToolExecutor that routes all tools to a shared tool service.

    The service is expected to expose ``POST /tools/{tool_name}/execute``
    endpoints and stream back SSE lines::

        data: {"type": "tool_progress", "data": <chunk_data>}
        data: {"type": "tool_end",    "data": <result_data>}

    The ``data`` field is the tool implementer's actual output, passed
    through transparently as ``ToolEnd.result`` / ``ToolProgress.chunk``.
    """

    def __init__(self, service_url: str, timeout: int = 30) -> None:
        if httpx is None:
            raise ImportError(
                "httpx is required for ToolServiceExecutor. "
                "Install it via `pip install httpx`."
            )
        self._service_url = service_url.rstrip("/")
        self._timeout = timeout

    async def execute(
        self,
        args: dict[str, Any],
        tool_call: ToolCall,
        stop_event: asyncio.Event | None,
    ) -> AsyncIterator[ToolEvent]:
        yield ToolStart(tool_call)
        tool_name = tool_call["function"]["name"]
        logger.info(
            "tool.start name=%s service=%s args=%s",
            tool_name,
            self._service_url,
            list(args.keys()),
        )
        result: Any = None
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, trust_env=False
            ) as client:
                async with client.stream(
                    "POST",
                    f"{self._service_url}/tools/{tool_name}/execute",
                    json={"args": args, "tool_call_id": tool_call.get("id", "")},
                ) as resp:
                    logger.debug(
                        "tool.response service=%s/tools/%s status=%d",
                        self._service_url,
                        tool_name,
                        resp.status_code,
                    )
                    async for line in resp.aiter_lines():
                        if stop_event and stop_event.is_set():
                            break
                        if not line.startswith("data: "):
                            continue
                        ev = json.loads(line[6:])
                        event_type = ev.get("type")
                        tool_data = ev.get("data")

                        if event_type is None:
                            logger.debug(
                                "tool.sse.missing_type service=%s/tools/%s payload=%s",
                                self._service_url,
                                tool_name,
                                ev,
                            )
                            continue

                        if "data" not in ev:
                            logger.debug(
                                "tool.sse.missing_data service=%s/tools/%s type=%s payload=%s",
                                self._service_url,
                                tool_name,
                                event_type,
                                ev,
                            )
                            continue

                        if event_type == "tool_progress":
                            yield ToolProgress(tool_call, tool_data)
                        elif event_type == "tool_end":
                            result = _unwrap_tool_result(tool_data)
        except Exception as e:
            logger.warning(
                "tool.service.error name=%s service=%s error=%s",
                tool_name,
                self._service_url,
                e,
            )
            result = f"Tool execution error: {e}"
        yield ToolEnd(tool_call, result)


def make_remote_tool(schema: dict) -> RemoteTool:
    """Create a ``RemoteTool`` from a tool schema with ``endpoint_url``.

    The schema can be either the outer OpenAI format::

        {"type": "function", "function": {"name": "...", ...}, "endpoint_url": "http://..."}

    or the inner function dict directly.
    """
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
