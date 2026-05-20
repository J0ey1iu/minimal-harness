from __future__ import annotations

from typing import Any, AsyncIterator, Protocol, runtime_checkable


@runtime_checkable
class RegistryProvider(Protocol):
    """Registry metadata provider (agents + tools + scenarios).

    Customer deployment: implement this protocol to query your own
    registry system instead of the built-in registry-service.
    """

    async def get_agent(self, name: str) -> dict[str, Any] | None: ...
    async def list_agents(self) -> list[dict[str, Any]]: ...
    async def get_tool(self, name: str) -> dict[str, Any] | None: ...
    async def list_tools(self) -> list[dict[str, Any]]: ...
    async def get_scenario(self, scenario_id: str) -> dict[str, Any] | None: ...
    async def list_scenarios(self) -> list[dict]: ...


@runtime_checkable
class ToolProvider(Protocol):
    """Provides tool definitions and execution.

    Customer deployment: implement this protocol to register the
    customer's own tools (e.g. loaded from a database, YAML files,
    or a remote tool registry).
    """

    def list_tools(self) -> list[dict]:
        """Return metadata for all available tools.

        Each dict should contain ``name``, ``display_name``,
        ``description``, ``parameters``, and optionally
        ``display_name_locale`` / ``description_locale``.
        """
        ...

    def execute(
        self, tool_name: str, args: dict, tool_call_id: str
    ) -> AsyncIterator[str]:
        """Execute *tool_name* with *args* and yield SSE lines.

        Each yielded line should follow the format::

            data: {"type": "tool_progress", "content": "..."}
            data: {"type": "tool_end", "result": "..."}
        """
        ...
