from __future__ import annotations

from typing import Any, AsyncIterator, Protocol, runtime_checkable


@runtime_checkable
class AgentMetadataProvider(Protocol):
    """Provides agent metadata from an external registry (built-in or customer).

    Customer deployment: implement this protocol to query the customer's
    own Agent Registry system instead of the built-in registry-service.
    """

    async def get_agent(self, name: str) -> dict[str, Any] | None:
        """Return agent metadata dict or None if not found.

        Expected keys: ``name``, ``display_name``, ``description``,
        ``endpoint_url``, ``system_prompt``, ``display_name_locale``,
        ``description_locale``.
        """
        ...

    async def list_agents(self) -> list[dict[str, Any]]:
        """Return all registered agent metadata dicts."""
        ...


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


@runtime_checkable
class ScenarioProvider(Protocol):
    """Provides scenario configurations (the "agent+tool presets").

    Customer deployment: implement this protocol to load scenarios
    from an external configuration system instead of the local SQLite store.
    """

    async def list_scenarios(self) -> list[dict]:
        """Return all scenario config dicts.

        Each dict should contain ``id``, ``name``, ``agents`` (list of
        ``{name, tool_names}``), and optionally ``icon``, ``name_locale``,
        ``description``, ``description_locale``.
        """
        ...


@runtime_checkable
class SecretResolver(Protocol):
    """Resolves secrets (API keys, tokens, etc.) at runtime.

    Customer deployment: implement this protocol to load secrets from
    a vault system (HashiCorp Vault, AWS Secrets Manager, etc.) instead
    of environment variables.
    """

    async def get(self, key: str) -> str | None:
        """Return the secret value for *key*, or None if not found."""
        ...


@runtime_checkable
class ToolMetadataProvider(Protocol):
    """Provides tool metadata from an external registry (built-in or customer).

    Customer deployment: implement this protocol to query the customer's
    own Tool Registry system instead of the built-in registry-service.
    """

    async def get_tool(self, name: str) -> dict[str, Any] | None:
        """Return tool metadata dict or None if not found.

        Expected keys: ``name``, ``display_name``, ``description``,
        ``endpoint_url``, ``parameters_schema``, ``display_name_locale``,
        ``description_locale``.
        """
        ...

    async def list_tools(self) -> list[dict[str, Any]]:
        """Return all registered tool metadata dicts."""
        ...
