from __future__ import annotations

from typing import Any, AsyncIterator, Callable, Protocol, cast, runtime_checkable

from minimal_harness.tool.base import StreamingTool, Tool
from minimal_harness.tool.remote import (
    SSEToolExecutor,
    RemoteTool,
    RemoteToolExecutor,
)
from minimal_harness.types import (
    ExternalScriptToolBinding,
    LocalToolBinding,
    RemoteToolBinding,
    ToolMetadata,
)
from minimal_harness.tool.wrapper import ExternalToolWrapper


@runtime_checkable
class ToolExecutorFactory(Protocol):
    """Factory that creates a ``RemoteToolExecutor`` from a binding."""

    def create(self, binding: RemoteToolBinding) -> RemoteToolExecutor: ...


class DefaultToolExecutorFactory:
    """Default factory: returns ``SSEToolExecutor`` for any remote binding."""

    def create(self, binding: RemoteToolBinding) -> RemoteToolExecutor:
        return cast(RemoteToolExecutor, SSEToolExecutor(binding))


class ToolFactory(Protocol):
    """Creates a concrete ``Tool`` from ``ToolMetadata``.

    Implement this protocol to customise how tool metadata is
    turned into an executable tool (e.g. wire up a custom
    ``RemoteToolExecutor``).
    """

    def create(self, metadata: ToolMetadata) -> Tool: ...


class DefaultToolFactory:
    """Default ``ToolFactory`` that handles all built-in binding types.

    Resolves local, external-script and remote bindings.  Users can
    register custom ``ToolExecutorFactory`` implementations for
    specific driver names via ``executor_factories``.
    """

    def __init__(
        self,
        executor_factories: dict[str, ToolExecutorFactory] | None = None,
    ) -> None:
        self._executor_factories: dict[str, ToolExecutorFactory] = {
            "default": DefaultToolExecutorFactory(),
            **(executor_factories or {}),
        }

    def register_executor_factory(
        self, driver: str, factory: ToolExecutorFactory
    ) -> None:
        self._executor_factories[driver] = factory

    def create(self, metadata: ToolMetadata) -> Tool:
        name = metadata.name
        description = metadata.description
        parameters = metadata.parameters
        display_name = metadata.display_name
        dn_locale = metadata.display_name_locale
        desc_locale = metadata.description_locale

        binding = metadata.binding
        if binding is None:
            raise ValueError(
                f"Cannot create a Tool from '{name}': no binding set on metadata. "
                "Provide a binding (LocalToolBinding, ExternalScriptToolBinding, "
                "or RemoteToolBinding) when registering the ToolMetadata."
            )

        match binding:
            case LocalToolBinding(fn=fn):
                if fn is None:
                    raise ValueError(
                        f"LocalToolBinding for '{name}' requires a 'fn' callable"
                    )
                return StreamingTool(
                    name=name,
                    description=description,
                    parameters=parameters,
                    fn=fn,
                    display_name=display_name,
                    display_name_locale=dn_locale,
                    description_locale=desc_locale,
                )

            case ExternalScriptToolBinding(script_path=uri):
                return StreamingTool(
                    name=name,
                    description=description,
                    parameters=parameters,
                    fn=_make_external_wrapper(
                        uri, name, description, parameters, dn_locale, desc_locale
                    ),
                    display_name=display_name,
                    display_name_locale=dn_locale,
                    description_locale=desc_locale,
                )

            case RemoteToolBinding(driver=driver):
                factory = self._executor_factories.get(driver)
                if factory is None:
                    raise ValueError(
                        f"Unknown remote tool driver '{driver}' for tool '{name}'. "
                        f"Available drivers: {list(self._executor_factories)}"
                    )
                executor = factory.create(binding)
                return RemoteTool(
                    name=name,
                    description=description,
                    parameters=parameters,
                    executor=executor,
                    display_name=display_name,
                    display_name_locale=dn_locale,
                    description_locale=desc_locale,
                )

            case _:
                raise ValueError(
                    f"Unsupported tool binding type for '{name}': "
                    f"{type(binding).__name__}"
                )


def _make_external_wrapper(
    uri: str,
    name: str,
    description: str,
    parameters: dict,
    display_name_locale: dict[str, str] | None,
    description_locale: dict[str, str] | None,
) -> Callable[..., AsyncIterator[Any]]:
    async def _dummy_fn(**kwargs: Any) -> AsyncIterator[Any]:
        yield None

    return ExternalToolWrapper(
        original_fn=_dummy_fn,
        script_path=uri,
        tool_name=name,
        tool_description=description,
        tool_params=parameters,
        display_name_locale=display_name_locale,
        description_locale=description_locale,
    )
