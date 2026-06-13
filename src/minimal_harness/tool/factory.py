from __future__ import annotations

from typing import Any, AsyncIterator, Callable, Protocol, runtime_checkable

from minimal_harness.tool.base import StreamingTool, Tool
from minimal_harness.tool.remote import RemoteTool, RemoteToolExecutor
from minimal_harness.tool.wrapper import ExternalToolWrapper
from minimal_harness.types import (
    ExternalScriptToolBinding,
    LocalToolBinding,
    RemoteToolBinding,
    ToolMetadata,
)


@runtime_checkable
class ToolExecutorFactory(Protocol):
    """Factory that creates a ``RemoteToolExecutor`` from a binding.

    The default SSE-over-HTTP implementation lives in
    :mod:`mh_service_kit.sse`. Consumers that use ``DefaultToolFactory``
    for remote bindings MUST register an executor factory per driver
    name.
    """

    def create(self, binding: RemoteToolBinding) -> RemoteToolExecutor: ...


class ToolFactory(Protocol):
    """Creates a concrete ``Tool`` from ``ToolMetadata``.

    Implement this protocol to customise how tool metadata is
    turned into an executable tool (e.g. wire up a custom
    ``RemoteToolExecutor``).
    """

    def create(self, metadata: ToolMetadata) -> Tool: ...


class DefaultToolFactory:
    """Default ``ToolFactory`` that handles all built-in binding types.

    Resolves local and external-script bindings directly. For
    ``RemoteToolBinding``, delegates to a registered
    ``ToolExecutorFactory`` keyed by driver name. The default
    driver ``"default"`` is NOT registered — consumers that rely
    on the SSE-over-HTTP executor should register it explicitly
    via ``executor_factories={"default": ...}``.
    """

    def __init__(
        self,
        executor_factories: dict[str, ToolExecutorFactory] | None = None,
    ) -> None:
        self._executor_factories: dict[str, ToolExecutorFactory] = dict(
            executor_factories or {}
        )

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

            case RemoteToolBinding():
                driver = binding.driver
                factory = self._executor_factories.get(driver)
                if factory is None:
                    raise ValueError(
                        f"Unknown remote tool driver '{driver}' for tool '{name}'. "
                        f"Registered drivers: {list(self._executor_factories)}. "
                        f"Register an executor factory via DefaultToolFactory({{'{driver}': ...}})."
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
                    endpoint_url=binding.url,
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
