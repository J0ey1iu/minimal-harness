from typing import Any, Awaitable, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


def register_tool(
    name: str | None = None,
    description: str | None = None,
    parameters: dict | None = None,
) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        from minimal_harness.tool import Tool
        from minimal_harness.tool.registry import ToolRegistry

        tool_name = name or fn.__name__
        tool_description = description or (fn.__doc__ or "").strip()
        tool_params = parameters or {}

        ToolRegistry.get_instance().register(
            Tool(
                name=tool_name,
                description=tool_description,
                parameters=tool_params,
                fn=fn,
            )
        )
        return fn

    return decorator


def register(
    name: str,
    description: str,
    parameters: dict,
    fn: Callable[..., Awaitable[Any]],
) -> None:
    from minimal_harness.tool import Tool
    from minimal_harness.tool.registry import ToolRegistry

    ToolRegistry.get_instance().register(
        Tool(name=name, description=description, parameters=parameters, fn=fn)
    )


def unregister(name: str) -> bool:
    from minimal_harness.tool.registry import ToolRegistry

    return ToolRegistry.get_instance().unregister(name)
