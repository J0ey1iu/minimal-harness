"""Backward compatibility shim for minimal_harness.client.built_in.tui."""

from minimal_harness.client.built_in import (
    DEFAULT_CONFIG,
    J0EY1IU_QUOTES,
    THEMES,
    ChatInput,
    ConfigScreen,
    ConfirmScreen,
    PromptScreen,
    StreamBuffer,
    ToolSelectScreen,
    TUIApp,
    collect_tools,
    load_config,
    main,
    save_config,
)

__all__ = [
    "TUIApp",
    "main",
    "StreamBuffer",
    "DEFAULT_CONFIG",
    "J0EY1IU_QUOTES",
    "THEMES",
    "collect_tools",
    "load_config",
    "save_config",
    "ConfigScreen",
    "ConfirmScreen",
    "PromptScreen",
    "ToolSelectScreen",
    "ChatInput",
]