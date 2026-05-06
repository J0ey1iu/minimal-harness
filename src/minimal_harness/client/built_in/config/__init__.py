"""Configuration subpackage — settings, agents, models, and tools."""

from minimal_harness.client.built_in.config.agents import (
    SYSTEM_PROMPTS_DIR,
    ensure_agents_config,
    ensure_system_prompts_dir,
    list_system_prompts,
    load_agents_config,
    read_system_prompt,
)
from minimal_harness.client.built_in.config.models import (
    add_model,
    load_models,
    save_models,
)
from minimal_harness.client.built_in.config.settings import (
    CONFIG_FILE,
    DEFAULT_CONFIG,
    load_config,
    save_config,
)
from minimal_harness.client.built_in.config.tools import collect_tools

__all__ = [
    "CONFIG_FILE",
    "DEFAULT_CONFIG",
    "add_model",
    "collect_tools",
    "ensure_agents_config",
    "ensure_system_prompts_dir",
    "list_system_prompts",
    "SYSTEM_PROMPTS_DIR",
    "load_agents_config",
    "load_config",
    "load_models",
    "read_system_prompt",
    "save_config",
    "save_models",
]
