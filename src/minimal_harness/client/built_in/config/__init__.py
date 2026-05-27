"""Configuration subpackage — settings, agents, models, and tools."""

from minimal_harness.client.built_in.config.agents import (
    ensure_agents_config,
    ensure_system_prompts_dir,
    list_system_prompts,
    load_agents_config,
    read_system_prompt,
)
from minimal_harness.client.built_in.config.defaults import DEFAULT_CONFIG
from minimal_harness.client.built_in.config.models import (
    add_model,
    load_models,
    save_models,
)
from minimal_harness.client.built_in.config.paths import (
    get_config_dir,
    resolve_config_dir,
)
from minimal_harness.client.built_in.config.settings import (
    load_config,
    save_config,
)
from minimal_harness.client.built_in.config.tools import collect_tools

__all__ = [
    "DEFAULT_CONFIG",
    "add_model",
    "collect_tools",
    "ensure_agents_config",
    "ensure_system_prompts_dir",
    "get_config_dir",
    "list_system_prompts",
    "load_agents_config",
    "load_config",
    "load_models",
    "read_system_prompt",
    "resolve_config_dir",
    "save_config",
    "save_models",
]
