"""Config path resolution.

Resolves the configuration directory at startup and caches the result.
When the current working directory has no `.minimal_harness/`, the home
config is bootstrapped (copied, excluding session data) so each working
directory gets its own independent configuration.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

_config_dir: Path | None = None

HOME_CONFIG = Path.home() / ".minimal_harness"


def get_config_dir() -> Path:
    """Return the resolved config directory.

    If :func:`resolve_config_dir` has not been called yet, auto-resolves
    with the default detection logic.
    """
    return _config_dir if _config_dir is not None else resolve_config_dir()


def resolve_config_dir(explicit: Path | str | None = None) -> Path:
    """Resolve and cache the configuration directory.

    Detection priority:
    1. ``explicit`` parameter (e.g. from CLI)
    2. ``MH_CONFIG_DIR`` environment variable
    3. ``./.minimal_harness/`` if it exists in the current working directory
    4. Bootstrap from ``~/.minimal_harness/`` into CWD, then use CWD
    """
    global _config_dir

    if _config_dir is not None:
        return _config_dir

    if explicit:
        _config_dir = Path(explicit).resolve()
        return _config_dir

    env = os.environ.get("MH_CONFIG_DIR")
    if env:
        _config_dir = Path(env).resolve()
        return _config_dir

    cwd_config = Path.cwd() / ".minimal_harness"
    if cwd_config.exists():
        _config_dir = cwd_config
        return _config_dir

    _bootstrap_from_home(cwd_config)
    _config_dir = cwd_config
    return _config_dir


def _bootstrap_from_home(target: Path) -> None:
    """Ensure home has defaults, then copy to *target* (excluding session data)."""
    _ensure_home_defaults(HOME_CONFIG)

    target.mkdir(parents=True, exist_ok=True)

    for name in ("config.json", "agents.json", "models.json"):
        src = HOME_CONFIG / name
        if src.exists():
            shutil.copy2(src, target / name)

    prompts_src = HOME_CONFIG / "system-prompts"
    prompts_dst = target / "system-prompts"
    if prompts_src.is_dir() and not prompts_dst.exists():
        shutil.copytree(prompts_src, prompts_dst)


def _ensure_home_defaults(home_dir: Path) -> None:
    """Create default config files in *home_dir* if they don't exist."""
    from minimal_harness.client.built_in.config.defaults import DEFAULT_CONFIG

    home_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = home_dir / "system-prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    config_file = home_dir / "config.json"
    if not config_file.exists():
        config_file.write_text(
            json.dumps(DEFAULT_CONFIG, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    _ensure_agents_defaults(home_dir)


def _ensure_agents_defaults(home_dir: Path) -> None:
    """Ensure agents.json and system prompt files exist in *home_dir*."""
    from minimal_harness.client.built_in.config.defaults import (
        AGENT_PROMPTS,
        DEFAULT_AGENTS,
    )

    agents_file = home_dir / "agents.json"
    if not agents_file.exists():
        agents_file.write_text(
            json.dumps(DEFAULT_AGENTS, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        prompts_dir = home_dir / "system-prompts"
        for filename, content in AGENT_PROMPTS.items():
            path = prompts_dir / filename
            if not path.exists():
                path.write_text(content, encoding="utf-8")
