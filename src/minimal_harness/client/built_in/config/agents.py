"""Agent configuration and system prompt management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from minimal_harness.client.built_in.config.defaults import (
    AGENT_PROMPTS,
    DEFAULT_AGENTS,
)

_AGENT_PROMPTS_DIR = "system-prompts"


def _get_base_dir() -> Path:
    from minimal_harness.client.built_in.config.paths import get_config_dir

    return get_config_dir()


def _get_agents_file() -> Path:
    return _get_base_dir() / "agents.json"


def _get_prompts_dir() -> Path:
    return _get_base_dir() / _AGENT_PROMPTS_DIR


def ensure_system_prompts_dir() -> None:
    _get_prompts_dir().mkdir(parents=True, exist_ok=True)


def list_system_prompts() -> list[Path]:
    d = _get_prompts_dir()
    if not d.exists():
        return []
    return sorted(d.glob("*.md"))


def read_system_prompt(path: Path) -> str:
    if path.exists() and path.is_file():
        return path.read_text(encoding="utf-8")
    return ""


def load_agents_config() -> list[dict[str, Any]]:
    agents_file = _get_agents_file()
    if agents_file.exists():
        try:
            data = json.loads(agents_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                result = []
                for a in data:
                    if isinstance(a, dict) and "name" in a:
                        result.append(
                            {
                                "name": str(a["name"]),
                                "display_name": str(a.get("display_name", "")),
                                "description": str(a.get("description", "")),
                                "system_prompt": str(a.get("system_prompt", "")),
                                "default_tools": list(a.get("default_tools", [])),
                            }
                        )
                return result
        except (json.JSONDecodeError, OSError, KeyError):
            pass
    return []


_RUNTIME_TOOL_NAMES = ["handoff", "discover_agents"]


def ensure_agents_config() -> None:
    agents_file = _get_agents_file()
    agents_file.parent.mkdir(parents=True, exist_ok=True)
    if not agents_file.exists():
        agents_file.write_text(
            json.dumps(DEFAULT_AGENTS, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        try:
            data = json.loads(agents_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                changed = False
                for a in data:
                    if isinstance(a, dict) and "default_tools" not in a:
                        a["default_tools"] = list(_RUNTIME_TOOL_NAMES)
                        changed = True
                    elif isinstance(a, dict):
                        current = a.get("default_tools", [])
                        missing = [t for t in _RUNTIME_TOOL_NAMES if t not in current]
                        if missing:
                            a["default_tools"] = current + missing
                            changed = True
                has_general = any(
                    isinstance(a, dict) and a.get("name") == "general_assistant"
                    for a in data
                )
                if not has_general:
                    data.insert(0, dict(DEFAULT_AGENTS[0]))
                    changed = True
                if changed:
                    agents_file.write_text(
                        json.dumps(data, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
        except (json.JSONDecodeError, OSError):
            pass
    prompts_dir = _get_prompts_dir()
    for filename, content in AGENT_PROMPTS.items():
        path = prompts_dir / filename
        if not path.exists():
            path.write_text(content, encoding="utf-8")
