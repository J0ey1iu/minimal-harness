"""Configuration helpers for the TUI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from minimal_harness.settings import Settings
from minimal_harness.tool.base import StreamingTool
from minimal_harness.tool.built_in.bash import get_tools as get_bash_tools
from minimal_harness.tool.built_in.local_file_operation import (
    get_tools as get_local_file_operation_tools,
)
from minimal_harness.tool.external_loader import load_external_tools
from minimal_harness.tool.registry import ToolRegistry

CONFIG_FILE = Path.home() / ".minimal_harness" / "config.json"
SYSTEM_PROMPTS_DIR = Path.home() / ".minimal_harness" / "system-prompts"

DEFAULT_CONFIG: dict[str, Any] = {
    "base_url": Settings.base_url(),
    "api_key": Settings.api_key(),
    "model": Settings.model(),
    "tools_path": "",
    "theme": Settings.theme(),
    "provider": "openai",
    "default_agent": "general_assistant",
}

MODELS_FILE = Path.home() / ".minimal_harness" / "models.json"
AGENTS_FILE = Path.home() / ".minimal_harness" / "agents.json"

_DEFAULT_AGENTS: list[dict[str, Any]] = [
    {
        "name": "general_assistant",
        "description": "General-purpose assistant for everyday tasks, Q&A, and conversation",
        "system_prompt": "general_assistant.md",
        "default_tools": [],
    },
    {
        "name": "code_assistant",
        "description": "Specialized in software development, debugging, code review, and architecture",
        "system_prompt": "code_assistant.md",
        "default_tools": [],
    },
    {
        "name": "research_assistant",
        "description": "Focused on deep research, analysis, fact-checking, and information synthesis",
        "system_prompt": "research_assistant.md",
        "default_tools": [],
    },
]

_AGENT_PROMPTS: dict[str, str] = {
    "general_assistant.md": (
        "You are a versatile general-purpose assistant. "
        "You excel at handling everyday tasks, answering questions, "
        "engaging in conversation, and helping with a wide variety of topics. "
        "Be helpful, friendly, and thorough in your responses."
    ),
    "code_assistant.md": (
        "You are a specialized coding assistant with deep expertise "
        "in software development. You excel at writing, debugging, "
        "reviewing, and refactoring code across multiple programming "
        "languages. Provide clear explanations, best practices, "
        "and well-structured code examples."
    ),
    "research_assistant.md": (
        "You are a research-focused assistant specialized in deep analysis "
        "and information synthesis. You excel at breaking down complex topics, "
        "verifying facts, connecting ideas across domains, and presenting "
        "well-structured findings. Be thorough, precise, and cite your reasoning."
    ),
}


def load_models() -> list[str]:
    if MODELS_FILE.exists():
        try:
            data = json.loads(MODELS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(m) for m in data if m]
        except (json.JSONDecodeError, OSError):
            pass
    return []


def save_models(models: list[str]) -> None:
    MODELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODELS_FILE.write_text(
        json.dumps(models, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def add_model(model: str) -> None:
    if not model:
        return
    models = load_models()
    if model not in models:
        models.insert(0, model)
        save_models(models)


def load_config() -> dict[str, Any]:
    ensure_system_prompts_dir()
    ensure_agents_config()
    file_existed = CONFIG_FILE.exists()
    if file_existed:
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            data.pop("selected_tools", None)
            config = {
                **DEFAULT_CONFIG,
                **{k: data[k] for k in DEFAULT_CONFIG if k in data},
            }
        except (json.JSONDecodeError, OSError):
            config = dict(DEFAULT_CONFIG)
    else:
        config = dict(DEFAULT_CONFIG)
    if file_existed:
        save_config(config)

    if not MODELS_FILE.exists():
        model = config.get("model", "")
        if model:
            save_models([model])

    return config


def save_config(config: dict[str, Any]) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def ensure_system_prompts_dir() -> None:
    SYSTEM_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)


def list_system_prompts() -> list[Path]:
    if not SYSTEM_PROMPTS_DIR.exists():
        return []
    return sorted(SYSTEM_PROMPTS_DIR.glob("*.md"))


def read_system_prompt(path: Path) -> str:
    if path.exists() and path.is_file():
        return path.read_text(encoding="utf-8")
    return ""


def load_agents_config() -> list[dict[str, Any]]:
    if AGENTS_FILE.exists():
        try:
            data = json.loads(AGENTS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                result = []
                for a in data:
                    if isinstance(a, dict) and "name" in a:
                        result.append(
                            {
                                "name": str(a["name"]),
                                "description": str(a.get("description", "")),
                                "system_prompt": str(a.get("system_prompt", "")),
                                "default_tools": list(a.get("default_tools", [])),
                            }
                        )
                return result
        except (json.JSONDecodeError, OSError, KeyError):
            pass
    return []


def ensure_agents_config() -> None:
    AGENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not AGENTS_FILE.exists():
        AGENTS_FILE.write_text(
            json.dumps(_DEFAULT_AGENTS, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        try:
            data = json.loads(AGENTS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                changed = False
                for a in data:
                    if isinstance(a, dict) and "default_tools" not in a:
                        a["default_tools"] = []
                        changed = True
                has_general = any(
                    isinstance(a, dict) and a.get("name") == "general_assistant"
                    for a in data
                )
                if not has_general:
                    data.insert(0, dict(_DEFAULT_AGENTS[0]))
                    changed = True
                if changed:
                    AGENTS_FILE.write_text(
                        json.dumps(data, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
        except (json.JSONDecodeError, OSError):
            pass
    for filename, content in _AGENT_PROMPTS.items():
        path = SYSTEM_PROMPTS_DIR / filename
        if not path.exists():
            path.write_text(content, encoding="utf-8")


def collect_tools(
    config: dict[str, Any],
    registry: ToolRegistry,
) -> dict[str, StreamingTool]:
    if path := config.get("tools_path", "").strip():
        load_external_tools(path, registry)
    tools: dict[str, StreamingTool] = {}
    for getter in (get_bash_tools, get_local_file_operation_tools):
        tools.update(getter())
    for t in registry.get_all():
        if t.name in tools:
            import warnings

            warnings.warn(
                f"External tool '{t.name}' overwrites built-in tool of the same name."
            )
        tools[t.name] = t
    return tools
