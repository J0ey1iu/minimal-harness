"""Agent configuration and system prompt management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SYSTEM_PROMPTS_DIR = Path.home() / ".minimal_harness" / "system-prompts"
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
