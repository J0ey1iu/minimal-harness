"""Shared default configuration data.

All modules in the config subpackage import defaults from here to avoid
circular imports. This module has zero internal dependencies.
"""

from __future__ import annotations

from typing import Any

from minimal_harness.settings import Settings

DEFAULT_CONFIG: dict[str, Any] = {
    "base_url": Settings.base_url(),
    "api_key": Settings.api_key(),
    "model": Settings.model(),
    "tools_path": "",
    "theme": Settings.theme(),
    "provider": "openai",
    "reasoning_effort": None,
    "default_agent": "general_assistant",
}

DEFAULT_AGENTS: list[dict[str, Any]] = [
    {
        "name": "general_assistant",
        "display_name": "General Assistant",
        "description": "General-purpose assistant for everyday tasks, Q&A, and conversation",
        "system_prompt": "general_assistant.md",
        "default_tools": ["handoff", "discover_agents"],
    },
    {
        "name": "code_assistant",
        "display_name": "Code Assistant",
        "description": "Specialized in software development, debugging, code review, and architecture",
        "system_prompt": "code_assistant.md",
        "default_tools": ["handoff", "discover_agents"],
    },
    {
        "name": "research_assistant",
        "display_name": "Research Assistant",
        "description": "Focused on deep research, analysis, fact-checking, and information synthesis",
        "system_prompt": "research_assistant.md",
        "default_tools": ["handoff", "discover_agents"],
    },
]

AGENT_PROMPTS: dict[str, str] = {
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
