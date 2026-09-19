"""Anthropic Skills frontmatter parser (``SKILL.md`` folder format).

RFC #57 (mhc-desktop): the base parsed script-tool metadata
(:mod:`minimal_harness.tool.script_parser`) but had no counterpart for
**Anthropic Skills** — a folder containing a ``SKILL.md`` with YAML
frontmatter:

    ---
    name: my-skill
    description: When to use this skill
    ---

    # body content follows

We accept the standard Anthropic names (``name`` and ``description``)
plus the optional fields the proposers use (``version``, ``license``).
Anything else in the frontmatter is preserved verbatim on the dataclass
so consumers can extend without us shipping a parser for their keys.

Format spec reference:
https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview

API mirrors ``tool/script_parser.py``: :func:`parse_skill_md` /
:func:`render_skill_md` for text round-trips plus
:func:`install_from_folder` for the folder-level loader the RFC asks
for. PyYAML is a hard dependency (the same version the proposers'
client already uses), so ``yaml.safe_load`` behaves identically on both
sides.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Mirrors the proposers' frontmatter module: PyYAML is a hard dependency
# of the SDK, but the import stays lazy with a graceful error so that
# environments with a partial install can still import this module (and
# the rest of minimal_harness) without crashing on `import yaml`.
try:
    import yaml  # type: ignore[import-untyped]

    _HAVE_YAML = True
except Exception:  # pragma: no cover - depends on environment
    yaml = None  # type: ignore[assignment]
    _HAVE_YAML = False


def _yaml():
    """Return the (imported) yaml module."""
    if not _HAVE_YAML or yaml is None:  # pragma: no cover
        raise FrontmatterError("PyYAML not installed; cannot parse SKILL.md")
    return yaml


_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(.*?)\n---\s*(?:\n|$)(.*)\Z",
    re.DOTALL,
)


class FrontmatterError(ValueError):
    """Raised when SKILL.md frontmatter is missing or malformed."""


@dataclass
class SkillFrontmatter:
    """Frontmatter block of a SKILL.md file."""

    name: str = ""
    description: str = ""
    version: str = ""
    license: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty = OK)."""
        errors: list[str] = []
        if not self.name:
            errors.append("name is required")
        elif not _NAME_RE.match(self.name):
            errors.append(
                "name must be lowercase letters, digits, and hyphens "
                "(start/end with letter or digit, no consecutive hyphens)"
            )
        elif len(self.name) > 64:
            errors.append("name must be 64 characters or fewer")
        if not self.description:
            errors.append("description is required")
        elif len(self.description) > 1024:
            errors.append("description must be 1024 characters or fewer")
        return errors


def parse_skill_md(text: str) -> tuple[SkillFrontmatter, str]:
    """Split a SKILL.md file into ``(frontmatter, body)``.

    Body is the markdown following the closing ``---`` delimiter. If no
    frontmatter block is present we return an empty frontmatter and the
    full text as the body, so the caller can decide what to do.

    Raises :class:`FrontmatterError` if the YAML between the delimiters
    is not a mapping (e.g. a string or list).
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return SkillFrontmatter(), text

    raw_yaml, body = m.group(1), m.group(2)
    parsed = _yaml().safe_load(raw_yaml) or {}
    if not isinstance(parsed, dict):
        raise FrontmatterError(
            "SKILL.md frontmatter must be a YAML mapping "
            "(e.g. 'name:' / 'description:')"
        )

    fm = SkillFrontmatter(
        name=str(parsed.get("name", "")).strip(),
        description=str(parsed.get("description", "")).strip(),
        version=str(parsed.get("version", "")).strip(),
        license=str(parsed.get("license", "")).strip(),
    )
    known = {"name", "description", "version", "license"}
    fm.extra = {k: v for k, v in parsed.items() if k not in known}
    return fm, body.lstrip("\n")


def render_skill_md(fm: SkillFrontmatter, body: str) -> str:
    """Inverse of :func:`parse_skill_md`.

    Used when editing metadata through a UI without losing the body.
    Unknown frontmatter keys are preserved verbatim via ``extra``.
    """
    out: dict[str, Any] = {"name": fm.name, "description": fm.description}
    if fm.version:
        out["version"] = fm.version
    if fm.license:
        out["license"] = fm.license
    out.update(fm.extra)
    yaml_text = _yaml().safe_dump(out, sort_keys=False, allow_unicode=True).rstrip()
    body = body.lstrip("\n").rstrip() + "\n"
    return f"---\n{yaml_text}\n---\n\n{body}"


@dataclass
class SkillMetadata:
    """A skill installed from a folder: ``frontmatter`` + ``body``."""

    folder: str
    frontmatter: SkillFrontmatter
    body: str


def install_from_folder(path: str | Path) -> SkillMetadata:
    """Load a skill folder's ``SKILL.md`` into :class:`SkillMetadata`.

    Returns the parsed frontmatter and markdown body. Raises
    :class:`FrontmatterError` when the folder has no ``SKILL.md``.
    """
    folder = Path(path).expanduser().resolve()
    skill_file = folder / "SKILL.md"
    if not skill_file.is_file():
        raise FrontmatterError(f"Skill folder '{folder}' has no SKILL.md")
    text = skill_file.read_text(encoding="utf-8", errors="replace")
    fm, body = parse_skill_md(text)
    return SkillMetadata(folder=str(folder), frontmatter=fm, body=body)


__all__ = [
    "FrontmatterError",
    "SkillFrontmatter",
    "SkillMetadata",
    "install_from_folder",
    "parse_skill_md",
    "render_skill_md",
]
