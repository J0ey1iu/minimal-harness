"""AST-based metadata extraction from one-tool-per-script files.

This module parses Python tool scripts **without executing them** to
extract standard metadata variables and find the ``execute()`` function.

Required fields (upload is rejected if any is missing):
- ``TOOL_NAME`` — unique tool name
- ``TOOL_DISPLAY_NAME_LOCALE`` — dict like ``{"zh": "中文名", "en": "Name"}``
- ``TOOL_DESCRIPTION`` — human-readable description
- ``TOOL_DESCRIPTION_LOCALE`` — dict like ``{"zh": "中文描述", "en": "Description"}``
- ``TOOL_PARAMETERS`` — JSON Schema dict for tool parameters
- ``execute()`` — async generator/coroutine function

Optional:
- ``TOOL_DISPLAY_NAME`` — defaults to ``TOOL_NAME``
- Shebang line (``#!python3``) — defaults to ``sys.executable``
"""

from __future__ import annotations

import ast
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ScriptParseResult:
    """Parsed metadata from a tool script file."""

    name: str
    description: str
    parameters: dict
    display_name: str = ""
    display_name_locale: dict[str, str] | None = None
    description_locale: dict[str, str] | None = None
    script_path: str = ""
    errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = self.name

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


def _validate_locale_dict(value: object, var_name: str) -> dict[str, str] | None:
    """Validate a locale dict value. Returns None on error (appends to *errors*)."""
    if not isinstance(value, dict):
        return None
    result: dict[str, str] = {}
    for k, v in value.items():
        if not isinstance(k, str) or not isinstance(v, str):
            return None
        if k.strip() and v.strip():
            result[k.strip()] = v.strip()
    return result if result else None


def check_interpreter(path: str | Path) -> str | None:
    """Validate that the shebang interpreter in the script exists.

    Reads first line of *path*.  If it's a ``#!python`` shebang,
    checks that the interpreter command is available via
    ``shutil.which`` (or a direct file-existence check for absolute
    paths).  Returns ``None`` if the interpreter is valid, or an
    error message string otherwise.
    """
    file_path = Path(path).expanduser().resolve()
    try:
        with file_path.open(encoding="utf-8", errors="ignore") as f:
            shebang = f.readline()
    except Exception:
        return None

    if not shebang.startswith("#!") or "python" not in shebang.lower():
        return None

    parts = shebang[2:].strip().split()
    if not parts:
        return None

    interp_cmd = parts[0]

    # Absolute path — check file exists
    if interp_cmd.startswith("/") or interp_cmd.startswith("\\") or ":" in interp_cmd:
        if not Path(interp_cmd).is_file():
            return f"Interpreter not found: {interp_cmd}"
        return None

    # Relative name — check via PATH
    if interp_cmd == "env" and len(parts) > 1:
        # #!/usr/bin/env python3 → env is the cmd, python3 is the arg
        found = shutil.which(parts[1])
        if not found:
            return (
                f"Interpreter '{parts[1]}' not found in PATH "
                f"(from shebang: '{shebang.strip()[:60]}')"
            )
    else:
        found = shutil.which(interp_cmd)
        if not found:
            return (
                f"Interpreter '{interp_cmd}' not found in PATH "
                f"(from shebang: '{shebang.strip()[:60]}')"
            )
    return None


def parse_tool_script(path: str | Path) -> ScriptParseResult:
    """Parse a single-file Python tool script and extract metadata.

    Returns a ``ScriptParseResult`` which is valid only when all
    required fields are present and the shebang interpreter (if any)
    is available on the current system.

    Required: ``TOOL_NAME``, ``TOOL_DESCRIPTION``, ``TOOL_PARAMETERS``,
    an ``execute()`` function definition.
    """
    file_path = Path(path).expanduser().resolve()
    result = ScriptParseResult(
        name=file_path.stem,
        description="",
        parameters={},
        display_name=file_path.stem.replace("_", " ").title(),
        script_path=str(file_path),
    )

    try:
        source = file_path.read_text(encoding="utf-8")
    except Exception as exc:
        result.errors.append(f"Cannot read file: {exc}")
        return result

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        result.errors.append(f"Syntax error in script: {exc}")
        return result

    has_execute = False

    for node in ast.iter_child_nodes(tree):
        # ── capture TOOL_* variable assignments ──
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id == "TOOL_NAME":
                    try:
                        val = ast.literal_eval(node.value)
                        if isinstance(val, str) and val.strip():
                            result.name = val.strip()
                        else:
                            result.errors.append("TOOL_NAME must be a non-empty string")
                    except (ValueError, TypeError):
                        result.errors.append(
                            "TOOL_NAME could not be parsed as a literal"
                        )
                elif target.id == "TOOL_DISPLAY_NAME":
                    try:
                        val = ast.literal_eval(node.value)
                        if isinstance(val, str):
                            result.display_name = val
                    except (ValueError, TypeError):
                        result.errors.append("TOOL_DISPLAY_NAME could not be parsed")
                elif target.id == "TOOL_DISPLAY_NAME_LOCALE":
                    try:
                        val = ast.literal_eval(node.value)
                        parsed = _validate_locale_dict(val, "TOOL_DISPLAY_NAME_LOCALE")
                        if parsed is not None:
                            result.display_name_locale = parsed
                        else:
                            result.errors.append(
                                "TOOL_DISPLAY_NAME_LOCALE must be a dict "
                                "with non-empty string keys/values, "
                                'e.g. {"zh": "中文名", "en": "Name"}'
                            )
                    except (ValueError, TypeError):
                        result.errors.append(
                            "TOOL_DISPLAY_NAME_LOCALE could not be parsed"
                        )
                elif target.id == "TOOL_DESCRIPTION":
                    try:
                        val = ast.literal_eval(node.value)
                        if isinstance(val, str) and val.strip():
                            result.description = val.strip()
                        else:
                            result.errors.append(
                                "TOOL_DESCRIPTION must be a non-empty string"
                            )
                    except (ValueError, TypeError):
                        result.errors.append(
                            "TOOL_DESCRIPTION could not be parsed as a literal"
                        )
                elif target.id == "TOOL_DESCRIPTION_LOCALE":
                    try:
                        val = ast.literal_eval(node.value)
                        parsed = _validate_locale_dict(val, "TOOL_DESCRIPTION_LOCALE")
                        if parsed is not None:
                            result.description_locale = parsed
                        else:
                            result.errors.append(
                                "TOOL_DESCRIPTION_LOCALE must be a dict "
                                "with non-empty string keys/values, "
                                'e.g. {"zh": "中文描述", "en": "Description"}'
                            )
                    except (ValueError, TypeError):
                        result.errors.append(
                            "TOOL_DESCRIPTION_LOCALE could not be parsed"
                        )
                elif target.id == "TOOL_PARAMETERS":
                    try:
                        val = ast.literal_eval(node.value)
                        if isinstance(val, dict):
                            result.parameters = val
                        else:
                            result.errors.append(
                                "TOOL_PARAMETERS must be a dict (JSON Schema)"
                            )
                    except (ValueError, TypeError):
                        result.errors.append(
                            "TOOL_PARAMETERS could not be parsed as a literal"
                        )

        # ── detect execute() function ──
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "execute":
                has_execute = True

    # ── mandatory field checks ──
    if not result.name.strip():
        result.errors.append("TOOL_NAME is missing or empty")
    if not result.description.strip():
        result.errors.append("TOOL_DESCRIPTION is missing or empty")
    if not isinstance(result.parameters, dict) or not result.parameters:
        result.errors.append("TOOL_PARAMETERS is missing, empty, or not a valid dict")
    if not result.display_name_locale:
        result.errors.append(
            "TOOL_DISPLAY_NAME_LOCALE is missing or empty — "
            "must be a dict with locale keys e.g. "
            '{"zh": "中文名", "en": "Name"}'
        )
    if not result.description_locale:
        result.errors.append(
            "TOOL_DESCRIPTION_LOCALE is missing or empty — "
            "must be a dict with locale keys e.g. "
            '{"zh": "中文描述", "en": "Description"}'
        )
    if not has_execute:
        result.errors.append(
            "execute() function is missing — define async def execute(...)"
        )

    # ── interpreter check ──
    interp_err = check_interpreter(file_path)
    if interp_err:
        result.errors.append(interp_err)

    return result
