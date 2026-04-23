"""Test external_loader.py - verify external tools use their own interpreter."""

import sys
import tempfile
from pathlib import Path

import pytest

from minimal_harness.tool.external_loader import load_tools_from_file


@pytest.fixture
def temp_tool_script():
    """Create a temporary tool script that reports which Python is running."""
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "test_tool.py"
        shebang_line = f"#!{sys.executable}"
        script_content = f'''{shebang_line}
import sys
import json
from typing import AsyncIterator

def get_interpreter_tool(value: str) -> AsyncIterator[dict]:
    """Report the Python interpreter being used."""
    yield {{"interpreter": sys.executable, "value": value, "python_version": sys.version}}

register("get_interpreter_tool", "Report interpreter", {{"type": "object", "properties": {{"value": {{"type": "string"}}}}}}, get_interpreter_tool)
'''
        script_path.write_text(script_content, encoding="utf-8")
        yield script_path


@pytest.mark.asyncio
async def test_external_tool_uses_script_interpreter(temp_tool_script):
    """Verify that external tools run with the script's Python interpreter, not the harness's."""
    from minimal_harness.tool.registry import ToolRegistry

    registry = ToolRegistry()
    tool_names = load_tools_from_file(temp_tool_script, registry)

    assert len(tool_names) == 1
    assert tool_names[0] == "get_interpreter_tool"

    tool = registry.get(tool_names[0])
    assert tool is not None

    results = []
    async for chunk in tool.fn(value="test_value"):
        results.append(chunk)

    assert len(results) == 1
    result = results[0]
    assert "interpreter" in result
    assert result["interpreter"] == sys.executable
    assert result["value"] == "test_value"


@pytest.mark.asyncio
async def test_external_tool_detects_shebang(temp_tool_script):
    """Verify that the shebang is correctly detected from the script."""
    content = temp_tool_script.read_text(encoding="utf-8")
    first_line = content.splitlines()[0]
    assert first_line.startswith("#!")
    assert "python" in first_line.lower()


@pytest.fixture
def temp_script_with_subprocess_check():
    """Create a script that uses subprocess to verify interpreter isolation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "subprocess_tool.py"
        shebang_line = f"#!{sys.executable}"
        script_content = f'''{shebang_line}
import sys
import json
import subprocess
from typing import AsyncIterator

def subprocess_check_tool() -> AsyncIterator[dict]:
    """Verify the subprocess uses the same interpreter as the script."""
    result = subprocess.run(
        [{sys.executable!r}, "-c", "import sys; print(sys.executable)"],
        capture_output=True,
        text=True
    )
    yield {{
        "parent_interpreter": sys.executable,
        "subprocess_interpreter": result.stdout.strip(),
        "match": result.stdout.strip() == sys.executable
    }}

register("subprocess_check_tool", "Verify subprocess interpreter", {{}}, subprocess_check_tool)
'''
        script_path.write_text(script_content, encoding="utf-8")
        yield script_path


@pytest.mark.asyncio
async def test_external_tool_subprocess_uses_same_interpreter(temp_script_with_subprocess_check):
    """Verify that subprocesses spawned by external tools use the same interpreter as the script."""
    from minimal_harness.tool.registry import ToolRegistry

    registry = ToolRegistry()
    tool_names = load_tools_from_file(temp_script_with_subprocess_check, registry)

    assert len(tool_names) == 1
    assert tool_names[0] == "subprocess_check_tool"

    tool = registry.get(tool_names[0])

    assert tool is not None
    assert callable(tool.fn)
    results = []
    async for chunk in tool.fn():
        results.append(chunk)

    assert len(results) == 1
    result = results[0]
    assert result["match"] is True
    assert result["parent_interpreter"] == sys.executable
    assert result["subprocess_interpreter"] == sys.executable


def test_load_tools_from_file_returns_empty_for_nonexistent():
    """Verify load_tools_from_file returns empty list for nonexistent files."""
    from minimal_harness.tool.registry import ToolRegistry

    registry = ToolRegistry()
    result = load_tools_from_file("/nonexistent/path/to/tool.py", registry)
    assert result == []


def test_load_tools_from_file_with_register_decorator(temp_tool_script):
    """Test that @register decorator works correctly in external scripts."""
    from minimal_harness.tool.registry import ToolRegistry

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "decorated_tool.py"
        shebang_line = f"#!{sys.executable}"
        script_content = f'''{shebang_line}
import sys
import json
from typing import AsyncIterator

def custom_tool_fn(msg: str) -> AsyncIterator[dict]:
    yield {{"received": msg, "interpreter": sys.executable}}

register("custom_named_tool", "A tool with custom name via register", {{}}, custom_tool_fn)
'''
        script_path.write_text(script_content, encoding="utf-8")
        registry = ToolRegistry()
        tool_names = load_tools_from_file(script_path, registry)

        assert len(tool_names) == 1
        assert tool_names[0] == "custom_named_tool"

        tool = registry.get("custom_named_tool")
        assert tool is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
