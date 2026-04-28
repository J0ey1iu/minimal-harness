from __future__ import annotations

import json
from typing import Any

from minimal_harness.client.built_in.constants import MAX_DISPLAY_LENGTH
from minimal_harness.client.built_in.renderer import (
    format_tool_call_static,
    format_tool_result_static,
    truncate_static,
)


class TestFormatToolCallStatic:
    def test_valid_json_args(self):
        call = {"name": "get_weather", "arguments": '{"location": "NYC"}'}
        result = format_tool_call_static(call)
        text = str(result)
        assert "get_weather" in text
        assert "NYC" in text

    def test_invalid_json_args(self):
        call = {"name": "bad_tool", "arguments": "not-json-at-all"}
        result = format_tool_call_static(call)
        assert "bad_tool" in result.plain
        assert "not-json-at-all" in result.plain

    def test_no_name(self):
        call = {"arguments": "{}"}
        result = format_tool_call_static(call)
        assert "?" in result.plain

    def test_empty_args(self):
        call = {"name": "noop", "arguments": "{}"}
        result = format_tool_call_static(call)
        assert "noop()" in result.plain or "noop" in result.plain

    def test_none_args(self):
        call = {"name": "noop", "arguments": ""}
        result = format_tool_call_static(call)
        assert "noop()" in result.plain or "noop" in result.plain

    def test_no_args_key(self):
        call = {"name": "simple"}
        result = format_tool_call_static(call)
        assert "simple" in result.plain

    def test_args_pretty_printed(self):
        call = {"name": "test", "arguments": '{"a":1,"b":2}'}
        result = format_tool_call_static(call)
        parsed = json.loads(call["arguments"])
        expected_dump = json.dumps(parsed, ensure_ascii=False)
        assert expected_dump in str(result)


class TestFormatToolResultStatic:
    def test_error_result(self):
        result = {
            "error": "Something broke",
            "traceback": "Traceback...",
            "stderr": "Error details",
        }
        text = format_tool_result_static(result)
        assert "Something broke" in text.plain
        assert "Traceback" in text.plain
        assert "Error details" in text.plain

    def test_error_no_traceback(self):
        result = {"error": "fail"}
        text = format_tool_result_static(result)
        assert "fail" in text.plain

    def test_success_dict(self):
        result = {"temperature": 72, "unit": "F"}
        text = format_tool_result_static(result)
        assert "72" in text.plain
        assert "F" in text.plain

    def test_success_str(self):
        result = "command executed successfully"
        text = format_tool_result_static(result)
        assert "command executed successfully" in text.plain

    def test_success_non_dict_non_str(self):
        result: Any = 42
        text = format_tool_result_static(result)
        assert "42" in text.plain

    def test_truncate_long_result(self):
        long_str = "x" * (MAX_DISPLAY_LENGTH + 100)
        text = format_tool_result_static(long_str)
        assert len(text.plain) <= MAX_DISPLAY_LENGTH + 1  # +1 for the ellipsis
        assert text.plain.endswith("…")

    def test_short_result_not_truncated(self):
        short = "hello world"
        text = format_tool_result_static(short)
        assert text.plain == short

    def test_dict_result_truncated(self):
        long_val = "x" * (MAX_DISPLAY_LENGTH + 100)
        result = {"data": long_val}
        text = format_tool_result_static(result)
        assert len(text.plain) <= MAX_DISPLAY_LENGTH + 1


class TestTruncateStatic:
    def test_short_text(self):
        assert truncate_static("short") == "short"

    def test_long_text(self):
        long_str = "a" * (MAX_DISPLAY_LENGTH + 50)
        result = truncate_static(long_str)
        assert len(result) == MAX_DISPLAY_LENGTH + 1
        assert result.endswith("…")

    def test_exactly_max(self):
        exact = "b" * MAX_DISPLAY_LENGTH
        assert truncate_static(exact) == exact

    def test_custom_max_len(self):
        result = truncate_static("hello world", max_len=5)
        assert result == "hello…"
        assert len(result) == 6
