from __future__ import annotations

from minimal_harness.client.built_in.buffer import StreamBuffer
from minimal_harness.types import LLMChunkDelta, ToolCallDelta


class TestStreamBuffer:
    def test_init(self):
        buf = StreamBuffer()
        assert buf.content == ""
        assert buf.reasoning == ""
        assert buf.tool_calls == {}
        assert buf._flushed is False

    def test_add_chunk_none(self):
        buf = StreamBuffer()
        buf.add_chunk(None)
        assert buf.content == ""
        assert buf.reasoning == ""
        assert buf.tool_calls == {}

    def test_add_chunk_content(self):
        buf = StreamBuffer()
        buf.add_chunk(LLMChunkDelta(content="Hello "))
        buf.add_chunk(LLMChunkDelta(content="world!"))
        assert buf.content == "Hello world!"
        assert buf.reasoning == ""

    def test_add_chunk_reasoning(self):
        buf = StreamBuffer()
        buf.add_chunk(LLMChunkDelta(reasoning="thinking step 1"))
        buf.add_chunk(LLMChunkDelta(reasoning="\nthinking step 2"))
        assert buf.reasoning == "thinking step 1\nthinking step 2"
        assert buf.content == ""

    def test_add_chunk_reasoning_and_content(self):
        buf = StreamBuffer()
        buf.add_chunk(LLMChunkDelta(reasoning="think...", content="answer!"))
        assert buf.reasoning == "think..."
        assert buf.content == "answer!"

    def test_add_chunk_tool_call_new(self):
        buf = StreamBuffer()
        delta = LLMChunkDelta(
            tool_calls=[ToolCallDelta(index=0, id="call_1", name="get_weather")]
        )
        buf.add_chunk(delta)
        assert 0 in buf.tool_calls
        assert buf.tool_calls[0]["id"] == "call_1"
        assert buf.tool_calls[0]["name"] == "get_weather"

    def test_add_chunk_tool_call_accumulate(self):
        buf = StreamBuffer()
        buf.add_chunk(
            LLMChunkDelta(tool_calls=[ToolCallDelta(index=0, arguments='{"loc')])
        )
        buf.add_chunk(
            LLMChunkDelta(
                tool_calls=[ToolCallDelta(index=0, arguments='ation": "NYC"}')]
            )
        )
        assert buf.tool_calls[0]["arguments"] == '{"location": "NYC"}'

    def test_add_chunk_tool_call_multiple_indices(self):
        buf = StreamBuffer()
        buf.add_chunk(
            LLMChunkDelta(
                tool_calls=[
                    ToolCallDelta(index=0, id="call_0", name="tool_a"),
                    ToolCallDelta(index=1, id="call_1", name="tool_b"),
                ]
            )
        )
        assert len(buf.tool_calls) == 2
        assert buf.tool_calls[0]["name"] == "tool_a"
        assert buf.tool_calls[1]["name"] == "tool_b"

    def test_add_chunk_tool_call_id_and_name_appended(self):
        buf = StreamBuffer()
        buf.add_chunk(LLMChunkDelta(tool_calls=[ToolCallDelta(index=0, name="get_")]))
        buf.add_chunk(
            LLMChunkDelta(tool_calls=[ToolCallDelta(index=0, name="weather")])
        )
        assert buf.tool_calls[0]["name"] == "get_weather"

    def test_clear(self):
        buf = StreamBuffer()
        buf.add_chunk(LLMChunkDelta(content="test", reasoning="thinking"))
        buf.add_chunk(
            LLMChunkDelta(tool_calls=[ToolCallDelta(index=0, id="call_1", name="tool")])
        )
        buf.clear()
        assert buf.content == ""
        assert buf.reasoning == ""
        assert buf.tool_calls == {}
        assert buf._flushed is False

    def test_flushed_flag_unchanged_by_add_chunk(self):
        buf = StreamBuffer()
        buf._flushed = True
        buf.add_chunk(LLMChunkDelta(content="more"))
        assert buf._flushed is True
