"""Unit tests for the Anthropic-compatible LLM provider."""

import asyncio
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from anthropic.types import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    InputJSONDelta,
    Message,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    TextBlock,
    TextDelta,
    ToolUseBlock,
    Usage,
)
from anthropic.types.message_delta_usage import MessageDeltaUsage
from anthropic.types.raw_message_delta_event import Delta

from minimal_harness.llm.anthropic import AnthropicLLMProvider, _convert_messages
from minimal_harness.memory import (
    assistant_message,
    system_message,
    tool_message,
    user_message,
)
from minimal_harness.tool.base import StreamingTool
from minimal_harness.types import LLMChunkDelta, ToolCallDelta


async def _noop_streaming_tool_fn(**kwargs: object) -> AsyncIterator[dict]:
    yield {"result": "ok"}


class _MockAsyncStream:
    """Simple async stream that supports ``async with`` and ``async for``."""

    def __init__(self, items: list[Any]):
        self._items = items
        self._index = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args: object):
        pass

    def __aiter__(self):
        self._index = 0
        return self

    async def __anext__(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


@pytest.fixture
def mock_anthropic_client():
    client = MagicMock()
    client.messages = MagicMock()
    return client


class TestConvertMessages:
    def test_system_message_extracted(self):
        messages = [
            system_message("Be helpful."),
            user_message([{"type": "text", "text": "Hello"}]),
        ]
        system, anthropic_msgs = _convert_messages(messages)
        assert system == "Be helpful."
        assert anthropic_msgs == [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
        ]

    def test_user_text_message(self):
        messages = [user_message([{"type": "text", "text": "Hi"}])]
        system, anthropic_msgs = _convert_messages(messages)
        assert system is None
        assert anthropic_msgs == [
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]}
        ]

    def test_user_image_message(self):
        messages = [
            user_message([{"type": "image", "url": "data:image/png;base64,abc"}])
        ]
        system, anthropic_msgs = _convert_messages(messages)
        assert anthropic_msgs == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "[Image: data:image/png;base64,abc]"}
                ],
            }
        ]

    def test_user_file_message(self):
        messages = [
            user_message(
                [
                    {
                        "type": "file",
                        "file": {
                            "file_id": "1",
                            "file_name": "report.pdf",
                            "file_size": 1024,
                            "backend_type": "local",
                        },
                    }
                ]
            )
        ]
        system, anthropic_msgs = _convert_messages(messages)
        assert anthropic_msgs == [
            {
                "role": "user",
                "content": [{"type": "text", "text": "[File: report.pdf]"}],
            }
        ]

    def test_assistant_text_only(self):
        messages = [assistant_message("Hello back")]
        _, anthropic_msgs = _convert_messages(messages)
        assert anthropic_msgs == [
            {"role": "assistant", "content": [{"type": "text", "text": "Hello back"}]}
        ]

    def test_assistant_with_tool_calls(self):
        messages = [
            assistant_message(
                "Let me calculate",
                tool_calls=[
                    {
                        "id": "tu_1",
                        "type": "function",
                        "function": {"name": "calc", "arguments": '{"x": 1}'},
                    }
                ],
            )
        ]
        _, anthropic_msgs = _convert_messages(messages)
        assert anthropic_msgs == [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me calculate"},
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "calc",
                        "input": {"x": 1},
                    },
                ],
            }
        ]

    def test_assistant_with_invalid_tool_args(self):
        messages = [
            assistant_message(
                None,
                tool_calls=[
                    {
                        "id": "tu_1",
                        "type": "function",
                        "function": {"name": "calc", "arguments": "not-json"},
                    }
                ],
            )
        ]
        _, anthropic_msgs = _convert_messages(messages)
        assert anthropic_msgs[0]["content"] == [
            {"type": "tool_use", "id": "tu_1", "name": "calc", "input": {}}
        ]

    def test_tool_message(self):
        messages = [tool_message("tu_1", "42")]
        _, anthropic_msgs = _convert_messages(messages)
        assert anthropic_msgs == [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu_1", "content": "42"}
                ],
            }
        ]


@pytest.mark.asyncio
async def test_text_streaming(mock_anthropic_client: MagicMock):
    """Provider yields text chunks and ends with an LLMResponse."""
    events = [
        MessageStartEvent(
            type="message_start",
            message=Message(
                id="msg_1",
                type="message",
                role="assistant",
                content=[],
                model="claude-3",
                stop_reason=None,
                stop_sequence=None,
                usage=Usage(input_tokens=3, output_tokens=0),
            ),
        ),
        ContentBlockStartEvent(
            type="content_block_start",
            index=0,
            content_block=TextBlock(type="text", text=""),
        ),
        ContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=TextDelta(type="text_delta", text="Hello "),
        ),
        ContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=TextDelta(type="text_delta", text="world!"),
        ),
        MessageDeltaEvent(
            type="message_delta",
            delta=Delta(stop_reason="end_turn", stop_sequence=None),
            usage=MessageDeltaUsage(output_tokens=2),
        ),
        MessageStopEvent(type="message_stop"),
    ]

    mock_stream = _MockAsyncStream(events)
    mock_anthropic_client.messages.create = AsyncMock(return_value=mock_stream)

    provider = AnthropicLLMProvider(client=mock_anthropic_client, model="claude-3")
    messages = [user_message([{"type": "text", "text": "Hi"}])]
    stream = await provider.chat(messages=messages, tools=[])

    received = []
    async for chunk in stream:
        received.append(chunk)

    assert len(received) == 2
    assert received[0] == LLMChunkDelta(content="Hello ")
    assert received[1] == LLMChunkDelta(content="world!")

    response = stream.response
    assert response.content == "Hello world!"
    assert response.tool_calls == []
    assert response.finish_reason == "end_turn"
    assert response.usage == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }


@pytest.mark.asyncio
async def test_tool_call_streaming(mock_anthropic_client: MagicMock):
    """Provider accumulates streaming tool calls into ToolCall objects."""
    events = [
        MessageStartEvent(
            type="message_start",
            message=Message(
                id="msg_1",
                type="message",
                role="assistant",
                content=[],
                model="claude-3",
                stop_reason=None,
                stop_sequence=None,
                usage=Usage(input_tokens=5, output_tokens=0),
            ),
        ),
        ContentBlockStartEvent(
            type="content_block_start",
            index=0,
            content_block=ToolUseBlock(
                type="tool_use", id="tu_1", name="calc", input={}
            ),
        ),
        ContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=InputJSONDelta(type="input_json_delta", partial_json='{"a":'),
        ),
        ContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=InputJSONDelta(type="input_json_delta", partial_json=" 1}"),
        ),
        MessageDeltaEvent(
            type="message_delta",
            delta=Delta(stop_reason="tool_use", stop_sequence=None),
            usage=MessageDeltaUsage(output_tokens=4),
        ),
        MessageStopEvent(type="message_stop"),
    ]

    mock_stream = _MockAsyncStream(events)
    mock_anthropic_client.messages.create = AsyncMock(return_value=mock_stream)

    provider = AnthropicLLMProvider(client=mock_anthropic_client, model="claude-3")
    messages = [user_message([{"type": "text", "text": "Calculate"}])]
    tool = StreamingTool(
        name="calc",
        description="Calculate stuff",
        parameters={"type": "object", "properties": {}},
        fn=_noop_streaming_tool_fn,
    )
    stream = await provider.chat(messages=messages, tools=[tool])

    received = []
    async for chunk in stream:
        received.append(chunk)

    assert len(received) == 3
    assert received[0] == LLMChunkDelta(
        tool_calls=[ToolCallDelta(index=0, id="tu_1", name="calc")]
    )
    assert received[1] == LLMChunkDelta(
        tool_calls=[ToolCallDelta(index=0, arguments='{"a":')]
    )
    assert received[2] == LLMChunkDelta(
        tool_calls=[ToolCallDelta(index=0, arguments=" 1}")]
    )

    response = stream.response
    assert response.content is None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["id"] == "tu_1"
    assert response.tool_calls[0]["function"]["name"] == "calc"
    assert response.tool_calls[0]["function"]["arguments"] == '{"a": 1}'
    assert response.finish_reason == "tool_use"


@pytest.mark.asyncio
async def test_stop_event(mock_anthropic_client: MagicMock):
    """Provider respects the stop_event and breaks early."""
    events = [
        ContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=TextDelta(type="text_delta", text="One "),
        ),
        ContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=TextDelta(type="text_delta", text="two "),
        ),
        ContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=TextDelta(type="text_delta", text="three"),
        ),
    ]

    mock_stream = _MockAsyncStream(events)
    mock_anthropic_client.messages.create = AsyncMock(return_value=mock_stream)

    provider = AnthropicLLMProvider(client=mock_anthropic_client, model="claude-3")
    stop_event = asyncio.Event()
    stop_event.set()

    messages = [user_message([{"type": "text", "text": "Count"}])]
    stream = await provider.chat(messages=messages, tools=[], stop_event=stop_event)

    received = []
    async for chunk in stream:
        received.append(chunk)

    assert len(received) <= 1


@pytest.mark.asyncio
async def test_on_chunk_callback(mock_anthropic_client: MagicMock):
    """Provider invokes the on_chunk callback for every event."""
    events = [
        MessageStartEvent(
            type="message_start",
            message=Message(
                id="msg_1",
                type="message",
                role="assistant",
                content=[],
                model="claude-3",
                stop_reason=None,
                stop_sequence=None,
                usage=Usage(input_tokens=2, output_tokens=0),
            ),
        ),
        MessageStopEvent(type="message_stop"),
    ]

    mock_stream = _MockAsyncStream(events)
    mock_anthropic_client.messages.create = AsyncMock(return_value=mock_stream)

    callback_calls = []

    async def on_chunk(chunk, is_done):
        callback_calls.append((chunk, is_done))

    provider = AnthropicLLMProvider(
        client=mock_anthropic_client, model="claude-3", on_chunk=on_chunk
    )
    messages = [user_message([{"type": "text", "text": "Hi"}])]
    stream = await provider.chat(messages=messages, tools=[])

    async for _ in stream:
        pass

    # MessageStart and MessageStop produce no deltas, so only the final done call
    assert len(callback_calls) == 1
    assert callback_calls[-1] == (None, True)


@pytest.mark.asyncio
async def test_system_prompt_passed_separately(mock_anthropic_client: MagicMock):
    """The system prompt is passed as a top-level kwarg, not in messages."""
    mock_stream = _MockAsyncStream([MessageStopEvent(type="message_stop")])
    mock_anthropic_client.messages.create = AsyncMock(return_value=mock_stream)

    provider = AnthropicLLMProvider(client=mock_anthropic_client, model="claude-3")
    messages = [
        system_message("Be concise."),
        user_message([{"type": "text", "text": "Hi"}]),
    ]
    stream = await provider.chat(messages=messages, tools=[])
    async for _ in stream:
        pass

    call_kwargs = mock_anthropic_client.messages.create.call_args.kwargs
    assert call_kwargs["system"] == "Be concise."
    assert all(m["role"] != "system" for m in call_kwargs["messages"])
