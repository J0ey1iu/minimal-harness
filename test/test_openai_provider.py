"""Unit tests for the OpenAI-compatible LLM provider."""

import asyncio
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from minimal_harness.llm.openai import OpenAILLMProvider, _convert_messages
from minimal_harness.memory import (
    system_message,
    user_message,
)
from minimal_harness.tool.base import StreamingTool
from minimal_harness.types import LLMChunkDelta, ToolCallDelta
from openai.types.chat import (
    ChatCompletionChunk,
)
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)


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


def _chunk(
    content: str | None = None,
    tool_calls: list[ChoiceDeltaToolCall] | None = None,
    finish_reason: Any = None,
    usage: dict | None = None,
) -> ChatCompletionChunk:
    delta = ChoiceDelta(
        content=content,
        tool_calls=tool_calls,
    )
    choice = Choice(delta=delta, finish_reason=finish_reason, index=0)
    chunk = ChatCompletionChunk(
        id="test-id",
        choices=[choice],
        created=0,
        model="gpt-4",
        object="chat.completion.chunk",
    )
    if usage is not None:
        # Usage is set via private attribute to bypass validation
        object.__setattr__(chunk, "usage", usage)
    return chunk


async def _noop_streaming_tool_fn(**kwargs: object) -> AsyncIterator[dict]:
    yield {"result": "ok"}


@pytest.fixture
def mock_openai_client():
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    return client


@pytest.mark.asyncio
async def test_text_streaming(mock_openai_client: MagicMock):
    """Provider yields text chunks and ends with an LLMResponse."""
    chunks = [
        _chunk(content="Hello "),
        _chunk(content="world!"),
        _chunk(finish_reason="stop"),
    ]

    mock_stream = _MockAsyncStream(chunks)
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_stream)

    provider = OpenAILLMProvider(client=mock_openai_client, model="gpt-4")
    messages = [
        system_message("You are helpful."),
        user_message([{"type": "text", "text": "Hi"}]),
    ]
    stream = await provider.chat(messages=messages, tools=[])

    received_chunks = []
    async for chunk in stream:
        received_chunks.append(chunk)

    assert len(received_chunks) == 2
    assert received_chunks[0] == LLMChunkDelta(content="Hello ")
    assert received_chunks[1] == LLMChunkDelta(content="world!")

    response = stream.response
    assert response.content == "Hello world!"
    assert response.tool_calls == []
    assert response.finish_reason == "stop"


@pytest.mark.asyncio
async def test_tool_call_streaming(mock_openai_client: MagicMock):
    """Provider accumulates streaming tool calls into ToolCall objects."""
    chunks = [
        _chunk(
            tool_calls=[
                ChoiceDeltaToolCall(
                    index=0,
                    id="call_1",
                    type="function",
                    function=ChoiceDeltaToolCallFunction(name="calc", arguments=""),
                )
            ]
        ),
        _chunk(
            tool_calls=[
                ChoiceDeltaToolCall(
                    index=0,
                    function=ChoiceDeltaToolCallFunction(arguments='{"a":'),
                )
            ]
        ),
        _chunk(
            tool_calls=[
                ChoiceDeltaToolCall(
                    index=0,
                    function=ChoiceDeltaToolCallFunction(arguments=" 1}"),
                )
            ]
        ),
        _chunk(finish_reason="tool_calls"),
    ]

    mock_stream = _MockAsyncStream(chunks)
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_stream)

    provider = OpenAILLMProvider(client=mock_openai_client, model="gpt-4")
    messages = [user_message([{"type": "text", "text": "Calculate"}])]
    tool = StreamingTool(
        name="calc",
        description="Calculate stuff",
        parameters={"type": "object", "properties": {}},
        fn=_noop_streaming_tool_fn,
    )
    stream = await provider.chat(messages=messages, tools=[tool])

    received_chunks = []
    async for chunk in stream:
        received_chunks.append(chunk)

    assert len(received_chunks) == 3
    assert received_chunks[0] == LLMChunkDelta(
        tool_calls=[ToolCallDelta(index=0, id="call_1", name="calc")]
    )
    assert received_chunks[1] == LLMChunkDelta(
        tool_calls=[ToolCallDelta(index=0, arguments='{"a":')]
    )
    assert received_chunks[2] == LLMChunkDelta(
        tool_calls=[ToolCallDelta(index=0, arguments=" 1}")]
    )

    response = stream.response
    assert response.content is None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["id"] == "call_1"
    assert response.tool_calls[0]["function"]["name"] == "calc"
    assert response.tool_calls[0]["function"]["arguments"] == '{"a": 1}'
    assert response.finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_stop_event(mock_openai_client: MagicMock):
    """Provider respects the stop_event and breaks early."""
    chunks = [
        _chunk(content="One "),
        _chunk(content="two "),
        _chunk(content="three"),
    ]

    mock_stream = _MockAsyncStream(chunks)
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_stream)

    provider = OpenAILLMProvider(client=mock_openai_client, model="gpt-4")
    stop_event = asyncio.Event()
    stop_event.set()

    messages = [user_message([{"type": "text", "text": "Count"}])]
    stream = await provider.chat(messages=messages, tools=[], stop_event=stop_event)

    received: list = []
    try:
        async for chunk in stream:
            received.append(chunk)
    except asyncio.CancelledError:
        pass

    assert len(received) == 0


@pytest.mark.asyncio
async def test_usage_tracking(mock_openai_client: MagicMock):
    """Provider extracts usage information from the final chunk."""
    usage_chunk = _chunk(finish_reason="stop")
    # Inject usage manually
    object.__setattr__(
        usage_chunk,
        "usage",
        MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    chunks = [
        _chunk(content="Done"),
        usage_chunk,
    ]

    mock_stream = _MockAsyncStream(chunks)
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_stream)

    provider = OpenAILLMProvider(client=mock_openai_client, model="gpt-4")
    messages = [user_message([{"type": "text", "text": "Hi"}])]
    stream = await provider.chat(messages=messages, tools=[])

    async for _ in stream:
        pass

    assert stream.response.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }


@pytest.mark.asyncio
async def test_stream_yields_chunks(mock_openai_client: MagicMock):
    """Provider stream yields LLMChunkDelta chunks."""
    chunks = [
        _chunk(content="A"),
        _chunk(content="B"),
    ]
    mock_openai_client.chat.completions.create = AsyncMock(
        return_value=_MockAsyncStream(chunks)
    )

    provider = OpenAILLMProvider(client=mock_openai_client, model="gpt-4")
    messages = [user_message([{"type": "text", "text": "Hi"}])]
    stream = await provider.chat(messages=messages, tools=[])

    deltas: list[LLMChunkDelta] = []
    async for chunk in stream:
        deltas.append(chunk)

    assert any(d.content == "A" for d in deltas)
    assert any(d.content == "B" for d in deltas)


def test_user_message_with_raw_string_is_wrapped_as_text_part() -> None:
    """Regression test: callers (e.g. the compaction summarizer in
    mh-tui) sometimes pass a plain string for ``user.content`` instead
    of the typed ``list[InputContentPart]`` shape. ``_convert_messages``
    used to crash with ``TypeError: string indices must be integers``
    because it iterated the string's characters. The defensive
    handling wraps a raw string as a single text part so the
    downstream call survives and produces the same OpenAI payload
    the typed shape would have produced.
    """
    raw_string_user_msg: dict[str, Any] = {
        "role": "user",
        "content": "Just a plain string, not a list of parts.",
    }
    converted = _convert_messages([raw_string_user_msg])
    assert len(converted) == 1
    assert converted[0]["role"] == "user"
    assert converted[0]["content"] == [
        {"type": "text", "text": "Just a plain string, not a list of parts."}
    ]


def test_typed_user_message_still_works() -> None:
    """The defensive wrap must not regress the typed-input path."""
    typed_msg = user_message([{"type": "text", "text": "typed"}])
    converted = _convert_messages([typed_msg])
    assert converted[0]["content"] == [{"type": "text", "text": "typed"}]


def test_user_file_message_projects_plain_text_with_id() -> None:
    """File parts must reach the model as plain text carrying the file_id,
    so text-only models can address attachments via attachment tools."""
    msg = user_message(
        [
            {
                "type": "file",
                "file": {
                    "file_id": "att-9",
                    "file_name": "report.docx",
                    "file_size": 1024,
                    "backend_type": "attachment",
                },
            }
        ]
    )
    converted = _convert_messages([msg])
    assert converted[0]["content"] == [
        {"type": "text", "text": "[File: report.docx (id=att-9)]"}
    ]
