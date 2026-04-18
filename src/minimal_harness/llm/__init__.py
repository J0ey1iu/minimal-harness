from minimal_harness.types import (
    TokenUsage,
    ToolCall,
    ToolCallFunction,
    ToolResultCallback,
)

from .llm import ChunkCallback, LLMProvider, LLMResponse, Stream
from .openai import OpenAILLMProvider

__ALL__ = [
    ChunkCallback,
    LLMProvider,
    LLMResponse,
    Stream,
    TokenUsage,
    ToolCall,
    ToolCallFunction,
    ToolResultCallback,
    OpenAILLMProvider,
]


def __getattr__(name: str):
    if name == "LiteLLMProvider":
        from .litellm import LiteLLMProvider

        return LiteLLMProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
