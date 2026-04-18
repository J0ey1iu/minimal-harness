from .agent import OpenAIAgent
from .llm import LLMProvider, LLMResponse, OpenAILLMProvider, Stream
from .memory import (
    ConversationMemory,
    InputContentPart,
    Memory,
    TextContentPart,
)
from .tool import AgenticTool, BaseTool, Tool
from .tool_executor import ToolExecutor

__ALL__ = [
    OpenAIAgent,
    LLMProvider,
    LLMResponse,
    Stream,
    Memory,
    ConversationMemory,
    OpenAILLMProvider,
    Tool,
    BaseTool,
    AgenticTool,
    ToolExecutor,
    InputContentPart,
    TextContentPart,
]


def __getattr__(name: str):
    if name == "LiteLLMAgent":
        from .agent.litellm import LiteLLMAgent

        return LiteLLMAgent
    if name == "LiteLLMProvider":
        from .llm.litellm import LiteLLMProvider

        return LiteLLMProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
