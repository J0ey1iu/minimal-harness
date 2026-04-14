from .openai import OpenAIAgent
from .protocol import Agent, InputContentConversionFunction

__all__ = [
    "Agent",
    "InputContentConversionFunction",
    "OpenAIAgent",
]


def __getattr__(name: str):
    if name == "LiteLLMAgent":
        from .litellm import LiteLLMAgent

        return LiteLLMAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
