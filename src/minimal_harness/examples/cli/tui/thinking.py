from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


def extract_thinking(delta: Any) -> str | None:
    reasoning = getattr(delta, "reasoning_content", None)
    if reasoning:
        return reasoning

    reasoning = getattr(delta, "reasoning", None)
    if reasoning:
        return reasoning

    provider_fields = getattr(delta, "provider_specific_fields", None)
    if provider_fields and isinstance(provider_fields, dict):
        reasoning = provider_fields.get("reasoning_content")
        if reasoning:
            return reasoning

    model_extra = getattr(delta, "model_extra", None)
    if model_extra and isinstance(model_extra, dict):
        reasoning = model_extra.get("reasoning_content")
        if reasoning:
            return reasoning
        reasoning = model_extra.get("reasoning")
        if reasoning:
            return reasoning

    return None
