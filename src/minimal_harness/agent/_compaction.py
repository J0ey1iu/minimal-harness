from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, AsyncIterator

from minimal_harness.memory import Message, sanitize_tool_calls

if TYPE_CHECKING:
    from minimal_harness.llm.llm import LLMProvider


DEFAULT_SUMMARY_REQUEST = (
    "Please produce a single, dense summary of the conversation above.\n"
    "\n"
    "Preserve, in this exact order, with these headings:\n"
    "  Goals — the user's original goal and any updated goals. When the\n"
    "    user shifts focus, the latest goal wins; record both if useful.\n"
    "  Decisions & Outcomes — concrete facts, choices, and results reached.\n"
    "  Open Questions / Pending — anything unresolved or waiting on\n"
    "    input/action; the next concrete step if known.\n"
    "  Entities & IDs — file paths, function/class names, identifiers,\n"
    "    quoted verbatim from the transcript.\n"
    "\n"
    "Rules:\n"
    "  - Output only these four sections, in this order, with these exact\n"
    "    headings. No preamble, no closing remarks, no labels.\n"
    "  - When a later message contradicts an earlier one, the later wins;\n"
    "    record both points with their relative position if it matters.\n"
    "  - Do not invent facts. If a section has nothing to record, write\n"
    '    "(none)" under the heading — do not omit the heading.\n'
    "  - Keep the summary dense. Drop pleasantries, hedging, redundant\n"
    "    clarifications, and any content the user can re-derive from\n"
    "    already-stated facts.\n"
    "  - Do not narrate the assistant's reasoning chain or quote\n"
    "    tool/function call JSON.\n"
    "\n"
    "If a prior summary is present in the conversation above (as an\n"
    "earlier assistant turn), fold its content into the new summary\n"
    "using the same four-heading structure. Drop detail that is no\n"
    "longer relevant; preserve anything still needed to continue the\n"
    "user's task.\n"
    "\n"
    "The summary will replace the conversation above for future LLM\n"
    "calls, so any information the assistant needs to continue the\n"
    "user's task must appear in it."
)


def _resolve_localised_prompt(
    base_prompt: str | None,
    locale_json: str | dict | None,
    locale: str,
) -> str | None:
    """Resolve the compaction prompt with locale awareness.

    If ``locale_json`` is a valid JSON dict and ``locale`` is present
    as a key, return the locale-specific version.  Otherwise fall back
    to ``base_prompt``.

    ``locale_json`` can be:
    * a JSON string (e.g. ``'{"zh":"...","en":"..."}'``)
    * a Python dict (e.g. ``{"zh":"...","en":"..."}``)
    * ``None``
    """
    if locale and locale_json is not None and locale_json != "":
        parsed: dict | None = None
        if isinstance(locale_json, dict):
            parsed = locale_json
        elif isinstance(locale_json, str):
            try:
                parsed = json.loads(locale_json)
            except (json.JSONDecodeError, TypeError):
                pass
        if isinstance(parsed, dict) and locale in parsed:
            val = parsed[locale]
            if isinstance(val, str) and val.strip():
                return val
    return base_prompt if base_prompt else None


def build_chat_payload(
    system_prompt: str,
    messages: list[Message],
    existing_summary: str | None,
    summary_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Compose the chat payload sent to the LLM for summarization.

    ``summary_prompt`` is an optional user-customisable summarization
    instruction that replaces the built-in ``DEFAULT_SUMMARY_REQUEST``
    when provided.  Pass ``None`` to keep the default.
    """
    chat: list[dict[str, Any]] = []
    if system_prompt:
        chat.append({"role": "system", "content": system_prompt})
    if existing_summary:
        chat.append({"role": "assistant", "content": existing_summary})
    for m in messages:
        role = m.get("role")
        if role == "compaction":
            chat.append({"role": "assistant", "content": str(m.get("content", ""))})
        else:
            # ``id`` is a session-identity key, not part of the LLM wire
            # format — strip it before it reaches the summarizer.
            chat.append({k: v for k, v in m.items() if k != "id"})

    # Strip tool_calls from assistant messages without a following tool
    # response — the LLM API rejects dangling calls (InferHub 2013).
    # 1. Drop calls with truncated arguments (broken/stopped stream).
    for m in chat:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            m["tool_calls"] = sanitize_tool_calls(m["tool_calls"])
    # 2. Drop calls that are never answered by a tool message (mirror of
    #    the same healing in ``Memory.get_forward_messages``): an assistant
    #    tool_call must be followed by its result before any non-tool
    #    message, otherwise the payload is rejected.
    pending: dict[str, int] = {}  # call id -> index in chat
    for i, m in enumerate(chat):
        role = m.get("role")
        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                if tc.get("id"):
                    pending[tc["id"]] = i
        elif role == "tool":
            _tid = m.get("tool_call_id")
            if _tid:
                pending.pop(_tid, None)
        elif pending:
            for _tid, _idx in pending.items():
                _calls = chat[_idx].get("tool_calls") or []
                chat[_idx]["tool_calls"] = [
                    tc for tc in _calls if tc.get("id") != _tid
                ] or None
            pending.clear()
    if pending:  # buffer ends with unanswered calls
        for _tid, _idx in pending.items():
            _calls = chat[_idx].get("tool_calls") or []
            chat[_idx]["tool_calls"] = [
                tc for tc in _calls if tc.get("id") != _tid
            ] or None
    # Drop tool messages whose assistant call was dropped (e.g. truncated
    # arguments) — the API rejects tool messages referencing an
    # undeclared tool_call_id.
    visible_tool_call_ids = {
        tc["id"]
        for m in chat
        if m.get("role") == "assistant" and m.get("tool_calls")
        for tc in m["tool_calls"]
        if tc.get("id")
    }
    chat = [
        m
        for m in chat
        if not (
            m.get("role") == "tool"
            and m.get("tool_call_id")
            and m["tool_call_id"] not in visible_tool_call_ids
        )
    ]
    # After stripping tool_calls, remove assistant messages that now
    # have neither content nor tool_calls (LLM API rejects them).
    chat = [
        m
        for m in chat
        if not (
            m.get("role") == "assistant"
            and not m.get("content")
            and not m.get("tool_calls")
        )
    ]

    # Use user-provided summary prompt if given, otherwise fall back to default.
    effective_prompt = summary_prompt if summary_prompt else DEFAULT_SUMMARY_REQUEST
    chat.append({"role": "user", "content": effective_prompt})
    return chat


def build_summarizer(
    llm_provider: "LLMProvider",
    system_prompt: str,
    system_prompt_locale: dict[str, str] | None = None,
    summary_prompt: str | None = None,
    summary_prompt_locale: str | None = None,
):
    """Build a streaming summarizer callback bound to ``llm_provider``.

    ``summary_prompt`` is an optional user-customisable instruction
    that replaces the built-in ``DEFAULT_SUMMARY_REQUEST``.  Pass
    ``None`` to keep the default.

    ``summary_prompt_locale`` is an optional JSON dict (e.g.
    ``{"zh": "...", "en": "..."}``) providing locale-specific
    overrides for ``summary_prompt``.  At call time the current
    locale (from the agent run context) is used to pick the right
    version, falling back to ``summary_prompt`` if no match.

    ``system_prompt_locale`` is an optional dict providing locale-specific
    overrides for ``system_prompt``.  At call time the current locale is
    used to resolve the system prompt, the same way the agent loop does
    via ``AgentMetadata.resolve_system_prompt(locale)``.
    """

    async def _summarize(
        messages: list[Message],
        existing_summary: str | None,
    ) -> AsyncIterator[str]:
        # Resolve locale-aware system prompt and compaction prompt at call time.
        from minimal_harness.agent.runtime import get_current_locale

        locale = get_current_locale()
        effective_prompt = _resolve_localised_prompt(
            summary_prompt, summary_prompt_locale, locale
        )
        # Resolve system_prompt with locale awareness, just like
        # AgentMetadata.resolve_system_prompt() does at run time.
        resolved_system_prompt = system_prompt
        if locale and system_prompt_locale and locale in system_prompt_locale:
            resolved_system_prompt = system_prompt_locale[locale]
        payload = build_chat_payload(
            resolved_system_prompt,
            messages,
            existing_summary,
            summary_prompt=effective_prompt,
        )
        response = await llm_provider.chat(messages=payload, tools=[])  # type: ignore[arg-type]
        # ``Stream.__anext__`` 内部吞掉末位的 ``LLMResponse``（只存到
        # ``.response``，不 yield），所以这里遍历拿到的是逐段 delta。
        # 最后再 yield 一次全文会把摘要翻倍 —— 只在没有任何 delta 产出时
        # （非流式 provider）才用 ``final.content`` 兑底。
        streamed = False
        async for chunk in response:
            if chunk.content:
                streamed = True
                yield chunk.content
        final = response.response
        if final.content and not streamed:
            yield final.content

    return _summarize
