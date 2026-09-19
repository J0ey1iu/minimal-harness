from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, AsyncIterator

from minimal_harness.memory import Message, sanitize_tool_calls

if TYPE_CHECKING:
    from minimal_harness.llm.llm import LLMProvider


# ``compaction_prompt="restated-goal"`` resolves to this preset (RFC #57,
# mhc-desktop). It is the production-verified Restated-goal format: four
# fixed sections that keep the user's ORIGINAL goal and load-bearing
# constraints alive across every fold. Deliberately language-neutral.
#
# To migrate: set ``compaction_prompt: "restated-goal"`` in the agent's
# CompactionSettings (or ToolCompactionSettings) — no need to copy the
# prompt text into your config.
COMPACTION_SUMMARY_PROMPT = (
    "Produce a single, dense summary of the conversation above. The summary "
    "will replace the conversation for future LLM calls, so any information "
    "the assistant needs to continue the user's task must appear in it.\n"
    "\n"
    "Preserve, in this exact order, with these headings:\n"
    "  Goals — the user's overarching objective AND any constraints that bound "
    "how it must be achieved. Preserve the earliest clear statement of the "
    "user's goal (the ORIGINAL goal), even after later goal shifts: record "
    "each shift, but do not drop the origin. Constraints (libraries, target "
    "platform, APIs that must not break, perf / compliance budgets, 'must' / "
    "'must not' rules) are facts about the task's boundaries; treat them as "
    "load-bearing and never fold them into 'obvious context'.\n"
    "  Decisions & Outcomes — concrete facts, choices, and results reached. "
    "If a Decisions entry references a specific library, platform, or API, "
    "the constraint that selected it MUST also appear under Goals — record "
    "it there if it is not already.\n"
    "  Open Questions / Pending — anything unresolved or waiting on input or "
    "action; the next concrete step if known.\n"
    "  Entities & IDs — file paths, function/class names, identifiers, quoted "
    "verbatim from the transcript.\n"
    "\n"
    "Rules:\n"
    "  - Output only these four sections, in this order, with these exact "
    "headings. No preamble, no labels. No closing remarks other than the "
    "Restated goal line below.\n"
    "  - When a later message contradicts an earlier one, the later wins; "
    "record both points with their relative position if it matters.\n"
    "  - Do not invent facts. If a section has nothing to record, write "
    "'(none)' under the heading — do not omit the heading.\n"
    "  - Keep the summary dense. Drop pleasantries, hedging, redundant "
    "clarifications, and any content the user can re-derive from "
    "already-stated facts in the transcript or other sections of this summary. "
    "The user's original goal, stated constraints, and any 'must' / 'must not' "
    "are NEVER eligible for dropping under any rule in this prompt. They must "
    "survive every fold.\n"
    "  - Length budget: aim for roughly 25–40% of the original transcript's "
    "token count. If you cannot fit everything within that, compress "
    "Decisions (especially tool outputs and superseded decisions) and "
    "Entities first. NEVER compress the Goals section or the Restated goal "
    "line to fit the budget.\n"
    "  - Do not narrate the assistant's reasoning chain or quote tool/function "
    "call JSON verbatim unless it affects a later turn.\n"
    "\n"
    "If a prior summary is present in the conversation above (as an earlier "
    "assistant turn), fold its content into the new summary using the same "
    "four-heading structure. Drop detail that is no longer relevant; preserve "
    "anything still needed to continue the user's task. The original goal and "
    "stated constraints survive every fold.\n"
    "\n"
    "End the summary with exactly one line in this form:\n"
    "\n"
    "  Restated goal: <one sentence restating the user's ACTIVE goal — the "
    "original goal as updated by all stated shifts and constraints — in the "
    "model's own words>"
)


# Deprecated: the built-in default summarization instruction. It predates
# the RFC #57 Restated-goal preset and is kept as the default for backward
# compatibility — existing consumers rely on its exact five-section output
# shape. New deployments should prefer ``compaction_prompt="restated-goal"``;
# a future major release may switch the default to
# :data:`COMPACTION_SUMMARY_PROMPT` or drop this constant.
DEFAULT_SUMMARY_REQUEST = (
    "Please produce a single, dense summary of the conversation above.\n"
    "\n"
    "Preserve, in this exact order, with these headings:\n"
    "  Goals — the user's overarching objective and why it matters, plus\n"
    "    any updates. Preserve the user's ORIGINAL goal verbatim or\n"
    "    near-verbatim even after later goal shifts: record the shift,\n"
    "    but never drop the origin. When a later decision only makes\n"
    "    sense given the original goal, retain enough of it to\n"
    "    interpret the decision.\n"
    "  Constraints & Non-negotiables — libraries/frameworks, target\n"
    "    platform, APIs that must not break, performance/compliance\n"
    '    budgets, and any "must / must not" the user has stated.\n'
    "    These belong here and only here.\n"
    "  Decisions & Outcomes — concrete facts, choices, and results reached.\n"
    "  Open Questions / Pending — anything unresolved or waiting on\n"
    "    input/action; the next concrete step if known.\n"
    "  Entities & IDs — file paths, function/class names, identifiers,\n"
    "    quoted verbatim from the transcript.\n"
    "\n"
    "Rules:\n"
    "  - Output only these five sections, in this order, with these exact\n"
    "    headings. No preamble, no closing remarks, no labels.\n"
    "  - When a later message contradicts an earlier one, the later wins;\n"
    "    record both points with their relative position if it matters.\n"
    "  - Do not invent facts. If a section has nothing to record, write\n"
    '    "(none)" under the heading — do not omit the heading.\n'
    "  - Keep the summary dense: drop pleasantries, hedging, and redundant\n"
    "    clarifications. Never drop the user's original goal, stated\n"
    "    constraints, or non-negotiables under any rule in this prompt;\n"
    "    the turns that stated them are deleted after summarization, so\n"
    "    this summary is the only place they can survive.\n"
    "  - Do not narrate the assistant's reasoning chain or quote\n"
    "    tool/function call JSON.\n"
    "  - End the summary with this exact closing line:\n"
    "    Restated goal: <one sentence restating the user's current overall\n"
    "    objective in your own words>\n"
    "\n"
    "If a prior summary is present in the conversation above (as an\n"
    "earlier assistant turn), fold its content into the new summary\n"
    "using the same five-heading structure. Drop detail that is no\n"
    "longer relevant, but preserve anything still needed to continue\n"
    "the user's task — including the original goal and constraints,\n"
    "which must survive every fold.\n"
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


def resolve_compaction_prompt(prompt: str | None) -> str | None:
    """Resolve named presets to their full prompt text.

    Currently knows one preset: ``"restated-goal"`` (RFC #57) which
    expands to :data:`COMPACTION_SUMMARY_PROMPT`. Any other value is
    passed through verbatim (a custom instruction). ``None`` keeps the
    built-in default.
    """
    if prompt == "restated-goal":
        return COMPACTION_SUMMARY_PROMPT
    return prompt


def estimate_prompt_tokens(memory) -> int:
    """Cheap byte→token leading-edge estimate of the LLM-visible buffer.

    byte//4 under-counts ASCII (~4 bytes/token) and CJK (~3 bytes/token);
    both err toward triggering LATE — the safe side for a leading edge,
    since post-hoc provider usage is the accurate backstop (RFC #57).
    """
    total = 0
    for m in memory.get_forward_messages():
        c = m.get("content")
        if isinstance(c, str):
            total += len(c.encode("utf-8", "replace"))
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict):
                    t = part.get("text") or part.get("content")
                    if isinstance(t, str):
                        total += len(t.encode("utf-8", "replace"))
    return total // 4


def compute_effective_keep_recent(
    memory,
    keep_recent: int,
    anchor: str = "tail",
) -> int:
    """Compute the tail-count floor for ``Memory.compact`` from an anchor.

    Anchor semantics (RFC #57):
    - ``"tail"`` — today's static-count behaviour (default, backward compat).
    - ``"last_user"`` — never fold the latest user turn or its reply.
    - ``"last_tool_round"`` — never fold the most recent
      assistant-with-``tool_calls`` round (a live tool round must survive
      the fold or the follow-up turn sees only a summary and replies
      "I understand" instead of continuing). Plain-chat sessions without
      any tool round fall back to the last user message.

    ``keep_recent=0`` therefore means "fold everything except the anchor",
    never "fold everything" under the non-``"tail"`` anchors.
    """
    if anchor == "tail":
        return keep_recent
    msgs = memory.get_all_messages()
    n = len(msgs)
    last_user_idx = -1
    for i in range(n - 1, -1, -1):
        if msgs[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx < 0:
        return keep_recent
    keep_start = last_user_idx
    if anchor == "last_tool_round":
        for i in range(n - 1, -1, -1):
            if msgs[i].get("tool_calls"):
                keep_start = i
                break
    return max(keep_recent, n - keep_start)


def should_fold(
    memory,
    threshold: int,
    *,
    soft_limit_ratio: float = 0.0,
    max_context_tokens: int = 0,
    estimate_leading_edge: bool = True,
) -> tuple[bool, int]:
    """Decide whether to fold, and with what token count for the start event.

    Post-hoc check (existing behaviour): provider-reported cumulative usage
    above ``threshold`` folds. RFC #57 leading edge: when ``soft_limit_ratio``
    and ``max_context_tokens`` are both positive, an estimated prompt size
    above ``int(max_context_tokens * ratio)`` ALSO folds — this fires before
    a request can 400-loop on ``context_length_exceeded``, which post-hoc
    usage can never catch. OR semantics, exactly as the proposers specify.
    """
    cumulative = int(memory.get_message_usage().get("total_tokens", 0)) or 0
    if cumulative > threshold:
        return True, cumulative
    if soft_limit_ratio > 0 and max_context_tokens > 0:
        soft_threshold = int(max_context_tokens * soft_limit_ratio)
        if estimate_leading_edge and estimate_prompt_tokens(memory) > soft_threshold:
            return True, cumulative
        if cumulative > soft_threshold:
            return True, cumulative
    return False, cumulative


def soft_limit_threshold(
    soft_limit_ratio: float,
    max_context_tokens: int,
) -> int:
    """The RFC #57 soft-limit trigger bound, or 0 when disabled."""
    if soft_limit_ratio > 0 and max_context_tokens > 0:
        return int(max_context_tokens * soft_limit_ratio)
    return 0


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
    effective_prompt = (
        resolve_compaction_prompt(summary_prompt)
        if summary_prompt
        else DEFAULT_SUMMARY_REQUEST
    )
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
        # Resolve the locale-aware, preset-expanded compaction prompt.
        effective_prompt = _resolve_localised_prompt(
            summary_prompt, summary_prompt_locale, locale
        )
        effective_prompt = resolve_compaction_prompt(effective_prompt)
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
