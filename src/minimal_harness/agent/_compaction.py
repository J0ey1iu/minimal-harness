from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator

from minimal_harness.memory import Message

if TYPE_CHECKING:
    from minimal_harness.llm.llm import LLMProvider


SUMMARY_REQUEST = (
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


def _project_history(
    messages: list[Message],
    existing_summary: str | None,
) -> list[dict[str, Any]]:
    """Project a memory slice into chat messages.

    Re-projects ``role="compaction"`` to ``role="assistant"`` (matches
    :meth:`Memory.get_forward_messages`). If ``existing_summary`` is
    provided it is prepended as an assistant turn, mirroring how prior
    compaction summaries are exposed to the LLM in the agent loop.
    """
    chat: list[dict[str, Any]] = []
    if existing_summary:
        chat.append({"role": "assistant", "content": existing_summary})
    for m in messages:
        role = m.get("role")
        if role == "compaction":
            chat.append(
                {
                    "role": "assistant",
                    "content": str(m.get("content", "")),
                }
            )
        else:
            chat.append(dict(m))

    # Strip tool_calls from any assistant message that does not have
    # a following tool response — the LLM API rejects dangling calls.
    for i in range(len(chat) - 1):
        if chat[i].get("role") == "assistant" and chat[i].get("tool_calls"):
            if chat[i + 1].get("role") != "tool":
                chat[i].pop("tool_calls", None)
    # Also check the very last message
    if chat and chat[-1].get("role") == "assistant" and chat[-1].get("tool_calls"):
        chat[-1].pop("tool_calls", None)

    return chat


def build_chat_payload(
    system_prompt: str,
    messages: list[Message],
    existing_summary: str | None,
) -> list[dict[str, Any]]:
    """Compose the chat payload sent to the LLM for summarization.

    The payload mirrors what the agent loop sees — same system prompt,
    same conversation history — with a trailing user message that
    instructs the model to produce a summary.
    """
    chat: list[dict[str, Any]] = []
    if system_prompt:
        chat.append({"role": "system", "content": system_prompt})
    chat.extend(_project_history(messages, existing_summary))
    chat.append({"role": "user", "content": SUMMARY_REQUEST})
    return chat


async def stream_summary(
    llm_provider: "LLMProvider",
    system_prompt: str,
    messages: list[Message],
    existing_summary: str | None,
) -> AsyncIterator[str]:
    """Stream summary content chunks for the given memory slice."""
    payload = build_chat_payload(system_prompt, messages, existing_summary)
    response = await llm_provider.chat(messages=payload, tools=[])  # type: ignore[arg-type]
    async for chunk in response:
        if chunk.content:
            yield chunk.content
    final = response.response
    if final.content:
        yield final.content


def build_summarizer(
    llm_provider: "LLMProvider",
    system_prompt: str,
):
    """Build a streaming summarizer callback bound to ``llm_provider``.

    The callback signature matches :class:`CompactionSummarizer`, so it
    can be plugged straight into :class:`CompactionConfig`.
    """

    async def _summarize(
        messages: list[Message],
        existing_summary: str | None,
    ) -> AsyncIterator[str]:
        async for chunk in stream_summary(
            llm_provider, system_prompt, messages, existing_summary
        ):
            yield chunk

    return _summarize
