from __future__ import annotations

import logging


class CorrelationFilter(logging.Filter):
    """Automatically prepend ``[corr=<id>]`` to every log message

    when a ``correlation_id`` exists in the active run context.
    Install on any logger whose output you want correlated:
    ``addFilter(CorrelationFilter())``.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            from minimal_harness.agent.runtime import _current_context

            ctx = _current_context.get()
            cid = ctx.get("correlation_id", "")
            if cid:
                prefix = f"[corr={cid}] "
                if not record.msg.startswith(prefix):
                    record.msg = f"{prefix}{record.msg}"
        except Exception:
            pass
        return True
