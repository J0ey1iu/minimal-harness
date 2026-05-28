# Agent coding guide

## Must do

1. When you need to start the project in any way, use python interpreter in ./.venv/bin/python.
2. Run `uv run ruff check --fix`, `uv run ruff check --select I --fix` and `uv run pyright .` after finished code editing. And fix the errors and warning found by these wonderful linters.
3. Run `uv run ruff format` to format the file for standard convention formatting.

## Logging conventions

All log messages within an agent run context (inside `AgentRuntime._run()` or
`SSEAgentRunner.run()`) use this format:

    <action> key=val key=val ...

The ``[corr=<id>]`` prefix is injected automatically by `CorrelationFilter`
— never include it manually.

Level guidelines:

| Level     | When to use                                 | Example                          |
|-----------|---------------------------------------------|----------------------------------|
| ``INFO``  | Lifecycle events (start/end, LLM call, tool)| ``agent.run.start agent=abc``    |
| ``DEBUG`` | Per-event noise (SSE protocol, every chunk) | ``tool.sse.missing_type url=…``  |
| ``WARNING``| Recoverable failures (HTTP 4xx/5xx)        | ``tool.http.error name=…``      |
| ``ERROR`` | Operational failures                        | ``tool.subprocess.error name=…`` |

Message format rules:
1. Action first, dot-separated: ``agent.run.start``, ``llm.chat``, ``tool.start``
2. Key=value pairs after, space-separated: ``model=gpt-4 msgs=5 tools=2``
3. No prefixes like ``[AgentRuntime]`` or ``INBOUND/OUTBOUND`` — use the
   ``<action>`` alone to convey semantics

## Must NOT do

1. Don't ever commit anything without a user asking you to.
