from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

from minimal_harness.agent.simple import SimpleAgent
from minimal_harness.memory import ConversationMemory
from minimal_harness.tool.factory import DefaultToolFactory
from minimal_harness.types import AgentEnd

if TYPE_CHECKING:
    from minimal_harness.agent.registry import AgentRegistryProtocol
    from minimal_harness.llm.llm import LLMProvider
    from minimal_harness.tool.base import Tool
    from minimal_harness.tool.registry import ToolRegistryProtocol
    from minimal_harness.types import AgentMetadata

from .collector import EvalCollector
from .persistence import EvalPersistence
from .report import generate_html_report
from .types import EvalRunRecord, EvalSummary, EvalTaskConfig


async def run_evaluation(
    *,
    agent_registry: AgentRegistryProtocol,
    tool_registry: ToolRegistryProtocol,
    llm_provider_factory: Callable[[], LLMProvider],
    config: EvalTaskConfig,
    on_run_complete: Callable[[EvalRunRecord], None] | None = None,
) -> EvalSummary:
    agent_meta = await agent_registry.get(config.agent_metadata_id)
    if agent_meta is None:
        raise ValueError(
            f"Agent metadata '{config.agent_metadata_id}' not found in registry"
        )

    tool_factory = DefaultToolFactory()
    tools: list[Tool] = []
    for name in agent_meta.tool_names:
        tm = await tool_registry.get(name)
        if tm is not None:
            tools.append(tool_factory.create(tm))

    llm_provider = llm_provider_factory()

    return await _run_evaluation(
        llm_provider=llm_provider,
        tools=tools,
        system_prompt=agent_meta.resolve_system_prompt(),
        config=config,
        agent_metadata=agent_meta,
        on_run_complete=on_run_complete,
    )


async def run_evaluation_simple(
    *,
    llm_provider: LLMProvider,
    tools: Sequence[Tool],
    system_prompt: str,
    config: EvalTaskConfig,
    on_run_complete: Callable[[EvalRunRecord], None] | None = None,
) -> EvalSummary:
    return await _run_evaluation(
        llm_provider=llm_provider,
        tools=list(tools),
        system_prompt=system_prompt,
        config=config,
        agent_metadata=None,
        on_run_complete=on_run_complete,
    )


async def _run_evaluation(
    *,
    llm_provider: LLMProvider,
    tools: list[Tool],
    system_prompt: str,
    config: EvalTaskConfig,
    agent_metadata: AgentMetadata | None = None,
    on_run_complete: Callable[[EvalRunRecord], None] | None = None,
) -> EvalSummary:
    output_dir = _prepare_output_dir(config)
    persistence = EvalPersistence(str(output_dir))
    persistence.save_config(config)

    semaphore = asyncio.Semaphore(config.max_concurrency)
    runs: list[EvalRunRecord] = []
    _cancelled = False

    async def run_single(input_text: str) -> EvalRunRecord:
        nonlocal _cancelled
        run_record = EvalRunRecord(
            agent_metadata_id=config.agent_metadata_id,
            input_text=input_text,
            status="running",
            started_at=time.time(),
        )
        collector = EvalCollector(run_record.run_id, persistence)
        agent = SimpleAgent(
            llm_provider=llm_provider,
            max_iterations=config.max_iterations,
            middleware=[collector],
        )
        memory = ConversationMemory()

        try:
            async with semaphore:
                if _cancelled:
                    run_record.status = "interrupted"
                else:
                    async for event in agent.run(
                        user_input=[{"type": "text", "text": input_text}],
                        memory=memory,
                        tools=tools,
                        system_prompt=system_prompt,
                    ):
                        if isinstance(event, AgentEnd):
                            run_record.response = event.response
                            run_record.time_taken = event.time_taken
                            run_record.exceeded = event.exceeded
                            if event.interrupted:
                                run_record.status = "interrupted"
                            else:
                                run_record.status = "completed"
        except asyncio.CancelledError:
            run_record.status = "interrupted"
        except Exception as exc:
            run_record.status = "failed"
            run_record.error = f"{type(exc).__name__}: {exc}"

        run_record.ended_at = time.time()
        if run_record.time_taken is None:
            end = run_record.ended_at
            start = run_record.started_at
            if end is not None and start is not None:
                run_record.time_taken = end - start
        run_record.llm_call_count = collector.llm_call_count
        run_record.tool_call_count = collector.tool_call_count
        run_record.token_usage = collector.token_usage
        _compute_costs(run_record, config)

        persistence.write_run_summary(run_record)
        persistence.close_run(run_record.run_id)

        if on_run_complete:
            on_run_complete(run_record)

        return run_record

    tasks = [
        asyncio.create_task(run_single(input_text)) for input_text in config.inputs
    ]

    try:
        for coro in asyncio.as_completed(tasks):
            run_record = await coro
            runs.append(run_record)
    except asyncio.CancelledError:
        _cancelled = True
        for t in tasks:
            if not t.done():
                t.cancel()
        for t in tasks:
            try:
                runs.append(await t)
            except (asyncio.CancelledError, Exception):
                pass

    runs.sort(key=lambda r: r.started_at or 0.0)
    summary = _build_summary(config, runs, str(output_dir))
    persistence.write_summary(summary)

    report_path = output_dir / "report.html"
    generate_html_report(summary, str(report_path))

    persistence.close_all()

    return summary


def _prepare_output_dir(config: EvalTaskConfig) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dir_name = f"{config.name}_{timestamp}"
    output_dir = Path(config.output_dir) / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _compute_costs(run: EvalRunRecord, config: EvalTaskConfig) -> None:
    usage = run.token_usage
    if usage is None:
        return
    inp = config.cost_per_million_input_tokens
    outp = config.cost_per_million_output_tokens
    if inp is not None:
        usage.input_cost = usage.input_tokens / 1_000_000 * inp
        if outp is not None:
            usage.output_cost = usage.output_tokens / 1_000_000 * outp
            usage.total_cost = (usage.input_cost or 0) + (usage.output_cost or 0)
        else:
            usage.total_cost = usage.input_cost


def _build_summary(
    config: EvalTaskConfig,
    runs: list[EvalRunRecord],
    output_path: str,
) -> EvalSummary:
    completed = sum(1 for r in runs if r.status == "completed")
    failed = sum(1 for r in runs if r.status == "failed")
    interrupted = sum(1 for r in runs if r.status == "interrupted")
    total_time = sum((r.time_taken or 0.0) for r in runs)
    avg_time = total_time / len(runs) if runs else 0.0
    total_tokens = sum(
        (r.token_usage.total_tokens if r.token_usage else 0) for r in runs
    )
    costs = [
        r.token_usage.total_cost
        for r in runs
        if r.token_usage and r.token_usage.total_cost is not None
    ]
    total_cost = sum(costs) if costs else None

    return EvalSummary(
        task_name=config.name,
        description=config.description,
        agent_metadata_id=config.agent_metadata_id,
        total_runs=len(runs),
        completed=completed,
        failed=failed,
        interrupted=interrupted,
        total_time=total_time,
        avg_time=avg_time,
        total_tokens=total_tokens,
        total_cost=total_cost,
        runs=runs,
        output_path=output_path,
    )
