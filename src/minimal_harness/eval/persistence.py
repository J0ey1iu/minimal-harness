from __future__ import annotations

import json
import os
from pathlib import Path
from typing import IO, Any

from .types import EvalEventRecord, EvalRunRecord, EvalSummary, EvalTaskConfig


class EvalPersistence:
    def __init__(self, output_dir: str) -> None:
        self._output_dir = Path(output_dir)
        self._runs_dir = self._output_dir / "runs"
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._open_files: dict[str, IO[str]] = {}

    def write_event(self, run_id: str, event: EvalEventRecord) -> None:
        if run_id not in self._open_files:
            path = self._runs_dir / f"{run_id}.jsonl"
            self._open_files[run_id] = open(path, "a", encoding="utf-8")
        f = self._open_files[run_id]
        data = _event_to_dict(event)
        line = json.dumps(data, ensure_ascii=False, default=str)
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())

    def write_run_summary(self, run: EvalRunRecord) -> None:
        path = self._runs_dir / f"{run.run_id}_summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_run_to_dict(run), f, ensure_ascii=False, default=str, indent=2)
            f.flush()
            os.fsync(f.fileno())

    def write_summary(self, summary: EvalSummary) -> None:
        path = self._output_dir / "summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                _summary_to_dict(summary), f, ensure_ascii=False, default=str, indent=2
            )

    def save_config(self, config: EvalTaskConfig) -> None:
        path = self._output_dir / "config.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                _config_to_dict(config), f, ensure_ascii=False, default=str, indent=2
            )

    def close_run(self, run_id: str) -> None:
        f = self._open_files.pop(run_id, None)
        if f:
            f.flush()
            os.fsync(f.fileno())
            f.close()

    def close_all(self) -> None:
        for f in self._open_files.values():
            f.flush()
            os.fsync(f.fileno())
            f.close()
        self._open_files.clear()


def _event_to_dict(event: EvalEventRecord) -> dict[str, Any]:
    return {
        "event_type": event.event_type,
        "timestamp": event.timestamp,
        "data": event.data,
    }


def _run_to_dict(run: EvalRunRecord) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "agent_metadata_id": run.agent_metadata_id,
        "input_text": run.input_text,
        "status": run.status,
        "started_at": run.started_at,
        "ended_at": run.ended_at,
        "time_taken": run.time_taken,
        "error": run.error,
        "response": run.response,
        "token_usage": _usage_to_dict(run.token_usage) if run.token_usage else None,
        "llm_call_count": run.llm_call_count,
        "tool_call_count": run.tool_call_count,
        "exceeded": run.exceeded,
    }


def _usage_to_dict(u: Any) -> dict[str, Any]:
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "total_tokens": u.total_tokens,
        "input_cost": u.input_cost,
        "output_cost": u.output_cost,
        "total_cost": u.total_cost,
    }


def _summary_to_dict(summary: EvalSummary) -> dict[str, Any]:
    return {
        "task_name": summary.task_name,
        "description": summary.description,
        "agent_metadata_id": summary.agent_metadata_id,
        "total_runs": summary.total_runs,
        "completed": summary.completed,
        "failed": summary.failed,
        "interrupted": summary.interrupted,
        "total_time": summary.total_time,
        "avg_time": summary.avg_time,
        "total_tokens": summary.total_tokens,
        "total_cost": summary.total_cost,
        "output_path": summary.output_path,
        "runs": [_run_to_dict(r) for r in summary.runs],
    }


def load_run_events(output_dir: str, run_id: str) -> list[dict[str, Any]]:
    path = Path(output_dir) / "runs" / f"{run_id}.jsonl"
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _config_to_dict(config: EvalTaskConfig) -> dict[str, Any]:
    return {
        "name": config.name,
        "description": config.description,
        "agent_metadata_id": config.agent_metadata_id,
        "inputs": config.inputs,
        "max_concurrency": config.max_concurrency,
        "output_dir": config.output_dir,
        "max_iterations": config.max_iterations,
        "cost_per_million_input_tokens": config.cost_per_million_input_tokens,
        "cost_per_million_output_tokens": config.cost_per_million_output_tokens,
    }
