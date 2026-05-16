from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalEventRecord:
    event_type: str
    timestamp: float
    data: dict[str, Any]


@dataclass
class TokenUsageRecord:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_cost: float | None = None
    output_cost: float | None = None
    total_cost: float | None = None


@dataclass
class EvalRunRecord:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_metadata_id: str = ""
    input_text: str = ""
    status: str = "pending"
    started_at: float | None = None
    ended_at: float | None = None
    time_taken: float | None = None
    error: str | None = None
    response: str | None = None
    token_usage: TokenUsageRecord | None = None
    llm_call_count: int = 0
    tool_call_count: int = 0
    exceeded: bool = False


@dataclass
class EvalTaskConfig:
    name: str
    description: str = ""
    agent_metadata_id: str = ""
    inputs: list[str] = field(default_factory=list)
    max_concurrency: int = 4
    output_dir: str = "./eval_results"
    max_iterations: int = 20
    cost_per_million_input_tokens: float | None = None
    cost_per_million_output_tokens: float | None = None


@dataclass
class EvalSummary:
    task_name: str
    description: str
    agent_metadata_id: str
    total_runs: int
    completed: int
    failed: int
    interrupted: int
    total_time: float
    avg_time: float
    total_tokens: int
    total_cost: float | None
    runs: list[EvalRunRecord]
    output_path: str
