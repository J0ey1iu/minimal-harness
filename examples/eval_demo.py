#!/usr/bin/env python3
"""
Example: Evaluating a Math Agent with the Eval Module.

This demo shows how to:
  1. Define an AgentMetadata and ToolMetadata (with a calculator tool).
  2. Register them in the framework's registries.
  3. Run a batch evaluation with multiple test inputs.
  4. Generate a self-contained HTML report.

Usage:
  MH_API_KEY=sk-... MH_BASE_URL=https://... python examples/eval_demo.py

After execution, open the generated HTML report in your browser:
  open eval_results/math-eval_*/report.html
"""

import asyncio
import os
from typing import AsyncIterator

from minimal_harness.agent.registry import AgentRegistry
from minimal_harness.eval import EvalTaskConfig, run_evaluation
from minimal_harness.llm.openai import OpenAILLMProvider
from minimal_harness.tool.registry import ToolRegistry
from minimal_harness.types import AgentMetadata, LocalToolBinding, ToolMetadata

# ── 1. Define a tool implementation ──────────────────────────────────────────


async def calculator(expression: str) -> AsyncIterator[dict]:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)  # noqa: S307 — demo only
        yield {"success": True, "expression": expression, "result": result}
    except Exception as e:
        yield {"success": False, "error": str(e)}


# ── 2. Set up registries ─────────────────────────────────────────────────────


async def setup() -> tuple[AgentRegistry, ToolRegistry, OpenAILLMProvider]:
    agent_registry = AgentRegistry()
    tool_registry = ToolRegistry()

    # Register the calculator tool
    await tool_registry.register(
        ToolMetadata(
            name="calculator",
            display_name="Calculator",
            description="Evaluate a mathematical expression and return the result",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate (e.g. '2 + 2')",
                    },
                },
                "required": ["expression"],
            },
            binding=LocalToolBinding(fn=calculator),
        )
    )

    # Register a math agent that uses the calculator tool
    await agent_registry.register(
        AgentMetadata(
            name="math_agent",
            display_name="Math Assistant",
            description="A helpful assistant that can solve math problems",
            system_prompt=(
                "You are a helpful math assistant. "
                "Use the calculator tool to solve math problems."
            ),
            agent_type="simple",
            tool_names=["calculator"],
        )
    )

    # Create the LLM provider
    api_key = os.getenv("MH_API_KEY")
    base_url = os.getenv("MH_BASE_URL")
    model = os.getenv("MH_MODEL", "gpt-4o-mini")

    client_kwargs = {}
    if base_url:
        client_kwargs["base_url"] = base_url
    if api_key:
        client_kwargs["api_key"] = api_key

    from openai import AsyncOpenAI

    llm_provider = OpenAILLMProvider(
        client=AsyncOpenAI(**client_kwargs),
        model=model,
    )

    return agent_registry, tool_registry, llm_provider


# ── 3. Run the evaluation ────────────────────────────────────────────────────


async def main() -> None:
    print("Setting up registries and LLM provider...")
    agent_registry, tool_registry, llm_provider = await setup()

    print("Starting evaluation...")
    summary = await run_evaluation(
        agent_registry=agent_registry,
        tool_registry=tool_registry,
        llm_provider_factory=lambda: llm_provider,
        config=EvalTaskConfig(
            name="math-eval",
            description="Test the math agent's ability to solve arithmetic problems",
            agent_metadata_id="math_agent",
            inputs=[
                "What is 2 + 2?",
                "Calculate 15 * 37",
                "What is 100 / 3?",
                "If I have 8 apples and eat 3, how many are left?",
                "Solve: (12 + 8) * 5 - 3",
                "What is the square root of 144?",
            ],
            max_concurrency=3,
            output_dir="./eval_results",
            max_iterations=10,
            cost_per_million_input_tokens=0.15,
            cost_per_million_output_tokens=0.60,
        ),
        on_run_complete=lambda r: print(
            f"  [{r.status:>12}] {r.run_id}: {r.input_text[:50]}...  "
            f"({r.llm_call_count} LLM calls, {r.token_usage.total_tokens if r.token_usage else 0} tokens)"
        ),
    )

    print("\nEvaluation complete!")
    print(f"  Total runs:   {summary.total_runs}")
    print(f"  Completed:    {summary.completed}")
    print(f"  Failed:       {summary.failed}")
    print(f"  Total tokens: {summary.total_tokens}")
    if summary.total_cost is not None:
        print(f"  Total cost:   ${summary.total_cost:.4f}")
    print(f"  Report:       {summary.output_path}/report.html")


if __name__ == "__main__":
    asyncio.run(main())
