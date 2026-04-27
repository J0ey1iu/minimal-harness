"""Demo client events using pytest."""

import ast
import asyncio
import json
import os
from typing import AsyncIterator

from minimal_harness import StreamingTool
from minimal_harness.agent import SimpleAgent
from minimal_harness.client.events import (
    AgentEndEvent,
    ToolProgressEvent,
    ToolStartEvent,
    to_client_event,
)
from minimal_harness.llm.openai import OpenAILLMProvider
from minimal_harness.memory import ConversationMemory


async def calculator_handler(expression: str) -> AsyncIterator[dict]:
    yield {"status": "progress", "message": f"I'm about to calculate: {expression}"}
    try:
        result = ast.literal_eval(expression)
        yield {"success": True, "expression": expression, "result": result}
    except Exception as e:
        yield {"success": False, "error": str(e)}


calculator_tool = StreamingTool(
    name="calculator",
    description="Evaluate a mathematical expression and return the result.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate (e.g., '2+2', '10*5', '2**3')",
            },
        },
        "required": ["expression"],
    },
    fn=calculator_handler,
)


async def slow_calculator_handler(expression: str) -> AsyncIterator[dict]:
    yield {"status": "progress", "message": f"I'm about to calculate: {expression}"}
    await asyncio.sleep(2)
    try:
        result = ast.literal_eval(expression)
        yield {"success": True, "expression": expression, "result": result}
    except Exception as e:
        yield {"success": False, "error": str(e)}


slow_calculator_tool = StreamingTool(
    name="slow_calculator",
    description="Evaluate a mathematical expression slowly and return the result.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate",
            },
        },
        "required": ["expression"],
    },
    fn=slow_calculator_handler,
)


async def read_file_handler(file_path: str) -> AsyncIterator[dict]:
    yield {"status": "progress", "message": f"I'm about to read file: {file_path}"}
    try:
        with open(file_path, "r") as f:
            content = f.read()
        yield {"success": True, "file_path": file_path, "content": content}
    except Exception as e:
        yield {"success": False, "error": str(e)}


read_file_tool = StreamingTool(
    name="read_file",
    description="Read the contents of a file.",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to read",
            },
        },
        "required": ["file_path"],
    },
    fn=read_file_handler,
)


def get_agent(tools=None):
    from openai import AsyncOpenAI

    api_key = os.getenv("MH_API_KEY")
    base_url = os.getenv("MH_BASE_URL")
    model = os.getenv("MH_MODEL", "qwen3.5-27b")

    if base_url is not None and api_key is not None:
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    elif base_url is not None:
        client = AsyncOpenAI(base_url=base_url)
    elif api_key is not None:
        client = AsyncOpenAI(api_key=api_key)
    else:
        client = AsyncOpenAI()
    llm_provider = OpenAILLMProvider(client=client, model=model)
    memory = ConversationMemory(system_prompt="You are a helpful assistant.")
    agent = SimpleAgent(
        llm_provider=llm_provider,
        tools=tools or [],
        memory=memory,
    )
    return agent


def _safeSerialize(obj):
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


async def run_and_collect(agent, user_input, stop_event=None, output_file=None):
    async for event in agent.run(
        user_input=user_input,
        stop_event=stop_event,
    ):
        client_event = to_client_event(event)
        if output_file:
            with open(output_file, "a") as f:
                event_name = type(client_event).__name__
                event_data = {
                    k: _safeSerialize(v)
                    for k, v in vars(client_event).items()
                    if not k.startswith("_")
                }
                f.write(
                    f"{event_name}: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                )
        if isinstance(client_event, AgentEndEvent):
            break


async def demo_llm_only():
    """Demo 1: Simple user input that triggers only LLM (no tools)."""
    output_file = "./demo_01_llm_only.txt"
    if os.path.exists(output_file):
        os.remove(output_file)

    agent = get_agent(tools=[])
    await run_and_collect(
        agent,
        user_input=[{"type": "text", "text": "Say hello in exactly 3 words."}],
        output_file=output_file,
    )
    print(f"Output written to {output_file}")


async def demo_single_tool_success():
    """Demo 2: User input that triggers one tool and LLM response succeeded."""
    output_file = "./demo_02_single_tool_success.txt"
    if os.path.exists(output_file):
        os.remove(output_file)

    agent = get_agent(tools=[calculator_tool])
    await run_and_collect(
        agent,
        user_input=[{"type": "text", "text": "What is 125 * 37?"}],
        output_file=output_file,
    )
    print(f"Output written to {output_file}")


async def demo_single_tool_failure():
    """Demo 3: User input that triggers one tool and that tool failed once."""
    output_file = "./demo_03_single_tool_failure.txt"
    if os.path.exists(output_file):
        os.remove(output_file)

    agent = get_agent(tools=[calculator_tool])
    await run_and_collect(
        agent,
        user_input=[{"type": "text", "text": "What is 125 / 0?"}],
        output_file=output_file,
    )
    print(f"Output written to {output_file}")


async def demo_multiple_tools_success():
    """Demo 4: User input that triggers multiple tools and all of them succeeded."""
    output_file = "./demo_04_multiple_tools_success.txt"
    if os.path.exists(output_file):
        os.remove(output_file)

    test_file = "./demo_multifile.txt"
    with open(test_file, "w") as f:
        f.write("Hello World")

    agent = get_agent(tools=[read_file_tool, calculator_tool])
    await run_and_collect(
        agent,
        user_input=[
            {
                "type": "text",
                "text": f"Read the file at {test_file} and then calculate 10 + 20.",
            }
        ],
        output_file=output_file,
    )

    os.remove(test_file)
    print(f"Output written to {output_file}")


async def demo_stop_at_llm_response():
    """Demo 5.1: User input that triggers a tool and stop event emits at LLM response duration."""
    output_file = "./demo_05a_stop_at_llm.txt"
    if os.path.exists(output_file):
        os.remove(output_file)

    agent = get_agent(tools=[calculator_tool])
    stop_event = asyncio.Event()

    async def set_stop_early():
        await asyncio.sleep(0.1)
        stop_event.set()

    async def run_with_early_stop():
        task = asyncio.create_task(
            run_and_collect(
                agent,
                user_input=[{"type": "text", "text": "What is 125 * 37?"}],
                stop_event=stop_event,
                output_file=output_file,
            )
        )
        await set_stop_early()
        return await task

    await run_with_early_stop()
    print(f"Output written to {output_file}")


async def demo_stop_at_tool_execution():
    """Demo 5.2: User input that triggers a tool and stop event emits at Tool executing duration."""
    output_file = "./demo_05b_stop_at_tool.txt"
    if os.path.exists(output_file):
        os.remove(output_file)

    agent = get_agent(tools=[slow_calculator_tool])
    stop_event = asyncio.Event()
    tool_started = False

    async for event in agent.run(
        user_input=[{"type": "text", "text": "What is 1 + 1?"}],
        stop_event=stop_event,
    ):
        client_event = to_client_event(event)
        with open(output_file, "a") as f:
            event_name = type(client_event).__name__
            event_data = {
                k: _safeSerialize(v)
                for k, v in vars(client_event).items()
                if not k.startswith("_")
            }
            f.write(f"{event_name}: {json.dumps(event_data, ensure_ascii=False)}\n\n")
        if isinstance(client_event, ToolStartEvent):
            tool_started = True
        elif tool_started and isinstance(client_event, ToolProgressEvent):
            stop_event.set()
        if isinstance(client_event, AgentEndEvent):
            break

    print(f"Output written to {output_file}")


async def main():
    print("Running client events demos...\n")
    await demo_llm_only()
    await demo_single_tool_success()
    await demo_single_tool_failure()
    await demo_multiple_tools_success()
    await demo_stop_at_llm_response()
    await demo_stop_at_tool_execution()
    print("\nAll demos completed!")


if __name__ == "__main__":
    asyncio.run(main())
