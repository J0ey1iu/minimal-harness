from openai import AsyncOpenAI

from minimal_harness.agent import OpenAIAgent
from minimal_harness.llm.openai import OpenAILLMProvider
from minimal_harness.memory import ConversationMemory
from minimal_harness.tool import AgenticTool
from minimal_harness.tool.built_in import bash, glob, grep, read_file
from minimal_harness.tool_executor import ToolExecutor

PROJECT_READER_SYSTEM_PROMPT = (
    "You are a project analysis assistant. Your job is to analyze software projects "
    "and answer questions about them.\n\n"
    "You have access to the following tools:\n"
    "- read_file: Read the contents of a file\n"
    "- glob: Find files matching a pattern\n"
    "- grep: Search file contents for patterns\n"
    "- bash: Execute shell commands\n\n"
    "When given a project path and a question:\n"
    "1. First list the top-level files and directories to understand the project structure\n"
    "2. Read key files (README, package.json, main entry points) to understand the project\n"
    "3. Search for specific files or patterns relevant to the question\n"
    "4. Provide a clear, comprehensive answer based on your findings\n\n"
    "Be thorough but concise. Focus on answering the specific question asked."
)


def create_project_reader_tool(
    api_key: str,
    base_url: str,
    model: str,
) -> AgenticTool:
    client = AsyncOpenAI(base_url=base_url, api_key=api_key or None)
    llm_provider = OpenAILLMProvider(client=client, model=model)

    sub_tools = [
        read_file.read_file_tool,
        glob.glob_tool,
        grep.grep_tool,
        bash.bash_tool,
    ]

    tool_executor = ToolExecutor({t.name: t for t in sub_tools})
    memory = ConversationMemory(system_prompt=PROJECT_READER_SYSTEM_PROMPT)

    sub_agent = OpenAIAgent(
        llm_provider=llm_provider,
        tools=sub_tools,
        tool_executor=tool_executor,
        memory=memory,
    )

    return AgenticTool(
        name="project_reader",
        description=(
            "Analyze a software project and answer questions about it. "
            "Provide project_path (absolute path to the project directory) "
            "and question (what you want to know about the project)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "project_path": {
                    "type": "string",
                    "description": "Absolute path to the project directory to analyze",
                },
                "question": {
                    "type": "string",
                    "description": "The question to answer about the project",
                },
            },
            "required": ["project_path", "question"],
        },
        agent=sub_agent,
    )
