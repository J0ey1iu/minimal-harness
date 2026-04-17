from minimal_harness.tool.base import InteractiveTool


async def ask_user_first(question: str) -> str:
    return question


async def ask_user_final(user_input: str, question: str) -> str:
    return user_input


ask_user_tool = InteractiveTool(
    name="ask_user",
    description="Ask the user a question and wait for their response. Use this when you need user input to proceed.",
    parameters={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask the user",
            },
        },
        "required": ["question"],
    },
    fn_first=ask_user_first,
    fn_final=ask_user_final,
)


def get_tools() -> dict[str, InteractiveTool]:
    return {"ask_user": ask_user_tool}
