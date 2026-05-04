import asyncio
import os

from openai import AsyncOpenAI

from minimal_harness.agent.simple import SimpleAgent
from minimal_harness.client.events import to_client_event
from minimal_harness.llm.openai import OpenAILLMProvider
from minimal_harness.memory import ConversationMemory
from minimal_harness.tool.built_in.bash import get_tools as get_bash_tools
from minimal_harness.types import AgentEnd, LLMChunk

api_key = os.getenv("MH_API_KEY")
base_url = os.getenv("MH_BASE_URL")
model = os.getenv("MH_MODEL", "minimax-m2.7")

if base_url is not None and api_key is not None:
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
elif base_url is not None:
    client = AsyncOpenAI(base_url=base_url)
elif api_key is not None:
    client = AsyncOpenAI(api_key=api_key)
else:
    client = AsyncOpenAI()
llm_provider = OpenAILLMProvider(client=client, model=model)
memory = ConversationMemory()
agent = SimpleAgent(llm_provider=llm_provider)


async def main():
    stop_event = asyncio.Event()
    async for event in agent.run(
        user_input=[
            {
                "type": "text",
                "text": (
                    "I have an individual here named Bajie, "
                    "who's gonna take part in a pretty sensitive project. "
                    "I have make sure that he has nothing to do with "
                    "the people who are already in this project to "
                    "make sure that no one can steal money from it. "
                    "project files are in current directory, you can "
                    "look for a `files` folder inside a `relation_digging` "
                    "folder. You can refer to these files and use them to "
                    "find out whether Bajie here is a 'safe' person."
                ),
            }
        ],
        stop_event=stop_event,
        memory=memory,
        tools=list(get_bash_tools().values()),
    ):
        client_event = to_client_event(event)
        if isinstance(client_event, LLMChunk):
            continue
        print(str(client_event))
        print()
        if isinstance(client_event, AgentEnd):
            break


if __name__ == "__main__":
    asyncio.run(main())
