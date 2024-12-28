"""
Browser automation agent using LangChain and OpenAI.

@dev You need to add OPENAI_API_KEY to your environment variables.
"""

import os
import sys
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent, Controller, ActionResult

# Initialize the LLM
llm = ChatOpenAI(model='gpt-4o')  # Fixed model name from 'gpt-4o' to 'gpt-4'

# Initialize the controller
controller = Controller()

@controller.action('Ask user for information')
async def ask_human(question: str) -> ActionResult:
    """Ask the user a question and return their response."""
    print("Ask human")
    answer = input(f'\n{question}\nInput: ')
    return ActionResult(extracted_content=answer, include_in_memory=True)

# Create the agent with proper initialization
agent = Agent(
    task='Go to amazon.com, search for laptop, sort by best rating, and give me the price of the first result',
    llm=llm,
    controller=controller  # Added missing controller parameter
)

async def main():
    try:
        await agent.run(max_steps=10)
        # Uncomment the following line if you want to create a GIF of the browsing history
        # await agent.create_history_gif()
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
