import os
import sys

from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio

from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


from browser_use import Agent, Browser
from browser_use.agent.service import Agent, Browser, Controller

load_dotenv()

# Retrieve Azure-specific environment variables
# azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
# azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# if not azure_openai_api_key or not azure_openai_endpoint:
#     raise ValueError("AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT is not set")

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp')

async def main():
    browser = Browser()
    async with await browser.new_context() as context:
        # https://docs.browser-use.com/customize/supported-models
        # Initialize the Azure OpenAI client
        # llm = AzureChatOpenAI(
        #     model_name="gpt-4o-mini",
        #     openai_api_key=azure_openai_api_key,
        #     azure_endpoint=azure_openai_endpoint,  # Corrected to use azure_endpoint instead of openai_api_base
        #     deployment_name="gpt-4o-mini",  # Use deployment_name for Azure models
        #     api_version="2024-08-01-preview",  # Explicitly set the API version here
        # )

        # openAILlm = ChatOpenAI(
        #     model="gpt-4o-mini",
        #     temperature=0.0,
        # )

        print("Opening Kahoot game and entering nickname...")

        # enter the game and wait for it to start
        initAgent = Agent(
            task="Enter the nickname as KAIhoot AI and wait for the game to start by the host using your wait function",
            llm=llm,
            browser_context=context,
            use_vision=False,
            initial_actions=[
                {
                    "open_tab": {
                        "url": "https://kahoot.it/?pin=5772382&refer_method=link"
                    }
                }
            ],
            save_conversation_path="logs/conversation",
        )

        await initAgent.run()
        print("Login agent completed. Proceeding to answer questions...")
        
        print("Starting quiz agent...")
        # answer the question
        answerAgent = Agent(
            task=""""
            You are an expert in world knowledge participating in a contest via the Kahoot platform. Your goal is to follow the gameplay and achieve the highest score possible.

            Some rules:
            1. You should think step by step before answering tricky questions that appear on the screen
            2. Say your answer before selecting it.
            3. Choose a random answer that you think might be correct.""",
            browser_context=context,
            llm=llm,
            use_vision=False,
            save_conversation_path="logs/answering",
        )

        await initAgent.run()
        await answerAgent.run()
        # wait for the game to start next question
        input("Press Enter to continue...")


asyncio.run(main())
