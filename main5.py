from langchain_openai import ChatOpenAI
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from browser_use import Agent, Browser
import functools
import re

# Read GOOGLE_API_KEY into env
load_dotenv()

# Initialize the model
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp')

# Helper function to extract URLs from text
def extract_urls(text):
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.findall(text)

async def main():
    browser = Browser()
    async with await browser.new_context() as context:
        # First agent handles login and initial game state detection
        print("Opening Kahoot game and entering nickname...")
        initAgent = Agent(
            task="""
            Enter the nickname as KAIhoot AI and check the current game state:
            
            1. If you see a "Waiting for host to start" screen, wait briefly
            2. If a question is already displayed, don't wait at all
            3. Don't use arbitrary fixed waiting times
            4. Report what you see after login so we know the game state
            
            The goal is to be ready for answering questions as quickly as possible.
            """,
            llm=llm,
            browser_context=context,
            use_vision=True,  # Enable vision to detect screen state
            initial_actions=[
                {
                    "open_tab": {
                        "url": "https://kahoot.it/?pin=9146034&refer_method=link"
                    }
                }
            ],
            save_conversation_path="logs/conversation",
        )

        await initAgent.run()
        print("Login completed. Proceeding to answer questions...")
        
        # Main game agent that will handle answering all questions
        gameAgent = Agent(
            task="""
            You are an expert in world knowledge participating in a quiz contest via the Kahoot platform. 
            Your goal is to follow the gameplay and achieve the highest score possible.

            IMPORTANT SCREEN STATE MONITORING:
            1. Question screen: When you see a question with answer options, analyze and select an answer.
            
            2. Answer result screen: After answering, Kahoot will show if you were correct or incorrect.
               - You'll see a "Correct" screen (usually green) or "Incorrect" screen (usually red)
               - It may show your current streak and points earned
               - IMPORTANT: When you see this screen, WAIT patiently without taking action
               - The game will automatically advance - do not click or interact during this phase
            
            3. Scoreboard/Ranking screen: Sometimes appears between questions.
               - If you see this screen, continue waiting patiently
               - Do not interact with this screen - it will automatically advance
            
            4. Next question: After the result and/or scoreboard screens, a new question will appear.
               - As soon as you see a new question, begin answering it following the process below
            
            FOR EACH QUESTION:
            1. Read the question text carefully
            2. Check if the question contains a link:
               - If you see a URL/link in the question, open it in a new tab
               - Read the content on the linked page carefully
               - Use that information to answer the question
               - Return to the Kahoot tab after gathering the information
            3. Analyze all answer options
            4. Think step by step about the correct answer
            5. Verbalize your reasoning
            6. Select your answer
            7. After selecting, DO NOT take any action - wait for the answer result screen
            8. When the result screen appears, continue waiting
            9. When a new question appears, repeat the process
            
            CRITICAL WAITING INSTRUCTIONS:
            - After selecting an answer, always wait for the answer result screen
            - When you see the answer result screen (correct/incorrect), continue waiting
            - Only take action when a new question with answer options appears
            - Never try to skip or click through these transition screens
            
            KAHOOT GAME FLOW UNDERSTANDING:
            - Kahoot controls the timing between screens automatically
            - Your only job is to answer questions when they appear
            - All transition screens (results, scoreboards) are handled by the game
            - Be patient during transitions and ready when questions appear
            
            Remember: The key is to recognize which screen you're looking at and only take action when a new question appears. Otherwise, wait patiently through result screens and scoreboards.
            """,
            browser_context=context,
            llm=llm,
            use_vision=True,  # Enable vision to better see the questions and options
            save_conversation_path="logs/answering",
        )

        # Run the game agent - it will continue handling all questions
        await gameAgent.run()
        
        print("Game has completed. Checking final results...")
        
        # Optional: Add a final agent to capture and report game results
        resultsAgent = Agent(
            task="""
            Check and report the final game results and your position on the leaderboard.
            Specifically:
            1. Note your final score
            2. Note your position on the leaderboard
            3. Report if you were in the top 3 players
            4. Screenshot the final results if possible
            """,
            browser_context=context,
            llm=llm,
            use_vision=True,
            save_conversation_path="logs/results",
        )
        
        await resultsAgent.run()
        print("Kahoot session complete!")

if __name__ == "__main__":
    asyncio.run(main())