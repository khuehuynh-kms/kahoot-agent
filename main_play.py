from langchain_openai import ChatOpenAI
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from browser_use import Agent, Browser
import functools
import re
import time
import logging
from typing import Optional
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Read GOOGLE_API_KEY into env
load_dotenv()

kahootLink = os.getenv("KAHOOT_LINK")

# Initialize the model with better configuration
# llm = ChatGoogleGenerativeAI(
#     model='gemini-2.0-flash-exp',
#     temperature=0.1,  # Lower temperature for more consistent responses
#     max_tokens=2048
# )
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
)

# Helper function to extract URLs from text
def extract_urls(text):
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.findall(text)

# Enhanced retry mechanism
async def run_agent_with_retry(agent, max_retries=3, delay=2):
    """Run agent with retry logic for better reliability"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Running agent (attempt {attempt + 1}/{max_retries})")
            result = await agent.run()
            return result
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {max_retries} attempts failed")
                raise

async def main():
    browser = Browser()
    
    try:
        async with await browser.new_context() as context:
            # Enhanced initialization agent with better error handling
            logger.info("Starting Kahoot session...")
            
            initAgent = Agent(
                task=f"""
                Join the Kahoot game with nickname '{KahootConfig.NICKNAME}' and assess the current state:
                
                STEP-BY-STEP PROCESS:
                1. First, navigate to the Kahoot game URL: {KahootConfig.get_game_url()}
                2. Look for the nickname input field (usually has placeholder "Enter your nickname")
                3. Click on the input field and clear any existing text
                4. Type the nickname '{KahootConfig.NICKNAME}' exactly
                5. Look for and click the "Join" or "Enter" button to join the game
                6. After joining, observe the current screen state:
                
                IF YOU SEE "WAITING FOR HOST TO START" OR SIMILAR:
                - Take a screenshot and report "Waiting for game to start"
                - STAY ON THIS PAGE and keep monitoring
                - DO NOT exit or finish the task yet
                - Keep checking every few seconds for changes
                - Wait until you see the first question with colored answer buttons
                
                IF YOU SEE A QUESTION WITH ANSWER BUTTONS:
                - Take a screenshot
                - Report "First question is now visible - ready to start answering"
                - Your job is complete - the game agent will take over
                
                IF YOU SEE ERROR MESSAGES:
                - Take a screenshot
                - Report the specific error (invalid pin, game full, etc.)
                - Do not proceed further
                
                IMPORTANT: 
                - Take a screenshot after successfully joining to document the state
                - If you see any error messages, take a screenshot and report exactly what the error says
                - Report clearly what screen you see so the next agent knows the current game status
                """,
                llm=llm,
                browser_context=context,
                use_vision=True,
                save_conversation_path="logs/init_conversation",
            )

            await run_agent_with_retry(initAgent)
            logger.info("Login phase completed. Starting main game agent...")
            
            answerAgent = Agent(
                task=""""
                You are an expert in world knowledge participating in a contest via the Kahoot platform. Your goal is to follow the gameplay and achieve the highest score possible.

                Some rules:
                You are playing Kahoot. Follow these steps:
                1. Read the question text, which is displayed on a white background.
                2. Determine the correct answer based on the question.
                3. There are four answer options:
                  - The first answer has a red background.
                  - The second answer has a blue background.
                  - The third answer has a yellow background.
                  - The fourth answer has a green background.
                4. Log the question and answers to track.
                5. Identify the correct answer among the four options.
                6. Choose the matching answer by clicking the corresponding colored option.
                7. Wait for the next question to appear, then repeat from step 1.
                """,
                browser_context=context,
                llm=llm,
                use_vision=False,
                save_conversation_path="logs/answering",
            )

            # Run the main game with retry logic
            await run_agent_with_retry(answerAgent)
            
            logger.info("Game completed. Capturing final results...")
            
            logger.info("Kahoot session completed successfully!")
            
    except Exception as e:
        logger.error(f"Critical error in main execution: {str(e)}")
        raise
    finally:
        # Ensure browser cleanup
        try:
            await browser.close()
            logger.info("Browser closed successfully")
        except Exception as e:
            logger.warning(f"Error closing browser: {str(e)}")

# Enhanced configuration and monitoring
class KahootConfig:
    """Configuration class for Kahoot bot settings"""
    NICKNAME = "PADI"
    MAX_RETRIES = 3
    RETRY_DELAY = 2
    ANSWER_TIMEOUT = 30  # seconds to wait for answer selection
    RESULT_WAIT_TIME = 5  # seconds to wait on result screens
    DEFAULT_ANSWER_POSITION = 1  # Always choose first answer when unsure (1=first, 2=second, etc.)
    
    @classmethod
    def get_game_url(cls, pin: str = None) -> str:
        return kahootLink

async def run_kahoot_bot(nickname: Optional[str] = None):
    """
    Enhanced entry point for running Kahoot bot
    
    Args:
        nickname: Custom nickname (defaults to PADI)
    """
    if nickname:
        KahootConfig.NICKNAME = nickname
    
    logger.info(f"Starting Kahoot bot with nickname: {KahootConfig.NICKNAME}")
    logger.info(f"Game link: {kahootLink}")
    await main()

if __name__ == "__main__":
    # Run the bot with default nickname
    asyncio.run(run_kahoot_bot("PADI"))