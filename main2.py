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
llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash-exp',
    temperature=0.1,  # Lower temperature for more consistent responses
    max_tokens=2048
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
                6. Wait for the page to load and observe the screen state after joining:
                   - "Waiting for host to start" screen → Report this and wait patiently
                   - Question already visible → Report this immediately 
                   - Error messages (invalid pin, game full, etc.) → Report the specific error
                
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
            
            # Enhanced main game agent with improved question handling
            gameAgent = Agent(
                task="""
                You are participating in a Kahoot quiz. Your goal is to answer questions correctly and achieve high scores.

                SCREEN STATE RECOGNITION & ACTIONS:
                
                🔍 QUESTION SCREEN (Action Required):
                - You'll see a question at the top of the screen
                - Below are 2-4 colored answer buttons (typically red, blue, yellow, green)
                - Each answer button has both a color/shape and text
                - There's usually a countdown timer
                - ACTION: You MUST click on one of the colored answer buttons
                
                ⏳ RESULT SCREEN (Wait Only):
                - Shows "Correct!" (green) or "Incorrect" (red) 
                - May show points earned, streak info, and correct answer
                - ACTION: Wait patiently - do NOT click anything
                
                📊 SCOREBOARD/LEADERBOARD (Wait Only):
                - Shows player rankings, names, and scores
                - Usually appears between questions
                - ACTION: Wait patiently - do NOT click anything
                
                🎯 QUESTION ANSWERING STRATEGY:
                
                CRITICAL: When you see a question screen, you MUST click on one of the answer buttons. Here's how:
                
                1. READ THE QUESTION: Parse the question text at the top carefully
                
                2. IDENTIFY ANSWER BUTTONS: Look for the colored answer buttons (usually 2-4 options)
                   - Red button (triangle) - typically top-left
                   - Blue button (diamond) - typically top-right  
                   - Yellow button (circle) - typically bottom-left
                   - Green button (square) - typically bottom-right
                
                3. DECISION PROCESS:
                   - If you know the answer: Click the button with the correct answer text
                   - If you're uncertain but have an educated guess: Click your best guess
                   - If you don't know at all: Click the RED button (top-left, triangle shape)
                
                4. HOW TO CLICK:
                   - Look for the button that contains your chosen answer text
                   - Click directly on that colored button/shape
                   - The button should highlight or change when clicked
                   - Do NOT click on text alone - click on the actual colored button area
                
                5. FALLBACK RULE (VERY IMPORTANT):
                   - If you're completely unsure: ALWAYS click the RED button (triangle, usually top-left)
                   - If you can't identify the red button: Click the first/top-left answer option
                   - Never spend too much time deciding - quick decision is better than timing out
                
                6. AFTER CLICKING:
                   - Immediately STOP taking actions after clicking an answer
                   - Wait for the result screen to appear
                   - Do NOT click during result or scoreboard screens
                   - Only take action again when a new question appears
                
                TECHNICAL CLICKING INSTRUCTIONS:
                - In Kahoot, answers are presented as large colored buttons/shapes
                - Each button has both a geometric shape and answer text
                - Click on the entire button area, not just the text
                - The buttons are usually quite large and easy to click
                - If a button doesn't respond, try clicking in the center of the button
                
                ERROR HANDLING:
                - If you see "Time's up" or timer expires: Wait for next question
                - If buttons don't seem clickable: Try refreshing or report the issue
                - If game disconnects: Report the disconnection
                - If you're unsure what screen you're on: Describe what you see
                
                TIMING RULES:
                - Answer quickly - Kahoot rewards speed
                - Don't spend more than 10-15 seconds analyzing
                - If unsure after quick analysis: Default to RED button
                - Speed + participation is better than perfect accuracy
                
                Remember: Your primary job is to CLICK answer buttons when questions appear. Everything else is waiting.
                """,
                browser_context=context,
                llm=llm,
                use_vision=True,
                save_conversation_path="logs/game_conversation",
            )

            # Run the main game with retry logic
            await run_agent_with_retry(gameAgent)
            
            logger.info("Game completed. Capturing final results...")
            
            # Enhanced results capture agent
            resultsAgent = Agent(
                task="""
                Capture and analyze the final Kahoot game results:
                
                INFORMATION TO GATHER:
                1. Final score/points earned
                2. Final leaderboard position (1st, 2nd, 3rd, etc.)
                3. Number of correct vs incorrect answers
                4. Any achievements or badges earned
                5. Overall game statistics if visible
                
                ACTIONS TO TAKE:
                1. Take a screenshot of the final results screen
                2. Look for detailed statistics page if available
                3. Report all gathered information clearly
                4. Note if the bot achieved top 3 position
                5. Report the final nickname used and total points
                
                If the game is still in progress or results aren't visible yet, 
                wait briefly and check again.
                """,
                browser_context=context,
                llm=llm,
                use_vision=True,
                save_conversation_path="logs/results_conversation",
            )
            
            await run_agent_with_retry(resultsAgent, max_retries=2)
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