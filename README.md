# Kahoot Agent

A Python-based automated agent that participates in Kahoot quizzes using browser automation and Azure OpenAI's GPT-4o-mini model.

## Prerequisites

- Python 3.8 or higher
- Azure OpenAI API access with GPT-4o-mini deployment
- Internet connection

## Installation

You can set up this project using either UV (recommended) or traditional Venv. Choose the method that works best for your workflow.

### Option 1: Using UV (Recommended)

[UV](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

1. Install UV if you don't have it already:

   ```bash
   pip install uv
   ```

2. Clone the repository:

   ```bash
   git clone <repository-url>
   cd kahoot-agent
   ```

3. Create a virtual environment and install dependencies:

   ```bash
   uv venv
   uv pip install -r requirements.txt
   ```

4. Activate the virtual environment:
   - Windows:
     ```bash
     .\.venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

### Option 2: Using Traditional Venv

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd kahoot-agent
   ```

2. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:

   - Windows:
     ```bash
     .\.venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Fill the API key and endpoint in .env

2. Edit the `.env` file and add your Azure OpenAI API credentials:
   ```
   AZURE_OPENAI_API_KEY="your-api-key"
   AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com/"
   BROWSER_USE_LOGGING_LEVEL="info"
   ```

## Usage

1. Make sure your virtual environment is activated.

2. Update the Kahoot game PIN in `main.py` if needed:

   ```python
   "url": "https://kahoot.it/?pin=YOUR_GAME_PIN&refer_method=link"
   ```

3. Run the agent:

   ```bash
   python main.py
   ```

4. The agent will:
   - Join the Kahoot game with the nickname "KAIhoot AI"
   - Wait for the game to start
   - Read and answer questions as they appear
   - Prompt you to press Enter to continue to the next question

## Project Structure

- `main.py`: The main script that initializes and runs the Kahoot agent
- `.env`: Configuration file for API keys and other settings
- `logs/`: Directory where conversation logs are saved

## Troubleshooting

- If you encounter browser automation issues, make sure you have a compatible browser installed.
- For API connection issues, verify your Azure OpenAI credentials in the `.env` file.
- If the agent fails to read questions correctly, try adjusting the browser window size or ensuring the Kahoot interface is fully visible.

## Dependencies

The project relies on:

- `langchain_openai`: For interacting with Azure OpenAI API
- `browser_use`: For browser automation
- `python-dotenv`: For loading environment variables

## License

[Specify license information here]


## Start
