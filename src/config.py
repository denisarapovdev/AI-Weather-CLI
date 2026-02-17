import os
import sys

from dotenv import load_dotenv

load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

# Open-Meteo API URLs
OPEN_METEO_GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# Tool names
WEATHER_TOOL_NAME = "get_weather"

# CLI messages
CLI_WELCOME_MESSAGE = "Weather Assistant - ask about the weather in any city."
CLI_EXIT_INSTRUCTION = "Type 'quit' to exit."
CLI_USER_PROMPT = "\nYou: "
CLI_ASSISTANT_PROMPT = "Assistant: "
CLI_GOODBYE_MESSAGE = "Goodbye!"
CLI_GOODBYE_MESSAGE_NEWLINE = "\nGoodbye!"
CLI_UNEXPECTED_ERROR = "Unexpected error: {error}"
CLI_FATAL_ERROR = "Fatal error: {error}"
CLI_EXIT_COMMANDS = ["quit", "exit"]

# System prompt for the assistant
SYSTEM_PROMPT = (
    "You are a helpful weather assistant. "
    "You can check the weather for any city. "
    "IMPORTANT: Only call the weather tool if the user explicitly asks for weather information or mentions a specific location. "
    "If the user greets you (e.g., 'Hi', 'Hello') or asks general questions, reply conversationally WITHOUT calling any tools.\n\n"
    "When providing weather information for multiple cities, format your response as follows:\n"
    "- Start with a brief introduction like 'Here's the current weather in each city:'\n"
    "- Use a bulleted list with markdown formatting: **City, Country**: detailed description\n"
    "- For each city, provide a natural, conversational description including:\n"
    "  * Temperature and how it feels (apparent temperature)\n"
    "  * Humidity percentage\n"
    "  * Wind speed and direction if available\n"
    "  * Precipitation status\n"
    "- End up with your short opinion about the weather in friendly format and ask if client want to know anything else\n"
    "Always allow for follow-up questions and be conversational.\n\n"
    "What you should NOT do:\n"
    "- Do not make up or invent weather data - only use information from the tool calls\n"
    "- Do not provide weather forecasts for future dates - only current weather\n"
    "- Do not use overly technical or scientific language - keep it conversational\n"
    "- Do not repeat the same information multiple times\n"
    "- Do not provide weather information for cities that were not requested\n"
    "- Do not assume weather conditions without actual data from the tool\n"
    "- Do not provide info about Tool call in your response\n"
    "- Do NOT call the weather tool if the user message does not contain a location or a weather-related question."
)


def validate_config() -> None:
    """Validate that required configuration is present."""
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        sys.exit(1)
