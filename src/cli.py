"""CLI utilities and main loop for the weather assistant."""

import asyncio

from .assistant import WeatherAssistant
from .config import (
    CLI_EXIT_COMMANDS,
    CLI_EXIT_INSTRUCTION,
    CLI_GOODBYE_MESSAGE,
    CLI_GOODBYE_MESSAGE_NEWLINE,
    CLI_UNEXPECTED_ERROR,
    CLI_USER_PROMPT,
    CLI_WELCOME_MESSAGE,
)


async def get_input_async(prompt: str) -> str:
    """
    Reads input from stdin without blocking the asyncio event loop.

    Uses run_in_executor to offload the blocking sys.stdin.readline/input call.

    Args:
        prompt: Prompt string to display

    Returns:
        User input string
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, input, prompt)


async def run_cli() -> None:
    """
    Main CLI loop for the weather assistant.

    Handles user input, processes messages, and manages the conversation loop.
    """
    print(CLI_WELCOME_MESSAGE)
    print(CLI_EXIT_INSTRUCTION)
    assistant = WeatherAssistant()

    try:
        while True:
            try:
                user_text = await get_input_async(CLI_USER_PROMPT)
                user_text = user_text.strip()

                if user_text.lower() in CLI_EXIT_COMMANDS:
                    print(CLI_GOODBYE_MESSAGE)
                    break

                if not user_text:
                    continue

                await assistant.process_message(user_text)

            except (KeyboardInterrupt, EOFError):
                print(CLI_GOODBYE_MESSAGE_NEWLINE)
                break
            except Exception as e:
                print(CLI_UNEXPECTED_ERROR.format(error=e))
    finally:
        await assistant.close()
