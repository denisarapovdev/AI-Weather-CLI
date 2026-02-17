# AI Weather CLI

A command-line weather assistant powered by LLM (Large Language Model) with tool calling capabilities.

## Getting Started

### Prerequisites

- Python 3.12 or higher
- OpenAI API key (or compatible local LLM endpoint)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AI-Weather-CLI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=http://localhost:8000/v1  # Optional: for local models
MODEL_NAME=gpt-4o  # Optional: defaults to gpt-4o
```

**Note**: If using a local LLM (e.g., LM Studio, Ollama), set `OPENAI_BASE_URL` to your local endpoint. The `OPENAI_API_KEY` can be any string for local models.

### Running the Application

```bash
python -m src.main
```

Or if installed as a package:
```bash
python -m src
```

### Usage

Once running, you can interact with the weather assistant:

```
Weather Assistant - ask about the weather in any city.
Type 'quit' to exit.

You: What's the weather in London, Paris, and Berlin?
[Calling get_weather for London...]
[Calling get_weather for Paris...]
[Calling get_weather for Berlin...]
Assistant: Here's the current weather in each city:
...
```

Type `quit` or `exit` to exit the application.

## Design Decisions

### 1. Separation of Concerns

The codebase is modularized to ensure maintainability and testability:

- **assistant.py**: Manages the conversation state, the LLM interaction loop, and tool orchestration.

- **weather_service.py**: A dedicated layer for external API interactions (Open-Meteo). It handles HTTP requests, timeouts, and DTO conversion.

- **cli.py**: Handles User I/O. Crucially, it isolates the input mechanism from the business logic.

- **models.py**: Uses Python dataclasses to enforce type safety across the application, avoiding "dictionary passing hell."

### 2. Async/Await & Concurrency

**Non-blocking Input**: Standard `input()` is blocking. I implemented `loop.run_in_executor` in `cli.py` to offload user input to a separate thread. This ensures the main asyncio event loop remains alive (e.g., for potential background keep-alives or UI spinners).

**Parallel Fetching**: If a user asks for weather in multiple cities (e.g., "London, Paris, and Berlin"), the `WeatherAssistant` executes these requests concurrently using `asyncio.gather`, significantly reducing latency compared to sequential requests.

### 3. Tool Calling Strategy

**Batch Processing**: The `get_weather` tool is defined to accept a list of cities (`cities: array[string]`) rather than a single city. This optimization allows the LLM to request data for multiple locations in a single turn, preventing unnecessary network round-trips.

**Robust Parsing**: Added defensive logic (using `ast.literal_eval` fallback) to parse tool arguments. This ensures compatibility with smaller local models (like Llama 3 or Qwen) that might occasionally produce malformed JSON strings for arrays.

### 4. True Streaming

The application implements "true streaming." Text tokens are printed to the console immediately as they arrive from the API.

Tool calls are accumulated in a buffer during the stream and executed only when the tool call block is complete. This provides a responsive UX without breaking the logic of function calling.

### 5. Local LLM Compatibility

The code is designed to be vendor-agnostic. While it defaults to `gpt-4o`, it supports any OpenAI-compatible endpoint (e.g., LM Studio, Ollama).
