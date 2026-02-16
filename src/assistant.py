"""Weather assistant for managing LLM interactions and tool calls."""

import ast
import asyncio
import json
from typing import Any, Dict, List, Optional

from openai import APITimeoutError, AsyncOpenAI

from .config import (
    CLI_ASSISTANT_PROMPT,
    CLI_UNEXPECTED_ERROR,
    MODEL_NAME,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    SYSTEM_PROMPT,
    WEATHER_TOOL_NAME,
)
from .models import (
    CityWeatherResult,
    InvokedFunction,
    Message,
    MessageRole,
    OpenAIClientParams,
    ParsedToolArguments,
    ToolCallResponse,
    ToolDefinition,
    ToolFunctionDefinition,
    ToolInvocation,
    WeatherToolParams,
    WeatherToolResult,
)
from .weather_service import WeatherService


class WeatherAssistant:
    """
    Manages the conversation state and interaction with the OpenAI API.
    Handles tool calls, streaming responses, and conversation flow.
    """

    def __init__(self):
        """Initialize the weather assistant with LLM client and tools."""
        client_params = OpenAIClientParams(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        self.client = AsyncOpenAI(**client_params.to_dict())

        self.weather_service = WeatherService()

        sys_msg = Message(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT)
        self.conversation_history: List[Message] = [sys_msg]

        self._init_tools()

    def _init_tools(self):
        """Define available tools for the LLM."""
        tool_function = ToolFunctionDefinition(
            name=WEATHER_TOOL_NAME,
            description="Get the current weather for one or multiple cities if user asked.",
            parameters={
                "type": "object",
                "properties": {
                    "cities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Get the current weather ONLY if the user explicitly asks for a city. Do not use for general greetings.",
                    }
                },
                "required": ["cities"],
            },
        )
        self.tools = [ToolDefinition(function=tool_function).to_dict()]

    async def process_message(self, user_input: str) -> None:
        """
        Process a user message and handle the entire conversation turn.
        """
        self.conversation_history.append(Message(role=MessageRole.USER, content=user_input))
        await self._run_interaction_loop()

    async def _run_interaction_loop(self) -> None:
        """
        The main loop that handles multi-turn reasoning (Tool calling cycle).
        """
        while True:
            try:
                stream = await self._create_api_stream()
                if stream is None:
                    break

                content, tool_calls_buffer = await self._process_stream(stream)

                should_continue = await self._handle_stream_response(content, tool_calls_buffer)
                if not should_continue:
                    break

            except Exception as e:
                print(CLI_UNEXPECTED_ERROR.format(error=e))
                break

    async def _create_api_stream(self) -> Optional[Any]:
        """
        Create API stream request with timeout handling.

        Returns:
            Stream object or None if timeout occurred
        """
        try:
            return await self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[msg.to_dict() for msg in self.conversation_history],
                tools=self.tools,
                tool_choice="auto",
                stream=True,
            )
        except APITimeoutError as e:
            print(f"\n[System Error] API request timeout: {e}")
            return None

    async def _process_stream(self, stream: Any) -> tuple[str, Dict[int, ToolInvocation]]:
        """
        Process stream chunks and collect content and tool calls.

        Args:
            stream: OpenAI API stream

        Returns:
            Tuple of (collected_content, tool_calls_buffer)
        """
        print(CLI_ASSISTANT_PROMPT, end="", flush=True)

        collected_content = ""
        tool_calls_buffer: Dict[int, ToolInvocation] = {}

        async for chunk in stream:
            delta = chunk.choices[0].delta

            if delta.content:
                content_chunk = delta.content
                print(content_chunk, end="", flush=True)
                collected_content += content_chunk

            if delta.tool_calls:
                self._accumulate_tool_call_chunk(delta.tool_calls, tool_calls_buffer)

        print()
        return collected_content, tool_calls_buffer

    def _accumulate_tool_call_chunk(
        self, tool_calls_delta, tool_calls_buffer: Dict[int, ToolInvocation]
    ) -> None:
        """
        Accumulate tool call chunks into buffer.

        Args:
            tool_calls_delta: Tool calls delta from stream chunk
            tool_calls_buffer: Buffer to accumulate tool call data
        """
        for tc in tool_calls_delta:
            idx = tc.index
            if idx not in tool_calls_buffer:
                tool_calls_buffer[idx] = ToolInvocation()

            if tc.id:
                tool_calls_buffer[idx].id += tc.id
            if tc.function.name:
                tool_calls_buffer[idx].function.name += tc.function.name
            if tc.function.arguments:
                tool_calls_buffer[idx].function.arguments += tc.function.arguments

    def _convert_tool_calls_buffer(
        self, tool_calls_buffer: Dict[int, ToolInvocation]
    ) -> List[ToolInvocation]:
        """
        Convert tool calls buffer to list of ToolInvocation dataclasses.

        Args:
            tool_calls_buffer: Buffer with accumulated tool invocation data

        Returns:
            List of ToolInvocation dataclasses
        """
        return [tool_calls_buffer[idx] for idx in sorted(tool_calls_buffer.keys())]

    async def _handle_stream_response(
        self, content: str, tool_calls_buffer: Dict[int, ToolInvocation]
    ) -> bool:
        """
        Handle stream response: either tool calls or final text response.

        Args:
            content: Collected text content from stream
            tool_calls_buffer: Buffer with tool calls data

        Returns:
            True if should continue loop, False if should break
        """
        if tool_calls_buffer:
            tool_calls_list = self._convert_tool_calls_buffer(tool_calls_buffer)

            assistant_msg = Message(
                role=MessageRole.ASSISTANT,
                content=content if content else None,
                tool_calls=[tc.to_history_dict() for tc in tool_calls_list],
            )
            self.conversation_history.append(assistant_msg)

            await self._handle_tool_calls(tool_calls_list)
            return True
        else:
            assistant_msg = Message(
                role=MessageRole.ASSISTANT, content=content if content else None
            )
            self.conversation_history.append(assistant_msg)
            return False

    async def _handle_tool_calls(self, tool_calls: List[ToolInvocation]) -> None:
        """
        Execute tool call and update chat history.

        Args:
            tool_calls: List of tool invocations to execute
        """
        if len(tool_calls) == 0:
            return

        # Expect only 1 tool call which includes all city names
        tool_call = tool_calls[0]
        tool_output_dict = await self._execute_tool_call(tool_call)
        tool_msg = Message(
            role=tool_output_dict["role"],
            tool_call_id=tool_output_dict.get("tool_call_id"),
            name=tool_output_dict.get("name"),
            content=tool_output_dict.get("content", ""),
        )
        self.conversation_history.append(tool_msg)

    def _parse_tool_arguments(self, arguments_str: str) -> ParsedToolArguments:
        """
        Parse tool call arguments from JSON string.
        Handles special cases like string representations of arrays.
        """
        try:
            arguments = json.loads(arguments_str)
        except (json.JSONDecodeError, AttributeError):
            arguments = {}

        parsed_args = ParsedToolArguments(raw_arguments=arguments)

        # Handle case when cities comes as a string representation of array
        if "cities" in arguments and isinstance(arguments["cities"], str):
            try:
                parsed_args.cities = json.loads(arguments["cities"])
            except (json.JSONDecodeError, ValueError):
                # If JSON fails, try to parse as Python literal (e.g., "['London']")
                try:
                    parsed = ast.literal_eval(arguments["cities"])
                    if isinstance(parsed, list):
                        parsed_args.cities = parsed
                    else:
                        # If it's a single string, wrap it in a list
                        parsed_args.cities = [parsed] if parsed else []
                except (ValueError, SyntaxError):
                    # If all parsing fails, treat as single city name
                    parsed_args.cities = [arguments["cities"]]
        elif "cities" in arguments and isinstance(arguments["cities"], list):
            parsed_args.cities = arguments["cities"]

        return parsed_args

    async def _execute_tool_call(self, tool_call) -> Dict[str, Any]:
        """Executes the specific tool requested by the LLM."""
        func_name = tool_call.function.name

        parsed_args = self._parse_tool_arguments(tool_call.function.arguments)

        if func_name == WEATHER_TOOL_NAME:
            params = WeatherToolParams(call_id=tool_call.id, arguments=parsed_args.to_dict())
            return await self._run_weather_tool(params)

        return ToolCallResponse(
            tool_call_id=tool_call.id,
            name=func_name,
            content="Error: Unknown function requested.",
        ).to_dict()

    async def _run_weather_tool(self, params: WeatherToolParams) -> Dict[str, Any]:
        """Helper to run the weather logic specifically."""
        cities = self._validate_and_normalize_cities(params.arguments)

        if not cities:
            return self._create_error_response(
                params.call_id, "Error: No cities provided."
            )

        city_results = await self._fetch_weather_for_cities(cities)
        content = self._format_weather_content(city_results)

        result = WeatherToolResult(call_id=params.call_id, name=WEATHER_TOOL_NAME, content=content)
        return result.to_tool_response()

    def _validate_and_normalize_cities(self, arguments: Dict[str, Any]) -> List[str]:
        """
        Validate and normalize cities list from arguments.

        Args:
            arguments: Tool call arguments dictionary

        Returns:
            Normalized list of city names
        """
        cities = arguments.get("cities", [])

        if not isinstance(cities, list):
            cities = [cities] if cities else []

        return cities

    def _create_error_response(self, call_id: str, error_message: str) -> Dict[str, Any]:
        """
        Create an error response for tool execution.

        Args:
            call_id: Tool call ID
            error_message: Error message to include

        Returns:
            Tool response dictionary with error
        """
        result = WeatherToolResult(
            call_id=call_id,
            name=WEATHER_TOOL_NAME,
            content=error_message,
        )
        return result.to_tool_response()

    async def _fetch_weather_for_cities(
        self, cities: List[str]
    ) -> List[CityWeatherResult]:
        """
        Fetch weather data for multiple cities.

        Args:
            cities: List of city names

        Returns:
            List of CityWeatherResult objects
        """
        city_tasks = [self._get_weather_for_city(city) for city in cities]
        city_results = await asyncio.gather(*city_tasks, return_exceptions=True)

        return self._process_city_results(cities, city_results)

    def _process_city_results(
        self, cities: List[str], results: List[Any]
    ) -> List[CityWeatherResult]:
        """
        Process city results, converting exceptions to error results.

        Args:
            cities: Original list of city names
            results: Results from asyncio.gather (may include exceptions)

        Returns:
            List of CityWeatherResult objects (no exceptions)
        """
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    CityWeatherResult(
                        city=cities[i],
                        error=f"Error processing city '{cities[i]}': {str(result)}",
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    def _format_weather_content(self, results: List[CityWeatherResult]) -> str:
        """
        Format weather results into a single content string.

        Args:
            results: List of CityWeatherResult objects

        Returns:
            Formatted content string
        """
        return "\n\n".join([result.to_string() for result in results])

    async def _get_weather_for_city(self, city: str) -> CityWeatherResult:
        """Get weather for a single city string."""
        print(f"[Calling {WEATHER_TOOL_NAME} for {city}...]")

        # 1. Geocoding
        location = await self.weather_service.get_coordinates(city)
        if not location:
            return CityWeatherResult(
                city=city, error=f"Error: Could not find coordinates for city '{city}'."
            )

        # 2. Weather Fetch
        try:
            weather_data = await self.weather_service.get_current_weather(
                location.latitude, location.longitude
            )
            return CityWeatherResult(city=city, location=location, weather_data=weather_data)
        except Exception as e:
            return CityWeatherResult(
                city=city, error=f"Error fetching weather data for {city}: {str(e)}"
            )

    async def close(self):
        """Clean up resources."""
        await self.client.close()
        await self.weather_service.close()
