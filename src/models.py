import httpx
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Location:
    """Represents a geographic location with coordinates."""

    latitude: float
    longitude: float
    name: str
    country: str = ""

    def __str__(self) -> str:
        """String representation of the location."""
        if self.country:
            return f"{self.name}, {self.country}"
        return self.name


@dataclass
class WeatherData:
    """Represents current weather data with units."""

    temperature: float
    apparent_temperature: float
    humidity: float
    wind_speed: float
    precipitation: float
    weather_code: Optional[int] = None

    # Units
    temperature_unit: str = "°C"
    humidity_unit: str = "%"
    wind_speed_unit: str = "km/h"
    precipitation_unit: str = "mm"

    def format_for_llm(self) -> str:
        """Format weather data as a string for LLM consumption."""
        return (
            f"Temperature: {self.temperature} {self.temperature_unit}\n"
            f"Feels like: {self.apparent_temperature} {self.temperature_unit}\n"
            f"Humidity: {self.humidity} {self.humidity_unit}\n"
            f"Wind Speed: {self.wind_speed} {self.wind_speed_unit}\n"
            f"Precipitation: {self.precipitation} {self.precipitation_unit}"
        )


@dataclass
class WeatherResponse:
    """Complete weather response with location and weather data."""

    location: Location
    weather: WeatherData

    def format_for_llm(self) -> str:
        """Format complete weather response for LLM."""
        return f"Weather for {self.location}:\n{self.weather.format_for_llm()}"


class MessageRole:
    """Constants for message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """Represents a conversation message."""

    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for OpenAI API."""
        if self.role == MessageRole.ASSISTANT:
            has_content = self.content is not None and self.content != ""
            has_tool_calls = self.tool_calls is not None and len(self.tool_calls) > 0
            if not has_content and not has_tool_calls:
                raise ValueError(
                    "Assistant message must have either content or tool_calls"
                )

        result: Dict[str, Any] = {"role": self.role}

        if self.content is not None:
            result["content"] = self.content

        if self.tool_calls is not None:
            result["tool_calls"] = self.tool_calls

        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id

        if self.name is not None:
            result["name"] = self.name

        return result


@dataclass
class ToolFunctionDefinition:
    """Definition of a tool function (schema sent to LLM)."""

    name: str
    description: str
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass
class ToolDefinition:
    """Definition of a tool (schema sent to LLM)."""

    type: str = "function"
    function: Optional[ToolFunctionDefinition] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI tool format."""
        if self.function is None:
            raise ValueError("Tool function must be defined")
        return {
            "type": self.type,
            "function": self.function.to_dict(),
        }


@dataclass
class ToolCallResponse:
    """Response from a tool call."""

    tool_call_id: str
    role: str = MessageRole.TOOL
    name: str = ""
    content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "tool_call_id": self.tool_call_id,
            "role": self.role,
            "name": self.name,
            "content": self.content,
        }


@dataclass
class HTTPClientConfig:
    """HTTP client configuration."""

    timeout_seconds: float = 10.0
    connect_timeout_seconds: float = 5.0
    max_keepalive_connections: int = 5
    max_connections: int = 10

    def to_httpx_timeout(self):
        """Convert to httpx.Timeout."""

        return httpx.Timeout(self.timeout_seconds, connect=self.connect_timeout_seconds)

    def to_httpx_limits(self):
        """Convert to httpx.Limits."""

        return httpx.Limits(
            max_keepalive_connections=self.max_keepalive_connections,
            max_connections=self.max_connections,
        )


@dataclass
class GeocodingParams:
    """Parameters for geocoding API request."""

    name: str
    count: int = 1
    language: str = "en"
    format: str = "json"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API request."""
        return {
            "name": self.name,
            "count": self.count,
            "language": self.language,
            "format": self.format,
        }


@dataclass
class WeatherParams:
    """Parameters for weather API request."""

    latitude: float
    longitude: float
    current: List[str] = field(
        default_factory=lambda: [
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "precipitation",
            "weather_code",
            "wind_speed_10m",
        ]
    )
    timezone: str = "auto"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API request."""
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "current": self.current,
            "timezone": self.timezone,
        }


@dataclass
class WeatherToolParams:
    """Parameters for weather tool execution."""

    call_id: str
    arguments: Dict[str, Any]


@dataclass
class OpenAIClientParams:
    """Parameters for OpenAI client initialization."""

    api_key: str
    base_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for client initialization."""
        result = {"api_key": self.api_key}
        if self.base_url:
            result["base_url"] = self.base_url
        return result


@dataclass
class CityWeatherResult:
    """Result of weather fetch for a single city."""

    city: str
    location: Optional[Location] = None
    weather_data: Optional[WeatherData] = None
    error: Optional[str] = None

    def to_string(self) -> str:
        """Convert to formatted string for tool response."""
        if self.error:
            return self.error
        if self.location and self.weather_data:
            return f"Weather for {self.location}:\n{self.weather_data.format_for_llm()}"
        return f"Error: Incomplete data for city '{self.city}'."


@dataclass
class WeatherToolResult:
    """Result of weather tool execution."""

    call_id: str
    name: str
    content: str

    def to_tool_response(self) -> Dict[str, Any]:
        """Convert to ToolCallResponse format."""
        return ToolCallResponse(
            tool_call_id=self.call_id, name=self.name, content=self.content
        ).to_dict()


@dataclass
class ParsedToolArguments:
    """Parsed tool call arguments with type safety."""

    cities: List[str] = field(default_factory=list)
    raw_arguments: Dict[str, Any] = field(default_factory=dict)

    def get_cities(self) -> List[str]:
        """Get cities list, ensuring it's always a list."""
        if isinstance(self.raw_arguments.get("cities"), list):
            return self.raw_arguments["cities"]
        return self.cities

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = self.raw_arguments.copy()
        if self.cities:
            result["cities"] = self.cities
        return result


@dataclass
class InvokedFunction:
    """Function data from an actual tool invocation (returned by LLM)."""

    name: str = ""
    arguments: str = ""

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format."""
        return {"name": self.name, "arguments": self.arguments}


@dataclass
class ToolInvocation:
    """Actual tool invocation returned by LLM in response."""

    id: str = ""
    function: InvokedFunction = None

    def __post_init__(self):
        """Initialize function if not provided."""
        if self.function is None:
            self.function = InvokedFunction()

    def to_history_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for conversation history."""
        return {
            "id": self.id,
            "type": "function",
            "function": self.function.to_dict(),
        }
