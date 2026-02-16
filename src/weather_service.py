"""Weather service for interacting with Open-Meteo API."""

from typing import Optional

import httpx

from .config import OPEN_METEO_GEO_URL, OPEN_METEO_WEATHER_URL
from .models import (
    GeocodingParams,
    HTTPClientConfig,
    Location,
    WeatherData,
    WeatherParams,
)


class WeatherService:
    """
    Handles interactions with the Open-Meteo API.
    Responsible for geocoding cities and fetching weather data.
    """

    def __init__(self, client_config: HTTPClientConfig = None):
        """Initialize the weather service"""
        if client_config is None:
            client_config = HTTPClientConfig()
        self.client = httpx.AsyncClient(
            timeout=client_config.to_httpx_timeout(),
            limits=client_config.to_httpx_limits(),
        )

    async def close(self):
        """Close the HTTP client and clean up resources."""
        await self.client.aclose()

    async def get_coordinates(self, city_name: str) -> Optional[Location]:
        """
        Fetches latitude and longitude for a given city name.

        Args:
            city_name: Name of the city to geocode

        Returns:
            Location dataclass with coordinates, name, and country, or None if not found
        """
        params = GeocodingParams(name=city_name)

        try:
            response = await self.client.get(OPEN_METEO_GEO_URL, params=params.to_dict())
            response.raise_for_status()
            data = response.json()

            if not data.get("results"):
                return None

            location_data = data["results"][0]
            return Location(
                latitude=location_data["latitude"],
                longitude=location_data["longitude"],
                name=location_data["name"],
                country=location_data.get("country", ""),
            )
        except httpx.TimeoutException as e:
            print(f"\n[System Error] Geocoding API timeout: {e}")
            return None
        except httpx.HTTPError as e:
            print(f"\n[System Error] Geocoding API failed: {e}")
            return None

    async def get_current_weather(self, latitude: float, longitude: float) -> WeatherData:
        """
        Fetches current weather data using coordinates.

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate

        Returns:
            WeatherData dataclass with weather information

        Raises:
            httpx.HTTPError: If the API request fails
        """
        params = WeatherParams(latitude=latitude, longitude=longitude)

        try:
            response = await self.client.get(OPEN_METEO_WEATHER_URL, params=params.to_dict())
            response.raise_for_status()
            data = response.json()

            current = data.get("current", {})
            units = data.get("current_units", {})

            return WeatherData(
                temperature=current.get("temperature_2m", 0.0),
                apparent_temperature=current.get("apparent_temperature", 0.0),
                humidity=current.get("relative_humidity_2m", 0.0),
                wind_speed=current.get("wind_speed_10m", 0.0),
                precipitation=current.get("precipitation", 0.0),
                weather_code=current.get("weather_code"),
                temperature_unit=units.get("temperature_2m", "°C"),
                humidity_unit=units.get("relative_humidity_2m", "%"),
                wind_speed_unit=units.get("wind_speed_10m", "km/h"),
                precipitation_unit=units.get("precipitation", "mm"),
            )
        except httpx.TimeoutException as e:
            raise ValueError(f"Timeout while fetching weather data: {str(e)}") from e
        except httpx.HTTPError as e:
            raise ValueError(f"Error fetching weather data: {str(e)}") from e
