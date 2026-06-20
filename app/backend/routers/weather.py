from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import urlopen

from fastapi import APIRouter, Query

router = APIRouter(prefix="/v1/weather", tags=["weather"])

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Fog with frost",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    80: "Slight showers",
    81: "Moderate showers",
    82: "Violent showers",
    95: "Thunderstorm",
}


@router.get("/current")
def current_weather(lat: float = Query(...), lon: float = Query(...)):
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "precipitation",
            "rain",
            "weather_code",
            "cloud_cover",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
        ]),
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
        "timezone": "auto",
        "forecast_days": 1,
    }
    url = f"{OPEN_METEO_URL}?{urlencode(params)}"

    try:
        with urlopen(url, timeout=6) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {
            "source": "Open-Meteo",
            "status": "unavailable",
            "error": str(exc),
            "query": {"lat": lat, "lon": lon},
        }

    current = payload.get("current", {})
    units = payload.get("current_units", {})
    code = current.get("weather_code")
    return {
        "source": "Open-Meteo",
        "status": "ok",
        "query": {"lat": lat, "lon": lon},
        "time": current.get("time"),
        "temperature_C": current.get("temperature_2m"),
        "feels_like_C": current.get("apparent_temperature"),
        "humidity_pct": current.get("relative_humidity_2m"),
        "wind_speed_m_s": current.get("wind_speed_10m"),
        "wind_direction_deg": current.get("wind_direction_10m"),
        "wind_gust_m_s": current.get("wind_gusts_10m"),
        "pressure_hpa": current.get("surface_pressure"),
        "cloud_cover_pct": current.get("cloud_cover"),
        "precipitation_mm": current.get("precipitation"),
        "rain_mm": current.get("rain"),
        "weather_code": code,
        "condition": WEATHER_CODES.get(code, "Current weather"),
        "units": units,
    }
