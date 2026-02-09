import requests
import csv
import time
from datetime import datetime, timedelta

# Cities with their coordinates
CITIES = {
    "Dallas":      (32.7767, -96.7970),
    "San Antonio": (29.4241, -98.4936),
    "Houston":     (29.7604, -95.3698),
    "Austin":      (30.2672, -97.7431),
    "NYC":         (40.7128, -74.0060),
    "Los Angeles": (34.0522, -118.2437),
}

# Date range: last 2 years
END_DATE = datetime(2026, 2, 9)
START_DATE = END_DATE - timedelta(days=730)  # ~2 years

# Open-Meteo archive API (free, no key needed)
# Archive data may lag ~5 days behind present; we'll also hit the forecast API for recent days.
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARS = "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure"

# Archive data typically available up to ~5 days ago
ARCHIVE_END = (END_DATE - timedelta(days=6)).strftime("%Y-%m-%d")
FORECAST_START = (END_DATE - timedelta(days=5)).strftime("%Y-%m-%d")


def fetch_city_data(city_name, lat, lon):
    """Fetch hourly weather data for a city from both archive and forecast APIs."""
    rows = []

    # --- Archive (bulk historical) ---
    params_archive = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE.strftime("%Y-%m-%d"),
        "end_date": ARCHIVE_END,
        "hourly": HOURLY_VARS,
        "timezone": "auto",
    }
    print(f"  Fetching archive data for {city_name} ({START_DATE.strftime('%Y-%m-%d')} to {ARCHIVE_END})...")
    resp = requests.get(ARCHIVE_URL, params=params_archive, timeout=120)
    resp.raise_for_status()
    archive = resp.json()

    hourly = archive["hourly"]
    for i, ts in enumerate(hourly["time"]):
        rows.append({
            "city": city_name,
            "datetime": ts,
            "temperature_c": hourly["temperature_2m"][i],
            "relative_humidity_pct": hourly["relative_humidity_2m"][i],
            "wind_speed_kmh": hourly["wind_speed_10m"][i],
            "surface_pressure_hpa": hourly["surface_pressure"][i],
        })

    # --- Forecast / recent days ---
    params_forecast = {
        "latitude": lat,
        "longitude": lon,
        "start_date": FORECAST_START,
        "end_date": END_DATE.strftime("%Y-%m-%d"),
        "hourly": HOURLY_VARS,
        "timezone": "auto",
    }
    print(f"  Fetching recent data for {city_name} ({FORECAST_START} to {END_DATE.strftime('%Y-%m-%d')})...")
    resp2 = requests.get(FORECAST_URL, params=params_forecast, timeout=60)
    resp2.raise_for_status()
    forecast = resp2.json()

    hourly2 = forecast["hourly"]
    for i, ts in enumerate(hourly2["time"]):
        rows.append({
            "city": city_name,
            "datetime": ts,
            "temperature_c": hourly2["temperature_2m"][i],
            "relative_humidity_pct": hourly2["relative_humidity_2m"][i],
            "wind_speed_kmh": hourly2["wind_speed_10m"][i],
            "surface_pressure_hpa": hourly2["surface_pressure"][i],
        })

    return rows


def main():
    all_rows = []
    for city_name, (lat, lon) in CITIES.items():
        print(f"\n[{city_name}]")
        city_rows = fetch_city_data(city_name, lat, lon)
        all_rows.extend(city_rows)
        print(f"  -> {len(city_rows):,} hourly records")
        time.sleep(1)  # be polite to the free API

    # Write CSV
    out_path = "/Users/allenchen/weatherpedagogy/hourly_weather_data.csv"
    fieldnames = [
        "city", "datetime", "temperature_c",
        "relative_humidity_pct", "wind_speed_kmh", "surface_pressure_hpa",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone! Wrote {len(all_rows):,} rows to {out_path}")


if __name__ == "__main__":
    main()
