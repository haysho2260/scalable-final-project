
"""Fetch historical daily weather for Los Angeles and save to CSV."""

from datetime import date
import os
from pathlib import Path

import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# Los Angeles, CA coordinates
LATITUDE = 34.053345
LONGITUDE = 118.2437
CDD_HDD_BASE_C = 18.0  # common base for degree days


def _temperature_dir() -> Path:
    """Return path to the temperature data directory."""
    return Path(__file__).resolve().parent / "temperature"


def _read_month_file(path: Path) -> pd.DataFrame:
    """Read a monthly weather CSV if it exists."""
    return pd.read_csv(path, parse_dates=["date"]) if path.exists() else pd.DataFrame()


def _load_existing_months(base_dir: Path) -> pd.DataFrame:
    """Load and combine all monthly CSVs already stored."""
    frames = []
    for csv_path in sorted(base_dir.glob("la_daily_weather_*.csv")):
        frames.append(_read_month_file(csv_path))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if "date" in combined:
        combined["date"] = pd.to_datetime(combined["date"])
        combined = combined.drop_duplicates(subset="date").sort_values("date")
    return combined.reset_index(drop=True)


def _append_monthly(base_dir: Path, df: pd.DataFrame) -> None:
    """Write/append rows into per-month CSVs under base_dir."""
    if df.empty:
        return
    df["date"] = pd.to_datetime(df["date"])
    for period, month_df in df.groupby(df["date"].dt.to_period("M")):
        month_name = f"{period.year}_{period.month:02d}"
        csv_path = base_dir / f"la_daily_weather_{month_name}.csv"
        existing = _read_month_file(csv_path)
        combined = pd.concat([existing, month_df], ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"])
        combined = combined.drop_duplicates(subset="date").sort_values("date")
        combined.to_csv(csv_path, index=False)


def get_historical_weather(
    start_date: str = "2019-01-01",
    end_date: str | None = None,
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
    save_path: str | None = None,
    cdd_hdd_base_c: float = CDD_HDD_BASE_C,
) -> pd.DataFrame:
    """Download daily weather features starting Jan 1 2024 and save to monthly CSVs."""

    # Default end date is today in ISO format
    if end_date is None:
        end_date = date.today().isoformat()

    base_dir = _temperature_dir()
    base_dir.mkdir(parents=True, exist_ok=True)

    # Use existing monthly files to determine the next start date
    existing = _load_existing_months(base_dir)
    latest_local_date = (
        existing["date"].max() if not existing.empty else pd.NaT
    )
    fetch_start_dt = pd.to_datetime(start_date)
    if pd.notna(latest_local_date):
        fetch_start_dt = max(
            fetch_start_dt, latest_local_date + pd.Timedelta(days=1))

    end_date_dt = pd.to_datetime(end_date)
    if fetch_start_dt > end_date_dt:
        # Already up to date; return combined existing data
        return existing.reset_index(drop=True)

    fetch_start = fetch_start_dt.date().isoformat()

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # The order of variables in "daily" must match extraction order below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": fetch_start,
        "end_date": end_date,
        "daily": [
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "apparent_temperature_mean",
            "apparent_temperature_max",
            "apparent_temperature_min",
            "dew_point_2m_mean",
            "relative_humidity_2m_mean",
            "relative_humidity_2m_max",
            "relative_humidity_2m_min",
            "precipitation_sum",
            "rain_sum",
            "snowfall_sum",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "shortwave_radiation_sum",
            "cloudcover_mean",
            "et0_fao_evapotranspiration",
            "pressure_msl_mean",
            "weathercode",
        ],
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()

    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s"),
            end=pd.to_datetime(daily.TimeEnd(), unit="s"),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left",
        ),
        # mean 2m temp (°C)
        "temperature_2m_mean": daily.Variables(0).ValuesAsNumpy(),
        # max 2m temp (°C)
        "temperature_2m_max": daily.Variables(1).ValuesAsNumpy(),
        # min 2m temp (°C)
        "temperature_2m_min": daily.Variables(2).ValuesAsNumpy(),
        # mean feels-like (°C)
        "apparent_temperature_mean": daily.Variables(3).ValuesAsNumpy(),
        # max feels-like (°C)
        "apparent_temperature_max": daily.Variables(4).ValuesAsNumpy(),
        # min feels-like (°C)
        "apparent_temperature_min": daily.Variables(5).ValuesAsNumpy(),
        # mean dew point (°C)
        "dew_point_2m_mean": daily.Variables(6).ValuesAsNumpy(),
        # mean RH (%)
        "relative_humidity_2m_mean": daily.Variables(7).ValuesAsNumpy(),
        # max RH (%)
        "relative_humidity_2m_max": daily.Variables(8).ValuesAsNumpy(),
        # min RH (%)
        "relative_humidity_2m_min": daily.Variables(9).ValuesAsNumpy(),
        # total precip (mm)
        "precipitation_sum": daily.Variables(10).ValuesAsNumpy(),
        # rain-only precip (mm)
        "rain_sum": daily.Variables(11).ValuesAsNumpy(),
        # snowfall (cm water eq.)
        "snowfall_sum": daily.Variables(12).ValuesAsNumpy(),
        # max wind speed 10m (m/s)
        "wind_speed_10m_max": daily.Variables(13).ValuesAsNumpy(),
        # max gust 10m (m/s)
        "wind_gusts_10m_max": daily.Variables(14).ValuesAsNumpy(),
        # solar shortwave (MJ/m²)
        "shortwave_radiation_sum": daily.Variables(15).ValuesAsNumpy(),
        # mean cloud cover (%)
        "cloudcover_mean": daily.Variables(16).ValuesAsNumpy(),
        # ET0 (mm)
        "et0_fao_evapotranspiration": daily.Variables(17).ValuesAsNumpy(),
        # mean sea-level pressure (hPa)
        "pressure_msl_mean": daily.Variables(18).ValuesAsNumpy(),
        # weather condition code
        "weathercode": daily.Variables(19).ValuesAsNumpy(),
    }

    df = pd.DataFrame(data=daily_data)

    # 1. Enforce Continuity & 2. Interpolate Missing Values
    # Create complete date range
    if not df.empty:
        df = df.sort_values('date')
        full_idx = pd.date_range(
            start=df['date'].min(), end=df['date'].max(), freq='D')
        df = df.set_index('date').reindex(full_idx)
        df.index.name = 'date'
        df = df.reset_index()

        # Interpolate numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(
            method='linear', limit_direction='both')

        # Fill remaining NaNs (if any at edges) with ffill/bfill or 0
        df[numeric_cols] = df[numeric_cols].ffill().bfill().fillna(0)

    # 3. Sanity Checks / Clipping
    # Humidity [0, 100]
    rh_cols = [c for c in df.columns if 'relative_humidity' in c]
    for c in rh_cols:
        df[c] = df[c].clip(lower=0, upper=100)

    # Radiation >= 0
    rad_cols = [c for c in df.columns if 'radiation' in c]
    for c in rad_cols:
        df[c] = df[c].clip(lower=0)

    # Degree days are strong predictors for load
    df["cdd"] = np.clip(df["temperature_2m_mean"] -
                        cdd_hdd_base_c, a_min=0, a_max=None)
    df["hdd"] = np.clip(
        cdd_hdd_base_c - df["temperature_2m_mean"], a_min=0, a_max=None)

    # Save/append monthly CSVs
    _append_monthly(base_dir, df)

    # Return combined dataset (existing + new) sorted by date
    combined = pd.concat([existing, df], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.drop_duplicates(subset="date").sort_values("date")

    # Final check on combined
    combined_idx = pd.date_range(
        start=combined['date'].min(), end=combined['date'].max(), freq='D')
    if len(combined) != len(combined_idx):
        # Re-enforce continuity on master
        combined = combined.set_index('date').reindex(combined_idx)
        combined.index.name = 'date'
        combined = combined.reset_index()
        combined[numeric_cols] = combined[numeric_cols].interpolate(
            method='linear', limit_direction='both')

    # Write date as ISO for downstream consistency
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
    combined.to_csv(base_dir / "la_daily_weather_all.csv", index=False)
    return combined.reset_index(drop=True)


if __name__ == "__main__":
    df = get_historical_weather()
    print(
        f"Saved {len(df)} rows to monthly CSVs under {Path(__file__).resolve().parent / 'temperature'}"
    )
