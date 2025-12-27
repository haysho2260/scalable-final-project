"""Train tree-based models for residential spending prediction (daily & monthly).

This script builds a unified dataset from feature drops:
- energy_types (generation mix, hourly)
- hourly_load (system load)
- hourly_price (price + cost)
- lag_load (load lags)
- lag_prices (price/cost lags)
- temperature (daily weather with degree days)

Outputs:
- model/daily_spend_model.pkl
- model/monthly_spend_model.pkl
- simple evaluation metrics printed to stdout
"""

from __future__ import annotations

import argparse
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from typing import Iterable, Any
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT / "features"


# -------------- Loaders --------------
def _read_many(directory: Path, prefix: str) -> pd.DataFrame:
    """Concatenate CSVs sharing a prefix."""
    frames = []
    for csv_path in sorted(directory.glob(f"{prefix}_*.csv")):
        frames.append(pd.read_csv(csv_path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_lag_prices() -> pd.DataFrame:
    df = _read_many(FEATURES_DIR / "lag_prices", "CAISO_Price")
    if df.empty:
        return df
    df["HE_num"] = pd.to_numeric(df["HE"], errors="coerce")
    df = df.dropna(subset=["Date", "HE_num"])
    df["timestamp"] = pd.to_datetime(
        df["Date"]) + pd.to_timedelta(df["HE_num"] - 1, unit="h")
    df = df.sort_values("timestamp")
    return df


def load_lag_load() -> pd.DataFrame:
    df = _read_many(FEATURES_DIR / "lag_load", "CAISO_Load")
    if df.empty:
        return df
    # Handle Date column - may be in YYYY-MM-DD or YYYY-MM-DD HH:MM:SS format
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    df["HE_num"] = pd.to_numeric(df["HE"], errors="coerce")
    df = df.dropna(subset=["Date", "HE_num"])
    df["timestamp"] = pd.to_datetime(
        df["Date"]) + pd.to_timedelta(df["HE_num"] - 1, unit="h")
    df = df.sort_values("timestamp")
    return df


def load_energy_mix() -> pd.DataFrame:
    df = _read_many(FEATURES_DIR / "energy_types", "Fuel_Type")
    if df.empty:
        return df
    # period may be in various datetime string formats; parse robustly
    df["timestamp"] = pd.to_datetime(df["period"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.drop(columns=["period"])
    return df.reset_index(drop=True)


def load_temperature_daily() -> pd.DataFrame:
    frames = []
    for csv_path in sorted((FEATURES_DIR / "temperature").glob("la_daily_weather_*.csv")):
        frames.append(pd.read_csv(csv_path, parse_dates=["date"]))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)


# -------------- Feature assembly --------------
def build_hourly_dataset() -> pd.DataFrame:
    price = load_lag_prices()
    if price.empty:
        raise FileNotFoundError(
            "No lag price files found under features/lag_prices")

    load = load_lag_load()
    mix = load_energy_mix()

    df = price.copy()
    # Merge load lags
    if not load.empty:
        df = df.merge(
            load.drop(columns=["Date", "HE", "CAISO Total"], errors="ignore"),
            on="timestamp",
            how="left",
            suffixes=("", "_loadlag"),
        )
    # Merge generation mix
    if not mix.empty:
        df = df.merge(mix, on="timestamp", how="left", suffixes=("", "_mix"))

    # Basic time features
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    # Merge temperature daily on date
    temp = load_temperature_daily()
    if not temp.empty:
        # Normalize date types to datetime64[ns] (no time component) on both sides
        temp["date"] = pd.to_datetime(
            temp["date"], errors="coerce").dt.normalize()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
        temp_cols = [c for c in temp.columns if c != "date"]
        df = df.merge(
            temp.rename(columns={"date": "Date"}),
            on="Date",
            how="left",
            suffixes=("", "_temp"),
        )

    # Clean, deduplicate and sort
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Save for debugging
    debug_path = ROOT / "model" / "debug_hourly_dataset.csv"
    df.to_csv(debug_path, index=False)
    print(f"Saved hourly dataset for debugging to: {debug_path}")
    print(f"Dataset shape: {df.shape}, columns: {list(df.columns)}")

    return df


def _train_and_eval(
    data: pd.DataFrame,
    target_col: str,
    model_path: Path,
    feature_exclude: Iterable[str],
    model_type="hgb"  # "hgb" or "rf"
) -> tuple[Any, dict]:
    data = data.copy()
    y = data[target_col]

    exclude = set(feature_exclude) | {target_col}
    feature_cols = [
        c for c in data.columns if c not in exclude and pd.api.types.is_numeric_dtype(data[c])]
    X = data[feature_cols].ffill().bfill()

    # Time-based split: last 20% for test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if model_type == "rf":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = HistGradientBoostingRegressor(max_depth=8, max_iter=300, learning_rate=0.05)
    
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "mape": float(mean_absolute_percentage_error(y_test, preds)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": feature_cols,
                "metrics": metrics}, model_path)
    return model, metrics


def train_daily_and_monthly():
    hourly = build_hourly_dataset()
    target = "Estimated_Hourly_Cost_USD"
    if target not in hourly.columns:
        raise ValueError(
            f"Target column '{target}' not found in hourly dataset.")

    # Daily aggregation
    hourly["date"] = pd.to_datetime(hourly["Date"])
    daily = (
        hourly.groupby("date")
        .agg(
            {
                target: "sum",
                "CAISO Total": "mean",
                "Monthly_Price_Cents_per_kWh": "mean",
                "hour": "mean",
                "dayofweek": "mean",
                "month": "mean",
                **{col: "mean" for col in hourly.columns if col.endswith(("_lag_1", "_lag_7", "_lag_15", "_lag_30"))},
                **{col: "mean" for col in hourly.columns if col not in ["timestamp", "Date", "HE", target] and pd.api.types.is_numeric_dtype(hourly[col])},
            }
        )
        .reset_index()
    )
    daily = daily.loc[:, ~daily.columns.duplicated()]
    outliers = daily[daily[target] > 20]
    if not outliers.empty:
        print(f"WARNING: Found {len(outliers)} daily target outliers > 20. Max: {outliers[target].max():.2f}")
        print(outliers[['date', target]].head())
    # Monthly aggregation
    daily["year_month"] = daily["date"].dt.to_period("M")
    monthly = (
        daily.groupby("year_month")
        .agg(
            {
                target: "sum",
                "CAISO Total": "mean",
                "Monthly_Price_Cents_per_kWh": "mean",
                **{col: "mean" for col in daily.columns if col not in ["date", target, "year_month"] and pd.api.types.is_numeric_dtype(daily[col])},
            }
        )
        .reset_index()
    )
    monthly["year_month_start"] = monthly["year_month"].dt.to_timestamp()
    
    # Weekly aggregation
    daily["week_start"] = daily["date"] - pd.to_timedelta(daily["date"].dt.dayofweek, unit="d")
    weekly = (
        daily.groupby("week_start")
        .agg(
            {
                target: "sum",
                "CAISO Total": "mean",
                "Monthly_Price_Cents_per_kWh": "mean",
                **{col: "mean" for col in daily.columns if col not in ["date", target, "week_start", "year_month", "day_name", "avg_hourly_cost"] and pd.api.types.is_numeric_dtype(daily[col])},
            }
        )
        .reset_index()
    )

    # Train models (hourly, daily, monthly)
    _, hourly_metrics = _train_and_eval(
        hourly,
        target_col=target,
        model_path=ROOT / "model" / "hourly_spend_model.pkl",
        feature_exclude=["timestamp", "Date", "HE", "date", "year_month"],
        model_type="hgb"
    )
    _, daily_metrics = _train_and_eval(
        daily,
        target_col=target,
        model_path=ROOT / "model" / "daily_spend_model.pkl",
        feature_exclude=["date", "year_month"],
        model_type="rf"
    )
    _, monthly_metrics = _train_and_eval(
        monthly.rename(columns={"year_month_start": "date"}),
        target_col=target,
        model_path=ROOT / "model" / "monthly_spend_model.pkl",
        feature_exclude=["year_month"],
        model_type="rf"
    )
    _, weekly_metrics = _train_and_eval(
        weekly.rename(columns={"week_start": "date"}),
        target_col=target,
        model_path=ROOT / "model" / "weekly_spend_model.pkl",
        feature_exclude=[],
        model_type="rf"
    )

    return {"hourly": hourly_metrics, "daily": daily_metrics, "monthly": monthly_metrics, "weekly": weekly_metrics}


def main():
    parser = argparse.ArgumentParser(
        description="Train hourly, daily and monthly residential spending models.")
    parser.add_argument("--print-metrics", action="store_true",
                        help="Print evaluation metrics after training.")
    args = parser.parse_args()

    metrics = train_daily_and_monthly()
    if args.print_metrics:
        print("Hourly metrics:", metrics["hourly"])
        print("Daily metrics:", metrics["daily"])
        print("Weekly metrics:", metrics["weekly"])
        print("Monthly metrics:", metrics["monthly"])


if __name__ == "__main__":
    main()
