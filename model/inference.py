"""Inference script using trained daily/monthly spend models.

Loads the trained models from `model/train.py`, rebuilds the feature set from
the latest data under `features/`, and produces:
 - Next-day spend prediction (persistence on most recent features)
 - Next-month spend prediction (persistence on most recent monthly features)

Results are written to `results/predictions.csv`.
"""

from __future__ import annotations

from datetime import timedelta
import pandas as pd
from pathlib import Path

import joblib
from pandas.tseries.offsets import DateOffset

from model.train import build_hourly_dataset

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
DAILY_MODEL_PATH = ROOT / "model" / "daily_spend_model.pkl"
MONTHLY_MODEL_PATH = ROOT / "model" / "monthly_spend_model.pkl"
TARGET_COL = "Estimated_Hourly_Cost_USD"


def _load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    bundle = joblib.load(path)
    model = bundle["model"]
    features = bundle["features"]
    return model, features


def _build_daily_and_monthly(hourly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hourly = hourly.copy()
    hourly["date"] = pd.to_datetime(hourly["Date"])

    daily = (
        hourly.groupby("date")
        .agg(
            {
                TARGET_COL: "sum",
                "CAISO Total": "mean",
                "Monthly_Price_Cents_per_kWh": "mean",
                "hour": "mean",
                "dayofweek": "mean",
                "month": "mean",
                **{
                    col: "mean"
                    for col in hourly.columns
                    if col.endswith(("_lag_1", "_lag_7", "_lag_15", "_lag_30"))
                },
                **{
                    col: "mean"
                    for col in hourly.columns
                    if col
                    not in ["timestamp", "Date", "HE", TARGET_COL]
                    and pd.api.types.is_numeric_dtype(hourly[col])
                },
            }
        )
        .reset_index()
    )
    daily = daily.loc[:, ~daily.columns.duplicated()]

    daily["year_month"] = daily["date"].dt.to_period("M")
    monthly = (
        daily.groupby("year_month")
        .agg(
            {
                TARGET_COL: "sum",
                "CAISO Total": "mean",
                "Monthly_Price_Cents_per_kWh": "mean",
                **{
                    col: "mean"
                    for col in daily.columns
                    if col not in ["date", TARGET_COL, "year_month"]
                    and pd.api.types.is_numeric_dtype(daily[col])
                },
            }
        )
        .reset_index()
    )
    monthly["year_month_start"] = monthly["year_month"].dt.to_timestamp()
    return daily, monthly


def _predict_next(
    model,
    features,
    df: pd.DataFrame,
    date_col: str,
    next_label: str,
    freq: str = "D",
) -> pd.DataFrame:
    df = df.sort_values(date_col)
    if df.empty:
        raise ValueError("No data available to perform inference.")

    last_row = df.iloc[[-1]].copy()
    last_date = last_row[date_col].iloc[0]

    # For next period, reuse most recent features (persistence).
    next_row = last_row.copy()
    if freq.upper().startswith("M"):
        next_row[date_col] = pd.to_datetime(last_date) + DateOffset(months=1)
    elif isinstance(last_date, pd.Timestamp):
        next_row[date_col] = last_date + timedelta(days=1)
    else:
        # Period -> timestamp conversion handled upstream; here treat as pandas Period
        next_row[date_col] = (last_date + 1).to_timestamp()

    X = next_row[features]
    pred = model.predict(X)[0]

    return pd.DataFrame(
        {
            "target": [TARGET_COL],
            "prediction": [pred],
            "for": [next_label],
            "feature_date": [next_row[date_col].iloc[0]],
        }
    )


def run_inference():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    hourly = build_hourly_dataset()

    daily, monthly = _build_daily_and_monthly(hourly)
    # Persist history for dashboard use
    daily.to_csv(RESULTS_DIR / "daily_history.csv", index=False)
    monthly.to_csv(RESULTS_DIR / "monthly_history.csv", index=False)

    daily_model, daily_features = _load_model(DAILY_MODEL_PATH)
    monthly_model, monthly_features = _load_model(MONTHLY_MODEL_PATH)

    next_day_pred = _predict_next(
        daily_model,
        daily_features,
        daily.rename(columns={"date": "date"}),
        "date",
        "next_day",
        freq="D",
    )
    next_month_pred = _predict_next(
        monthly_model,
        monthly_features,
        monthly.rename(columns={"year_month_start": "date"}),
        "date",
        "next_month",
        freq="M",
    )

    results = pd.concat([next_day_pred, next_month_pred], ignore_index=True)
    out_path = RESULTS_DIR / "predictions.csv"
    results.to_csv(out_path, index=False)
    return out_path, results


if __name__ == "__main__":
    path, df = run_inference()
    print(f"Saved predictions to {path}")
    print(df)
