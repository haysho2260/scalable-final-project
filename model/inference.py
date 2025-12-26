"""Inference script using trained daily/monthly spend models.

Loads the trained models from `model/train.py`, rebuilds the feature set from
the latest data under `features/`, and produces:
 - Next-day spend prediction (persistence on most recent features)
 - Next-month spend prediction (persistence on most recent monthly features)

Results are written to `results/predictions.csv`.
"""

from __future__ import annotations


from datetime import timedelta
from pathlib import Path
import sys

import pandas as pd
import joblib
from pandas.tseries.offsets import DateOffset

ROOT = Path(__file__).resolve().parents[1]

# Ensure project root is on sys.path so `model` package can be imported when
# running this file as `python model/inference.py`.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.train import build_hourly_dataset

RESULTS_DIR = ROOT / "results"
HOURLY_MODEL_PATH = ROOT / "model" / "hourly_spend_model.pkl"
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

    # Calculate next date
    if freq.upper().startswith("M"):
        next_date = pd.to_datetime(last_date) + DateOffset(months=1)
    elif freq.upper().startswith("H"):
        # Hourly: add 1 hour
        if isinstance(last_date, pd.Timestamp):
            next_date = last_date + timedelta(hours=1)
        else:
            next_date = pd.to_datetime(last_date) + timedelta(hours=1)
    elif isinstance(last_date, pd.Timestamp):
        next_date = last_date + timedelta(days=1)
    else:
        # Period -> timestamp conversion handled upstream; here treat as pandas Period
        next_date = (last_date + 1).to_timestamp()

    # Convert date_col to datetime for easier comparison
    df_dates = pd.to_datetime(df[date_col], errors="coerce")

    # Find similar historical periods to use as a base
    # For daily: same day of week, same month (from previous years)
    # For monthly: same month from previous years
    similar_rows = pd.DataFrame()

    if freq.upper().startswith("M"):
        # Monthly: find same month from previous years
        similar_mask = df_dates.dt.month == next_date.month
        similar_rows = df[similar_mask]
    else:
        # Daily: find same day of week and same month from history
        similar_mask = (
            (df_dates.dt.month == next_date.month) &
            (df_dates.dt.dayofweek == next_date.dayofweek)
        )
        similar_rows = df[similar_mask]

    # Use the most recent similar row, or fall back to last row
    if not similar_rows.empty:
        # Use the most recent similar period (but not the exact same date)
        # Use iloc[[-1]] to ensure we get a DataFrame, not a Series
        base_row = similar_rows.iloc[[-1]].copy()
    else:
        # Fallback: use last row but we'll update features
        base_row = last_row.copy()

    # Ensure base_row is a DataFrame (not a Series)
    if isinstance(base_row, pd.Series):
        base_row = base_row.to_frame().T

    # Create next_row starting from base_row
    next_row = base_row.copy()

    # Ensure next_row is a DataFrame
    if isinstance(next_row, pd.Series):
        next_row = next_row.to_frame().T

    next_row[date_col] = next_date

    # Final safety check: ensure next_row is a DataFrame
    if not isinstance(next_row, pd.DataFrame):
        next_row = next_row.to_frame().T if isinstance(
            next_row, pd.Series) else pd.DataFrame([next_row])

    # Update temporal features based on next date
    if "hour" in next_row.columns:
        if freq.upper().startswith("M"):
            # For monthly, use average hour from similar months
            if not similar_rows.empty and "hour" in similar_rows.columns:
                next_row["hour"] = similar_rows["hour"].mean()
            else:
                next_row["hour"] = df["hour"].mean(
                ) if "hour" in df.columns else 12
        else:
            # For daily, use average hour from similar days
            if not similar_rows.empty and "hour" in similar_rows.columns:
                next_row["hour"] = similar_rows["hour"].mean()
            else:
                next_row["hour"] = df["hour"].mean(
                ) if "hour" in df.columns else 12

    if "dayofweek" in next_row.columns:
        next_row["dayofweek"] = next_date.dayofweek

    if "month" in next_row.columns:
        next_row["month"] = next_date.month

    # Update lag features using recent history
    # Lag features are like "daily_mean_cost_lag_1", "daily_std_cost_lag_1", etc.
    for lag_col in [col for col in features if "_lag_" in col]:
        lag_value = None
        try:
            # Extract lag number (e.g., "daily_mean_cost_lag_7" -> 7)
            lag_num = int(lag_col.split("_lag_")[1])

            if freq.upper().startswith("M"):
                # Monthly: look back by months (only use lag_1 for monthly)
                if lag_num == 1:
                    lag_date = next_date - DateOffset(months=1)
                    # Convert date_col to datetime for comparison
                    df_dates = pd.to_datetime(df[date_col], errors="coerce")
                    lag_rows = df[df_dates <= lag_date]
                    if not lag_rows.empty:
                        # Get the base feature (e.g., "daily_mean_cost" from "daily_mean_cost_lag_1")
                        base_col = lag_col.replace("_lag_" + str(lag_num), "")
                        if base_col in lag_rows.columns:
                            lag_value = lag_rows.iloc[-1][base_col]
                        elif lag_col in lag_rows.columns:
                            lag_value = lag_rows.iloc[-1][lag_col]
                else:
                    # For longer lags in monthly, use lag_1 value
                    base_col = lag_col.replace("_lag_" + str(lag_num), "")
                    lag_1_col = base_col + "_lag_1"
                    if lag_1_col in df.columns:
                        lag_date = next_date - DateOffset(months=1)
                        df_dates = pd.to_datetime(
                            df[date_col], errors="coerce")
                        lag_rows = df[df_dates <= lag_date]
                        if not lag_rows.empty:
                            base_col_clean = lag_1_col.replace("_lag_1", "")
                            if base_col_clean in lag_rows.columns:
                                lag_value = lag_rows.iloc[-1][base_col_clean]
            else:
                # Daily: look back by days
                lag_date = next_date - timedelta(days=lag_num)
                df_dates = pd.to_datetime(df[date_col], errors="coerce")
                lag_rows = df[df_dates <= lag_date]
                if not lag_rows.empty:
                    # Get the base feature value from that date
                    base_col = lag_col.replace("_lag_" + str(lag_num), "")
                    if base_col in lag_rows.columns:
                        lag_value = lag_rows.iloc[-1][base_col]
                    elif lag_col in lag_rows.columns:
                        lag_value = lag_rows.iloc[-1][lag_col]
        except (ValueError, KeyError, IndexError):
            pass

        if lag_value is not None and not pd.isna(lag_value):
            next_row[lag_col] = lag_value
        else:
            # Fallback: use recent average of the lag column itself
            if lag_col in df.columns:
                next_row[lag_col] = df[lag_col].tail(
                    30).mean() if len(df) >= 30 else df[lag_col].mean()
            else:
                # If lag column doesn't exist, try to compute from base column
                try:
                    base_col = lag_col.split("_lag_")[0]
                    if base_col in df.columns:
                        next_row[lag_col] = df[base_col].tail(
                            30).mean() if len(df) >= 30 else df[base_col].mean()
                except:
                    pass

    # Use historical values from similar periods (not just averages - use actual values for variation)
    if not similar_rows.empty:
        # Use the most recent similar period's actual values (gives natural variation)
        # Use iloc[[-1]] to ensure we get a DataFrame
        similar_row_actual = similar_rows.iloc[[-1]].copy()

        # Update all features from similar historical periods (except lag features which are handled separately)
        for col in features:
            if "_lag_" not in col and col in similar_row_actual.columns:
                # Use actual value from similar historical period (not average - this gives variation)
                next_row[col] = similar_row_actual[col].iloc[0]

    # For features not yet set or missing, use recent averages
    for col in features:
        if col not in next_row.columns:
            continue
        col_val = next_row[col].iloc[0] if hasattr(
            next_row[col], 'iloc') else next_row[col]
        if pd.isna(col_val):
            if col in df.columns:
                if freq.upper().startswith("M"):
                    # Monthly: use average from last 12 months
                    next_row[col] = df[col].tail(12).mean() if len(
                        df) >= 12 else df[col].mean()
                else:
                    # Daily: use average from last 30 days
                    next_row[col] = df[col].tail(30).mean() if len(
                        df) >= 30 else df[col].mean()

    # Fill any remaining NaN features with recent averages
    for col in features:
        if col in next_row.columns and pd.isna(next_row[col].iloc[0]):
            if col in df.columns:
                next_row[col] = df[col].tail(30).mean() if len(
                    df) >= 30 else df[col].mean()

    X = next_row[features].fillna(method="ffill").fillna(method="bfill")
    pred = model.predict(X)[0]

    return pd.DataFrame(
        {
            "target": [TARGET_COL],
            "prediction": [pred],
            "for": [next_label],
            "feature_date": [next_date],
        }
    )


def run_inference():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    hourly = build_hourly_dataset()

    daily, monthly = _build_daily_and_monthly(hourly)
    # Persist history for dashboard use
    hourly_history = hourly[["timestamp", "Date", "HE", TARGET_COL]].copy()
    hourly_history.to_csv(RESULTS_DIR / "hourly_history.csv", index=False)
    daily.to_csv(RESULTS_DIR / "daily_history.csv", index=False)
    monthly.to_csv(RESULTS_DIR / "monthly_history.csv", index=False)

    hourly_model, hourly_features = _load_model(HOURLY_MODEL_PATH)
    daily_model, daily_features = _load_model(DAILY_MODEL_PATH)
    monthly_model, monthly_features = _load_model(MONTHLY_MODEL_PATH)

    # Predict next hour
    hourly_sorted = hourly.sort_values("timestamp")
    next_hour_pred = _predict_next(
        hourly_model,
        hourly_features,
        hourly_sorted.rename(columns={"timestamp": "date"}),
        "date",
        "next_hour",
        freq="H",
    )

    # Predict next day
    next_day_pred = _predict_next(
        daily_model,
        daily_features,
        daily.rename(columns={"date": "date"}),
        "date",
        "next_day",
        freq="D",
    )

    # Predict next month
    next_month_pred = _predict_next(
        monthly_model,
        monthly_features,
        monthly.rename(columns={"year_month_start": "date"}),
        "date",
        "next_month",
        freq="M",
    )

    results = pd.concat([next_hour_pred, next_day_pred,
                        next_month_pred], ignore_index=True)
    out_path = RESULTS_DIR / "predictions.csv"
    results.to_csv(out_path, index=False)
    return out_path, results


if __name__ == "__main__":
    path, df = run_inference()
    print(f"Saved predictions to {path}")
    print(df)
