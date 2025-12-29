"""Inference script using trained daily/monthly spend models.

Loads the trained models from `model/train.py`, rebuilds the feature set from
the latest data under `features/`, and produces:
 - Next-day spend prediction (persistence on most recent features)
 - Next-month spend prediction (persistence on most recent monthly features)

Results are written to `results/predictions.csv`.
"""

from __future__ import annotations



from datetime import timedelta, datetime
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
WEEKLY_MODEL_PATH = ROOT / "model" / "weekly_spend_model.pkl"
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

    # Weekly aggregation
    daily["week_start"] = daily["date"] - \
        pd.to_timedelta(daily["date"].dt.dayofweek, unit="d")
    weekly = (
        daily.groupby("week_start")
        .agg(
            {
                TARGET_COL: "sum",
                "CAISO Total": "mean",
                "Monthly_Price_Cents_per_kWh": "mean",
                **{
                    col: "mean"
                    for col in daily.columns
                    if col not in ["date", TARGET_COL, "week_start", "year_month"]
                    and pd.api.types.is_numeric_dtype(daily[col])
                },
            }
        )
        .reset_index()
    )
    return daily, weekly, monthly


def _predict_next(
    model,
    features,
    df: pd.DataFrame,
    date_col: str,
    next_label: str,
    freq: str = "D",
    target_date: pd.Timestamp | None = None,
    max_hist_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    df = df.sort_values(date_col)
    if df.empty:
        raise ValueError("No data available to perform inference.")

    # OPTIMIZATION: Limit historical data to recent period for faster filtering
    # Use only last 2 years of data for feature calculation (faster than full dataset)
    if len(df) > 0:
        if freq.upper().startswith("H"):
            # For hourly: use last 730 days (2 years)
            cutoff_date = pd.to_datetime(
                df[date_col].iloc[-1]) - timedelta(days=730)
            df = df[pd.to_datetime(
                df[date_col], errors="coerce") >= cutoff_date].copy()
        elif freq.upper().startswith("D"):
            # For daily: use last 730 days (2 years)
            cutoff_date = pd.to_datetime(
                df[date_col].iloc[-1]) - timedelta(days=730)
            df = df[pd.to_datetime(
                df[date_col], errors="coerce") >= cutoff_date].copy()
        # For monthly, keep all data (smaller dataset)

    last_row = df.iloc[[-1]].copy()
    last_date = last_row[date_col].iloc[0]

    # Calculate next date (use target_date if provided, otherwise calculate from last_date)
    if target_date is not None:
        next_date = pd.to_datetime(target_date)
    elif freq.upper().startswith("M"):
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

    # OPTIMIZATION: Cache datetime conversion and date components
    df_dates = pd.to_datetime(df[date_col], errors="coerce")
    next_month = next_date.month
    next_dayofweek = next_date.dayofweek
    next_hour = next_date.hour if freq.upper().startswith("H") else None

    # OPTIMIZATION: Pre-compute date components for faster filtering
    # Cache date components to avoid repeated dt access
    # Isolate TRUE history from recursively generated predictions for feature anchoring
    if max_hist_date is not None:
        history_df = df[df_dates <= pd.to_datetime(max_hist_date)].copy()
    else:
        history_df = df.copy()

    # Pre-compute date components on history for filtering
    hist_dates = pd.to_datetime(history_df[date_col], errors="coerce")
    history_df['_cached_month'] = hist_dates.dt.month
    history_df['_cached_dayofweek'] = hist_dates.dt.dayofweek
    if freq.upper().startswith("H"):
        history_df['_cached_hour'] = hist_dates.dt.hour

    # Find similar historical periods to use as a base
    # For daily: same day of week, same month (from previous years)
    # For monthly: same month from previous years
    similar_rows = pd.DataFrame()

    if freq.upper().startswith("M"):
        # Monthly: find same month from previous years
        similar_mask = history_df['_cached_month'] == next_month
        similar_rows = history_df[similar_mask]
    elif freq.upper().startswith("H"):
        # Hourly: find same hour of day, same day of week, same month from history
        similar_mask = (
            (history_df['_cached_month'] == next_month) &
            (history_df['_cached_dayofweek'] == next_dayofweek) &
            (history_df['_cached_hour'] == next_hour)
        )
        similar_rows = history_df[similar_mask]
    else:
        # Daily: find same day of week and same month from history
        similar_mask = (
            (history_df['_cached_month'] == next_month) &
            (history_df['_cached_dayofweek'] == next_dayofweek)
        )
        similar_rows = history_df[similar_mask]

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
        if freq.upper().startswith("H"):
            # For hourly, use the actual next hour
            next_row["hour"] = next_date.hour
        elif freq.upper().startswith("M"):
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
            elif freq.upper().startswith("H"):
                # Hourly: look back by hours
                lag_date = next_date - timedelta(hours=lag_num)
                # OPTIMIZATION: Use cached df_dates instead of recalculating
                lag_rows = df[df_dates <= lag_date]
                if not lag_rows.empty:
                    # Get the base feature value from that hour
                    base_col = lag_col.replace("_lag_" + str(lag_num), "")
                    if base_col in lag_rows.columns:
                        lag_value = lag_rows.iloc[-1][base_col]
                    elif lag_col in lag_rows.columns:
                        lag_value = lag_rows.iloc[-1][lag_col]
            else:
                # Daily: look back by days
                lag_date = next_date - timedelta(days=lag_num)
                # OPTIMIZATION: Use cached df_dates instead of recalculating
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
            # OPTIMIZATION: Use iloc for faster tail access
            if lag_col in df.columns:
                tail_size = min(30, len(df))
                next_row[lag_col] = df[lag_col].iloc[-tail_size:
                                                     ].mean() if tail_size > 0 else df[lag_col].mean()
            else:
                # If lag column doesn't exist, try to compute from base column
                try:
                    base_col = lag_col.split("_lag_")[0]
                    if base_col in df.columns:
                        tail_size = min(30, len(df))
                        next_row[lag_col] = df[base_col].iloc[-tail_size:].mean(
                        ) if tail_size > 0 else df[base_col].mean()
                except:
                    pass

    # Use historical values from similar periods (not just averages - use actual values for variation)
    # Priority: ensure critical features (CAISO Total, price) are set from similar periods
    critical_features = ["CAISO Total", "Monthly_Price_Cents_per_kWh"]

    if not similar_rows.empty:
        # Use the most recent similar period's actual values (gives natural variation)
        # Use iloc[[-1]] to ensure we get a DataFrame
        similar_row_actual = similar_rows.iloc[[-1]].copy()

        # First, set critical features from similar periods (these are essential for accurate predictions)
        for col in critical_features:
            if col in features and col in similar_row_actual.columns:
                val = similar_row_actual[col].iloc[0]
                if not pd.isna(val) and val != 0:
                    next_row[col] = val

        # Then update all other features from similar historical periods (except lag features which are handled separately)
        for col in features:
            if "_lag_" not in col and col not in critical_features and col in similar_row_actual.columns:
                # Use actual value from similar historical period (not average - this gives variation)
                val = similar_row_actual[col].iloc[0]
                if not pd.isna(val):
                    next_row[col] = val

    # For features not yet set or missing, use recent averages
    for col in features:
        if col not in next_row.columns:
            continue
        col_val = next_row[col].iloc[0] if hasattr(
            next_row[col], 'iloc') else next_row[col]
        if pd.isna(col_val):
            if col in df.columns:
                # OPTIMIZATION: Use iloc for faster tail access
                if freq.upper().startswith("M"):
                    # Monthly: use average from last 12 months
                    tail_size = min(12, len(df))
                    next_row[col] = df[col].iloc[-tail_size:
                                                 ].mean() if tail_size > 0 else df[col].mean()
                elif freq.upper().startswith("H"):
                    # Hourly: use average from last 24 hours (same time period)
                    tail_size = min(24, len(df))
                    next_row[col] = df[col].iloc[-tail_size:
                                                 ].mean() if tail_size > 0 else df[col].mean()
                else:
                    # Daily: use average from last 30 days
                    tail_size = min(30, len(df))
                    next_row[col] = df[col].iloc[-tail_size:
                                                 ].mean() if tail_size > 0 else df[col].mean()

    # Ensure critical features are set even if similar_rows was empty
    for col in critical_features:
        if col in features:
            if col not in next_row.columns or pd.isna(next_row[col].iloc[0] if hasattr(next_row[col], 'iloc') else next_row[col]) or (next_row[col].iloc[0] if hasattr(next_row[col], 'iloc') else next_row[col]) == 0:
                if col in df.columns:
                    # For hourly, try to get from same hour of day from recent days
                    if freq.upper().startswith("H") and "hour" in next_row.columns:
                        hour = int(next_row["hour"].iloc[0] if hasattr(
                            next_row["hour"], 'iloc') else next_row["hour"])
                        # OPTIMIZATION: Use cached hour column if available
                        hour_col = '_cached_hour' if '_cached_hour' in df.columns else 'hour'
                        same_hour_data = df[df[hour_col] ==
                                            hour] if hour_col in df.columns else df
                        if not same_hour_data.empty:
                            tail_size = min(7, len(same_hour_data))
                            next_row[col] = same_hour_data[col].iloc[-tail_size:].mean(
                            ) if tail_size > 0 else same_hour_data[col].mean()
                        else:
                            tail_size = min(24, len(df))
                            next_row[col] = df[col].iloc[-tail_size:].mean(
                            ) if tail_size > 0 else df[col].mean()
                    else:
                        # For daily/monthly, use recent average
                        if freq.upper().startswith("M"):
                            tail_size = min(12, len(df))
                            next_row[col] = df[col].iloc[-tail_size:].mean(
                            ) if tail_size > 0 else df[col].mean()
                        else:
                            tail_size = min(30, len(df))
                            next_row[col] = df[col].iloc[-tail_size:].mean(
                            ) if tail_size > 0 else df[col].mean()

    # Fill any remaining NaN features with recent averages
    for col in features:
        if col in next_row.columns:
            col_val = next_row[col].iloc[0] if hasattr(
                next_row[col], 'iloc') else next_row[col]
            if pd.isna(col_val):
                if col in df.columns:
                    # OPTIMIZATION: Use iloc for faster tail access
                    if freq.upper().startswith("H"):
                        # Hourly: use last 24 hours
                        tail_size = min(24, len(df))
                        next_row[col] = df[col].iloc[-tail_size:
                                                     ].mean() if tail_size > 0 else df[col].mean()
                    elif freq.upper().startswith("M"):
                        # Monthly: use last 12 months
                        tail_size = min(12, len(df))
                        next_row[col] = df[col].iloc[-tail_size:
                                                     ].mean() if tail_size > 0 else df[col].mean()
                    else:
                        # Daily: use last 30 days
                        tail_size = min(30, len(df))
                        next_row[col] = df[col].iloc[-tail_size:
                                                     ].mean() if tail_size > 0 else df[col].mean()

    # Ensure critical features are not zero or NaN before prediction
    for col in critical_features:
        if col in next_row.columns:
            val = next_row[col].iloc[0] if hasattr(
                next_row[col], 'iloc') else next_row[col]
            if pd.isna(val) or val == 0:
                # Try to get from recent data - use more data for better estimate
                if col in df.columns:
                    # OPTIMIZATION: Use iloc for faster tail access
                    # Use last 168 hours (1 week) for hourly, last 30 days for daily
                    if freq.upper().startswith("H"):
                        tail_size = min(168, len(df))  # 1 week of hourly data
                    else:
                        tail_size = min(30, len(df))  # 30 days for daily

                    # For critical features, prefer non-zero values
                    if col in critical_features:
                        non_zero_data = df[df[col] > 0][col]
                        if len(non_zero_data) > 0:
                            recent_val = non_zero_data.iloc[-min(
                                tail_size, len(non_zero_data)):].mean()
                        else:
                            recent_val = df[col].iloc[-tail_size:].mean(
                            ) if tail_size > 0 else df[col].mean()
                    else:
                        recent_val = df[col].iloc[-tail_size:].mean(
                        ) if tail_size > 0 else df[col].mean()

                    if not pd.isna(recent_val) and recent_val != 0:
                        next_row[col] = recent_val
                    else:
                        # Last resort: use overall mean (prefer non-zero)
                        if col in critical_features:
                            non_zero_overall = df[df[col] > 0][col]
                            overall_mean = non_zero_overall.mean() if len(
                                non_zero_overall) > 0 else df[col].mean()
                        else:
                            overall_mean = df[col].mean()
                        if not pd.isna(overall_mean) and overall_mean != 0:
                            next_row[col] = overall_mean

    X = next_row[features].ffill().bfill()

    # Debug: Check if critical features are set
    if "CAISO Total" in features and "CAISO Total" in X.columns:
        caiso_val = X["CAISO Total"].iloc[0] if len(X) > 0 else None
        if caiso_val is None or pd.isna(caiso_val) or caiso_val == 0:
            # If CAISO Total is missing/zero, use recent NON-ZERO average from df
            if "CAISO Total" in df.columns:
                # Filter out zeros and get recent non-zero values
                non_zero_caiso = df[df["CAISO Total"] > 0]["CAISO Total"]
                if len(non_zero_caiso) > 0:
                    # Use last 168 hours of non-zero data, or all if less available
                    recent_caiso = non_zero_caiso.iloc[-min(
                        168, len(non_zero_caiso)):].mean()
                    X["CAISO Total"] = recent_caiso
                else:
                    # Last resort: use overall mean (even if some zeros)
                    recent_caiso = df["CAISO Total"].mean()
                    if recent_caiso > 0:
                        X["CAISO Total"] = recent_caiso

    # Debug logging for first few predictions to diagnose issues
    if not hasattr(_predict_next, '_debug_count'):
        _predict_next._debug_count = 0
    _predict_next._debug_count += 1
    if _predict_next._debug_count <= 3:
        caiso_val = X["CAISO Total"].iloc[0] if "CAISO Total" in X.columns else None
        price_val = X["Monthly_Price_Cents_per_kWh"].iloc[0] if "Monthly_Price_Cents_per_kWh" in X.columns else None
        print(
            f"  DEBUG Prediction #{_predict_next._debug_count}: CAISO Total={caiso_val:.2f}, Price={price_val:.2f}, Date={next_date}")

    pred = model.predict(X)[0]

    # Ensure prediction is not negative (costs can't be negative)
    if pred < 0:
        pred = 0.0

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

    daily, weekly, monthly = _build_daily_and_monthly(hourly)
    # Persist history for dashboard use
    hourly_history = hourly[["timestamp", "Date", "HE", TARGET_COL]].copy()
    hourly_history.to_csv(RESULTS_DIR / "hourly_history.csv", index=False)
    daily.to_csv(RESULTS_DIR / "daily_history.csv", index=False)
    weekly.to_csv(RESULTS_DIR / "weekly_history.csv", index=False)
    monthly.to_csv(RESULTS_DIR / "monthly_history.csv", index=False)

    hourly_model, hourly_features = _load_model(HOURLY_MODEL_PATH)
    daily_model, daily_features = _load_model(DAILY_MODEL_PATH)
    weekly_model, weekly_features = _load_model(WEEKLY_MODEL_PATH)
    monthly_model, monthly_features = _load_model(MONTHLY_MODEL_PATH)

    # Get last data point and current time
    hourly_sorted = hourly.sort_values("timestamp")
    last_hourly_timestamp = pd.to_datetime(hourly_sorted["timestamp"].iloc[-1])
    last_daily_date = pd.to_datetime(daily["date"].iloc[-1])
    last_monthly_date = pd.to_datetime(monthly["year_month_start"].iloc[-1])

    # Current time (round down to hour for hourly, day for daily, month for monthly)
    now = datetime.now()
    current_hour = now.replace(minute=0, second=0, microsecond=0)
    current_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    current_month = datetime(now.year, now.month, 1)

    # Limit predictions to reach current day + 1 month
    # Started from Sept 30, we need ~120 days to reach late Jan 2026
    target_future = now + timedelta(days=31)
    target_future_hour = target_future.replace(
        minute=0, second=0, microsecond=0)
    target_future_day = target_future.replace(
        hour=0, minute=0, second=0, microsecond=0)
    target_future_month = datetime(target_future.year, target_future.month, 1)

    max_pred_hour = last_hourly_timestamp + timedelta(days=150)
    max_pred_day = last_daily_date + timedelta(days=180)
    max_pred_week = pd.to_datetime(
        weekly["week_start"].iloc[-1]) + timedelta(weeks=32)
    max_pred_month = last_monthly_date + DateOffset(months=18)

    # Use the earlier of: current time or max prediction window
    pred_hour_limit = min(pd.to_datetime(target_future_hour),
                          pd.to_datetime(max_pred_hour))
    pred_day_limit = min(pd.to_datetime(target_future_day),
                         pd.to_datetime(max_pred_day))
    pred_week_limit = min(pd.to_datetime(target_future_day),
                          pd.to_datetime(max_pred_week))
    pred_month_limit = min(pd.to_datetime(target_future_month),
                           pd.to_datetime(max_pred_month))

    # Generate predictions for all periods from last data to current time (or limit)
    all_predictions = []

    # Hourly predictions: from last hour + 1 to limit
    if last_hourly_timestamp < pred_hour_limit:
        hourly_df = hourly_sorted.rename(columns={"timestamp": "date"})
        current_pred_hour = last_hourly_timestamp + timedelta(hours=1)
        max_hours = 3600  # Max 150 days of hourly predictions (24 * 150)
        hour_count = 0
        hourly_new_rows_buffer = []  # Use separate list instead of DataFrame attribute

        print(
            f"Generating hourly predictions from {current_pred_hour} to {pred_hour_limit} (max {max_hours} hours)...")
        while current_pred_hour <= pred_hour_limit and hour_count < max_hours:
            if hour_count % 24 == 0:  # Print progress every 24 hours
                print(f"  Progress: {hour_count}/{max_hours} hours")
            pred = _predict_next(
                hourly_model,
                hourly_features,
                hourly_df,
                "date",
                f"hour_{current_pred_hour.strftime('%Y-%m-%d %H:00')}",
                freq="H",
                target_date=current_pred_hour,
                max_hist_date=last_hourly_timestamp,
            )
            all_predictions.append(pred)

            # Update hourly_df with the prediction for next iteration
            # Add a dummy row with the predicted timestamp for feature calculation
            # Collect new rows in a list and concat in batches for better performance
            new_row = hourly_df.iloc[-1:].copy()
            new_row["date"] = current_pred_hour
            new_row[TARGET_COL] = pred["prediction"].iloc[0]
            hourly_new_rows_buffer.append(new_row)
            # Concat every 100 rows to balance memory and performance
            if len(hourly_new_rows_buffer) >= 100:
                hourly_df = pd.concat(
                    [hourly_df] + hourly_new_rows_buffer, ignore_index=True)
                hourly_new_rows_buffer = []

            current_pred_hour += timedelta(hours=1)
            hour_count += 1

        # Final concat of any remaining buffered rows
        if hourly_new_rows_buffer:
            hourly_df = pd.concat(
                [hourly_df] + hourly_new_rows_buffer, ignore_index=True)

    # Daily predictions: from last day + 1 to limit
    if last_daily_date < pred_day_limit:
        daily_df = daily.rename(columns={"date": "date"})
        current_pred_day = last_daily_date + timedelta(days=1)
        max_days = 180  # Max 180 days of daily predictions
        day_count = 0
        daily_new_rows_buffer = []  # Use separate list instead of DataFrame attribute

        print(
            f"Generating daily predictions from {current_pred_day} to {pred_day_limit} (max {max_days} days)...")
        while current_pred_day <= pred_day_limit and day_count < max_days:
            if day_count % 7 == 0:  # Print progress every week
                print(f"  Progress: {day_count}/{max_days} days")
            pred = _predict_next(
                daily_model,
                daily_features,
                daily_df,
                "date",
                f"day_{current_pred_day.strftime('%Y-%m-%d')}",
                freq="D",
                target_date=current_pred_day,
                max_hist_date=last_daily_date,
            )
            all_predictions.append(pred)

            # Update daily_df with the prediction for next iteration
            new_row = daily_df.iloc[-1:].copy()
            new_row["date"] = current_pred_day
            new_row[TARGET_COL] = pred["prediction"].iloc[0]
            # Buffer and concat in batches for better performance
            daily_new_rows_buffer.append(new_row)
            if len(daily_new_rows_buffer) >= 30:
                daily_df = pd.concat(
                    [daily_df] + daily_new_rows_buffer, ignore_index=True)
                daily_new_rows_buffer = []

            current_pred_day += timedelta(days=1)
            day_count += 1

        # Final concat of any remaining buffered rows
        if daily_new_rows_buffer:
            daily_df = pd.concat(
                [daily_df] + daily_new_rows_buffer, ignore_index=True)

    # Weekly predictions: from last week + 1 to limit
    last_weekly_date = pd.to_datetime(weekly["week_start"].iloc[-1])
    if last_weekly_date < pred_week_limit:
        weekly_df = weekly.rename(columns={"week_start": "date"})
        current_pred_week = last_weekly_date + timedelta(weeks=1)
        max_weeks = 32  # Max 32 weeks of weekly predictions
        week_count = 0

        print(
            f"Generating weekly predictions from {current_pred_week} to {pred_week_limit} (max {max_weeks} weeks)...")
        while current_pred_week <= pred_week_limit and week_count < max_weeks:
            print(f"  Progress: {week_count}/{max_weeks} weeks")
            pred = _predict_next(
                weekly_model,
                weekly_features,
                weekly_df,
                "date",
                f"week_{current_pred_week.strftime('%Y-%m-%d')}",
                freq="W",
                target_date=current_pred_week,
                max_hist_date=last_daily_date,  # Weekly uses daily data as base
            )
            all_predictions.append(pred)

            # Update weekly_df with the prediction for next iteration
            new_row = weekly_df.iloc[-1:].copy()
            new_row["date"] = current_pred_week
            new_row[TARGET_COL] = pred["prediction"].iloc[0]
            weekly_df = pd.concat([weekly_df, new_row], ignore_index=True)

            current_pred_week += timedelta(weeks=1)
            week_count += 1

    # Monthly predictions: from last month + 1 to limit
    # Also include the last historical month if we're currently in that month (it might be incomplete)
    if last_monthly_date < pred_month_limit:
        monthly_df = monthly.rename(columns={"year_month_start": "date"})

        # Check if we're currently in the same month as the last historical month
        # If so, that month might be incomplete and should be predicted
        current_month_start = datetime(now.year, now.month, 1)
        last_monthly_dt = pd.to_datetime(last_monthly_date)

        if last_monthly_dt == current_month_start:
            # We're in the same month as last historical, so it's incomplete - predict it
            current_pred_month = last_monthly_dt
            print(
                f"  Note: Last historical month ({last_monthly_dt.strftime('%Y-%m')}) is current month and may be incomplete, including it in predictions")
        else:
            # Start from the month after the last historical month
            current_pred_month = last_monthly_dt + DateOffset(months=1)

        max_months = 12  # Max 12 months of monthly predictions
        month_count = 0

        print(
            f"Generating monthly predictions from {current_pred_month} to {pred_month_limit} (max {max_months} months)...")
        print(
            f"  Last historical month: {last_monthly_dt.strftime('%Y-%m')}, Starting predictions from: {current_pred_month.strftime('%Y-%m')}")
        while current_pred_month <= pred_month_limit and month_count < max_months:
            print(f"  Progress: {month_count}/{max_months} months")
            pred = _predict_next(
                monthly_model,
                monthly_features,
                monthly_df,
                "date",
                f"month_{current_pred_month.strftime('%Y-%m')}",
                freq="M",
                target_date=current_pred_month,
                max_hist_date=last_monthly_date,
            )
            all_predictions.append(pred)

            # Update monthly_df with the prediction for next iteration
            new_row = monthly_df.iloc[-1:].copy()
            new_row["date"] = current_pred_month
            new_row[TARGET_COL] = pred["prediction"].iloc[0]
            # For monthly, concat immediately is fine (only 12 predictions)
            monthly_df = pd.concat([monthly_df, new_row], ignore_index=True)

            current_pred_month += DateOffset(months=1)
            month_count += 1

    if all_predictions:
        results = pd.concat(all_predictions, ignore_index=True)
        # Sort by date
        results = results.sort_values("feature_date")
    else:
        # Fallback: generate single predictions if no gap
        next_hour_pred = _predict_next(
            hourly_model,
            hourly_features,
            hourly_sorted.rename(columns={"timestamp": "date"}),
            "date",
            "next_hour",
            freq="H",
        )
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
        next_week_pred = _predict_next(
            weekly_model,
            weekly_features,
            weekly.rename(columns={"week_start": "date"}),
            "date",
            "next_week",
            freq="W",
        )
        results = pd.concat([next_hour_pred, next_day_pred, next_week_pred,
                            next_month_pred], ignore_index=True)

    out_path = RESULTS_DIR / "predictions.csv"
    results.to_csv(out_path, index=False)
    return out_path, results


if __name__ == "__main__":
    path, df = run_inference()
    print(f"Saved predictions to {path}")
    print(df)
