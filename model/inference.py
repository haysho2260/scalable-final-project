"""Inference script using trained daily/monthly spend models.

Loads the trained models from `model/train.py`, rebuilds the feature set from
the latest data under `features/`, and produces:
 - Next-day spend prediction (persistence on most recent features)
 - Next-month spend prediction (persistence on most recent monthly features)

Results are written to `results/predictions.csv`.
"""

from __future__ import annotations
from model.train import build_hourly_dataset


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
    target_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    df = df.sort_values(date_col)
    if df.empty:
        raise ValueError("No data available to perform inference.")

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
    elif freq.upper().startswith("H"):
        # Hourly: find same hour of day, same day of week, same month from history
        similar_mask = (
            (df_dates.dt.month == next_date.month) &
            (df_dates.dt.dayofweek == next_date.dayofweek) &
            (df_dates.dt.hour == next_date.hour)
        )
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
                df_dates = pd.to_datetime(df[date_col], errors="coerce")
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
                if freq.upper().startswith("M"):
                    # Monthly: use average from last 12 months
                    next_row[col] = df[col].tail(12).mean() if len(
                        df) >= 12 else df[col].mean()
                elif freq.upper().startswith("H"):
                    # Hourly: use average from last 24 hours (same time period)
                    next_row[col] = df[col].tail(24).mean() if len(
                        df) >= 24 else df[col].mean()
                else:
                    # Daily: use average from last 30 days
                    next_row[col] = df[col].tail(30).mean() if len(
                        df) >= 30 else df[col].mean()

    # Ensure critical features are set even if similar_rows was empty
    for col in critical_features:
        if col in features:
            if col not in next_row.columns or pd.isna(next_row[col].iloc[0] if hasattr(next_row[col], 'iloc') else next_row[col]) or (next_row[col].iloc[0] if hasattr(next_row[col], 'iloc') else next_row[col]) == 0:
                if col in df.columns:
                    # For hourly, try to get from same hour of day from recent days
                    if freq.upper().startswith("H") and "hour" in next_row.columns:
                        hour = int(next_row["hour"].iloc[0] if hasattr(
                            next_row["hour"], 'iloc') else next_row["hour"])
                        same_hour_data = df[df["hour"] ==
                                            hour] if "hour" in df.columns else df
                        if not same_hour_data.empty:
                            next_row[col] = same_hour_data[col].tail(7).mean() if len(
                                same_hour_data) >= 7 else same_hour_data[col].mean()
                        else:
                            next_row[col] = df[col].tail(24).mean() if len(
                                df) >= 24 else df[col].mean()
                    else:
                        # For daily/monthly, use recent average
                        if freq.upper().startswith("M"):
                            next_row[col] = df[col].tail(12).mean() if len(
                                df) >= 12 else df[col].mean()
                        else:
                            next_row[col] = df[col].tail(30).mean() if len(
                                df) >= 30 else df[col].mean()

    # Fill any remaining NaN features with recent averages
    for col in features:
        if col in next_row.columns:
            col_val = next_row[col].iloc[0] if hasattr(
                next_row[col], 'iloc') else next_row[col]
            if pd.isna(col_val):
                if col in df.columns:
                    if freq.upper().startswith("H"):
                        # Hourly: use last 24 hours
                        next_row[col] = df[col].tail(24).mean() if len(
                            df) >= 24 else df[col].mean()
                    elif freq.upper().startswith("M"):
                        # Monthly: use last 12 months
                        next_row[col] = df[col].tail(12).mean() if len(
                            df) >= 12 else df[col].mean()
                    else:
                        # Daily: use last 30 days
                        next_row[col] = df[col].tail(30).mean() if len(
                            df) >= 30 else df[col].mean()

    # Ensure critical features are not zero or NaN before prediction
    for col in critical_features:
        if col in next_row.columns:
            val = next_row[col].iloc[0] if hasattr(
                next_row[col], 'iloc') else next_row[col]
            if pd.isna(val) or val == 0:
                # Try to get from recent data
                if col in df.columns:
                    recent_val = df[col].tail(24).mean() if len(
                        df) >= 24 else df[col].mean()
                    if not pd.isna(recent_val) and recent_val != 0:
                        next_row[col] = recent_val

    X = next_row[features].fillna(method="ffill").fillna(method="bfill")
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

    daily, monthly = _build_daily_and_monthly(hourly)
    # Persist history for dashboard use
    hourly_history = hourly[["timestamp", "Date", "HE", TARGET_COL]].copy()
    hourly_history.to_csv(RESULTS_DIR / "hourly_history.csv", index=False)
    daily.to_csv(RESULTS_DIR / "daily_history.csv", index=False)
    monthly.to_csv(RESULTS_DIR / "monthly_history.csv", index=False)

    hourly_model, hourly_features = _load_model(HOURLY_MODEL_PATH)
    daily_model, daily_features = _load_model(DAILY_MODEL_PATH)
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

    # Generate predictions for all periods from last data to current time
    all_predictions = []

    # Hourly predictions: from last hour + 1 to current hour
    if last_hourly_timestamp < pd.to_datetime(current_hour):
        hourly_df = hourly_sorted.rename(columns={"timestamp": "date"})
        current_pred_hour = last_hourly_timestamp + timedelta(hours=1)

        while current_pred_hour <= pd.to_datetime(current_hour):
            pred = _predict_next(
                hourly_model,
                hourly_features,
                hourly_df,
                "date",
                f"hour_{current_pred_hour.strftime('%Y-%m-%d %H:00')}",
                freq="H",
                target_date=current_pred_hour,
            )
            all_predictions.append(pred)

            # Update hourly_df with the prediction for next iteration
            # Add a dummy row with the predicted timestamp for feature calculation
            new_row = hourly_df.iloc[-1:].copy()
            new_row["date"] = current_pred_hour
            new_row[TARGET_COL] = pred["prediction"].iloc[0]
            hourly_df = pd.concat([hourly_df, new_row], ignore_index=True)

            current_pred_hour += timedelta(hours=1)

    # Daily predictions: from last day + 1 to current day
    if last_daily_date < pd.to_datetime(current_day):
        daily_df = daily.rename(columns={"date": "date"})
        current_pred_day = last_daily_date + timedelta(days=1)

        while current_pred_day <= pd.to_datetime(current_day):
            pred = _predict_next(
                daily_model,
                daily_features,
                daily_df,
                "date",
                f"day_{current_pred_day.strftime('%Y-%m-%d')}",
                freq="D",
                target_date=current_pred_day,
            )
            all_predictions.append(pred)

            # Update daily_df with the prediction for next iteration
            new_row = daily_df.iloc[-1:].copy()
            new_row["date"] = current_pred_day
            new_row[TARGET_COL] = pred["prediction"].iloc[0]
            daily_df = pd.concat([daily_df, new_row], ignore_index=True)

            current_pred_day += timedelta(days=1)

    # Monthly predictions: from last month + 1 to current month
    if last_monthly_date < pd.to_datetime(current_month):
        monthly_df = monthly.rename(columns={"year_month_start": "date"})
        current_pred_month = pd.to_datetime(
            last_monthly_date) + DateOffset(months=1)

        while current_pred_month <= pd.to_datetime(current_month):
            pred = _predict_next(
                monthly_model,
                monthly_features,
                monthly_df,
                "date",
                f"month_{current_pred_month.strftime('%Y-%m')}",
                freq="M",
                target_date=current_pred_month,
            )
            all_predictions.append(pred)

            # Update monthly_df with the prediction for next iteration
            new_row = monthly_df.iloc[-1:].copy()
            new_row["date"] = current_pred_month
            new_row[TARGET_COL] = pred["prediction"].iloc[0]
            monthly_df = pd.concat([monthly_df, new_row], ignore_index=True)

            current_pred_month += DateOffset(months=1)

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
        results = pd.concat([next_hour_pred, next_day_pred,
                            next_month_pred], ignore_index=True)

    out_path = RESULTS_DIR / "predictions.csv"
    results.to_csv(out_path, index=False)
    return out_path, results


if __name__ == "__main__":
    path, df = run_inference()
    print(f"Saved predictions to {path}")
    print(df)
