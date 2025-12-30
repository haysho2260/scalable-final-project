"""Evaluate model performance by training on historical data and predicting the last complete year.

This script:
1. Dynamically determines the last complete year of data
2. Trains models on data up to the start of that year
3. Makes predictions for that full year
4. Compares predictions to actual data
5. Calculates comprehensive evaluation metrics
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)
import joblib

ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT / "features"
RESULTS_DIR = ROOT / "results"
MODEL_DIR = ROOT / "model"

# Import functions from train.py and inference.py
import sys
sys.path.insert(0, str(ROOT))
from model.train import build_hourly_dataset
from model.inference import _predict_next, _build_daily_and_monthly

TARGET_COL = "Estimated_Hourly_Cost_USD"


def get_last_complete_year() -> int:
    """Determine the last complete year of data available.
    
    A year is considered complete if it has data for at least 11 months
    (to account for partial data in the current year).
    """
    hourly = build_hourly_dataset()
    hourly["date"] = pd.to_datetime(hourly["Date"])
    
    # Get all unique years in the data
    hourly["year"] = hourly["date"].dt.year
    years = sorted(hourly["year"].unique(), reverse=True)
    
    if not years:
        raise ValueError("No data found in dataset")
    
    # Check each year from most recent backwards
    current_year = datetime.now().year
    for year in years:
        if year >= current_year:
            continue  # Skip current year and future years
        
        # Check if this year has substantial data (at least 11 months)
        year_data = hourly[hourly["year"] == year]
        if year_data.empty:
            continue
        
        months_in_year = year_data["date"].dt.month.nunique()
        if months_in_year >= 11:
            print(f"Found complete year: {year} (with {months_in_year} months of data)")
            return year
    
    # If no complete year found, use the most recent year with data
    if years:
        most_recent = max([y for y in years if y < current_year], default=years[-1])
        print(f"Warning: No complete year found. Using most recent year with data: {most_recent}")
        return most_recent
    
    raise ValueError("No suitable year found for evaluation")


def train_on_historical_data(cutoff_date: str):
    """Train models on data up to cutoff_date."""
    print(f"Building dataset and training models on data up to {cutoff_date}...")
    
    hourly = build_hourly_dataset()
    hourly["date"] = pd.to_datetime(hourly["Date"])
    
    # Filter to only data before cutoff
    cutoff = pd.to_datetime(cutoff_date)
    hourly_train = hourly[hourly["date"] < cutoff].copy()
    
    if hourly_train.empty:
        raise ValueError(f"No training data found before {cutoff_date}")
    
    print(f"Training data: {len(hourly_train)} hourly records from {hourly_train['date'].min()} to {hourly_train['date'].max()}")
    
    # Build aggregated datasets
    daily_train, weekly_train, monthly_train = _build_daily_and_monthly(hourly_train)
    
    # Train models
    models = {}
    
    # Hourly model
    print("\nTraining hourly model...")
    hourly_y = hourly_train[TARGET_COL]
    exclude = {"timestamp", "Date", "HE", "date", "year_month", TARGET_COL}
    hourly_features = [c for c in hourly_train.columns 
                      if c not in exclude and pd.api.types.is_numeric_dtype(hourly_train[c])]
    hourly_X = hourly_train[hourly_features].ffill().bfill()
    
    hourly_model = HistGradientBoostingRegressor(
        max_depth=12,
        max_iter=500,
        learning_rate=0.03,
        min_samples_leaf=5,
        l2_regularization=0.1,
        random_state=42
    )
    hourly_model.fit(hourly_X, hourly_y)
    models["hourly"] = (hourly_model, hourly_features)
    
    # Daily model
    print("Training daily model...")
    daily_y = daily_train[TARGET_COL]
    exclude = {"date", "year_month", "week_start", TARGET_COL}
    daily_features = [c for c in daily_train.columns 
                     if c not in exclude and pd.api.types.is_numeric_dtype(daily_train[c])]
    daily_X = daily_train[daily_features].ffill().bfill()
    
    daily_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    daily_model.fit(daily_X, daily_y)
    models["daily"] = (daily_model, daily_features)
    
    # Weekly model
    print("Training weekly model...")
    weekly_y = weekly_train[TARGET_COL]
    exclude = {"week_start", "year_month", TARGET_COL}
    weekly_features = [c for c in weekly_train.columns 
                      if c not in exclude and pd.api.types.is_numeric_dtype(weekly_train[c])]
    weekly_X = weekly_train[weekly_features].ffill().bfill()
    
    weekly_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    weekly_model.fit(weekly_X, weekly_y)
    models["weekly"] = (weekly_model, weekly_features)
    
    # Monthly model
    print("Training monthly model...")
    monthly_y = monthly_train[TARGET_COL]
    exclude = {"year_month", "year_month_start", TARGET_COL}
    monthly_features = [c for c in monthly_train.columns 
                       if c not in exclude and pd.api.types.is_numeric_dtype(monthly_train[c])]
    monthly_X = monthly_train[monthly_features].ffill().bfill()
    
    monthly_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    monthly_model.fit(monthly_X, monthly_y)
    models["monthly"] = (monthly_model, monthly_features)
    
    return models, {
        "hourly": (hourly_train, hourly_X, hourly_y),
        "daily": (daily_train, daily_X, daily_y),
        "weekly": (weekly_train, weekly_X, weekly_y),
        "monthly": (monthly_train, monthly_X, monthly_y),
    }


def predict_year(models, train_data, eval_year: int):
    """Make predictions for a specific year using trained models."""
    print("\n" + "="*60)
    print(f"Making predictions for {eval_year}...")
    print("="*60)
    
    # Load full dataset to get actuals for the evaluation year
    hourly_full = build_hourly_dataset()
    hourly_full["date"] = pd.to_datetime(hourly_full["Date"])
    
    # Get data for the evaluation year
    year_start = pd.to_datetime(f"{eval_year}-01-01")
    year_end = pd.to_datetime(f"{eval_year}-12-31")
    hourly_year = hourly_full[
        (hourly_full["date"] >= year_start) & 
        (hourly_full["date"] <= year_end)
    ].copy()
    
    if hourly_year.empty:
        raise ValueError(f"No {eval_year} data found for evaluation")
    
    print(f"{eval_year} data: {len(hourly_year)} hourly records from {hourly_year['date'].min()} to {hourly_year['date'].max()}")
    
    # Build aggregated datasets for the evaluation year
    daily_year, weekly_year, monthly_year = _build_daily_and_monthly(hourly_year)
    
    predictions = {}
    actuals = {}
    
    # Prepare training dataframes for prediction
    hourly_train_df = train_data["hourly"][0].copy()
    if "date" not in hourly_train_df.columns:
        hourly_train_df["date"] = pd.to_datetime(hourly_train_df["Date"])
    
    daily_train_df = train_data["daily"][0].copy()
    if "date" not in daily_train_df.columns:
        daily_train_df["date"] = pd.to_datetime(daily_train_df["date"])
    
    weekly_train_df = train_data["weekly"][0].copy()
    if "week_start" in weekly_train_df.columns:
        weekly_train_df = weekly_train_df.rename(columns={"week_start": "date"})
    if "date" not in weekly_train_df.columns:
        weekly_train_df["date"] = pd.to_datetime(weekly_train_df["date"])
    
    monthly_train_df = train_data["monthly"][0].copy()
    if "year_month_start" in monthly_train_df.columns:
        monthly_train_df = monthly_train_df.rename(columns={"year_month_start": "date"})
    if "date" not in monthly_train_df.columns:
        monthly_train_df["date"] = pd.to_datetime(monthly_train_df["date"])
    
    # Skip hourly predictions for now (too many, focus on daily/weekly/monthly)
    print("\nSkipping hourly predictions (focusing on daily/weekly/monthly for evaluation)...")
    
    # Daily predictions
    print("\nGenerating daily predictions...")
    daily_preds = []
    daily_actuals = []
    daily_dates = []
    
    daily_sorted = daily_train_df.sort_values("date")
    current_date = daily_sorted["date"].iloc[-1] + pd.Timedelta(days=1)
    end_date = daily_year["date"].max()
    
    daily_df = daily_train_df.copy()
    count = 0
    while current_date <= end_date and count < 400:
        if count % 30 == 0:
            print(f"  Progress: {count} days predicted")
        
        try:
            pred = _predict_next(
                models["daily"][0],
                models["daily"][1],
                daily_df.rename(columns={"date": "date"}),
                "date",
                f"day_{current_date}",
                freq="D",
                target_date=current_date,
            )
            
            # Get actual value if available
            actual_row = daily_year[daily_year["date"] == current_date]
            if not actual_row.empty:
                daily_preds.append(pred["prediction"].iloc[0])
                daily_actuals.append(actual_row[TARGET_COL].iloc[0])
                daily_dates.append(current_date)
            
            # Update dataframe
            new_row = daily_df.iloc[-1:].copy()
            new_row["date"] = current_date
            new_row[TARGET_COL] = pred["prediction"].iloc[0]
            daily_df = pd.concat([daily_df, new_row], ignore_index=True)
            
            current_date += pd.Timedelta(days=1)
            count += 1
        except Exception as e:
            print(f"  Error at {current_date}: {e}")
            break
    
    if daily_preds:
        predictions["daily"] = pd.DataFrame({
            "date": daily_dates,
            "prediction": daily_preds,
            "actual": daily_actuals
        })
    
    # Weekly predictions
    print("\nGenerating weekly predictions...")
    weekly_preds = []
    weekly_actuals = []
    weekly_dates = []
    
    weekly_sorted = weekly_train_df.sort_values("date")
    current_date = weekly_sorted["date"].iloc[-1] + pd.Timedelta(weeks=1)
    end_date = weekly_year["week_start"].max() if "week_start" in weekly_year.columns else weekly_year["date"].max()
    
    weekly_df = weekly_train_df.copy()
    count = 0
    while current_date <= end_date and count < 60:
        if count % 10 == 0:
            print(f"  Progress: {count} weeks predicted")
        
        try:
            pred = _predict_next(
                models["weekly"][0],
                models["weekly"][1],
                weekly_df.rename(columns={"date": "date"}),
                "date",
                f"week_{current_date}",
                freq="W",
                target_date=current_date,
            )
            
            # Get actual value
            if "week_start" in weekly_year.columns:
                actual_row = weekly_year[weekly_year["week_start"] == current_date]
            else:
                actual_row = weekly_year[weekly_year["date"] == current_date]
            
            if not actual_row.empty:
                weekly_preds.append(pred["prediction"].iloc[0])
                weekly_actuals.append(actual_row[TARGET_COL].iloc[0])
                weekly_dates.append(current_date)
            
            # Update dataframe
            new_row = weekly_df.iloc[-1:].copy()
            new_row["date"] = current_date
            new_row[TARGET_COL] = pred["prediction"].iloc[0]
            weekly_df = pd.concat([weekly_df, new_row], ignore_index=True)
            
            current_date += pd.Timedelta(weeks=1)
            count += 1
        except Exception as e:
            print(f"  Error at {current_date}: {e}")
            break
    
    if weekly_preds:
        predictions["weekly"] = pd.DataFrame({
            "date": weekly_dates,
            "prediction": weekly_preds,
            "actual": weekly_actuals
        })
    
    # Monthly predictions
    print("\nGenerating monthly predictions...")
    monthly_preds = []
    monthly_actuals = []
    monthly_dates = []
    
    monthly_sorted = monthly_train_df.sort_values("date")
    current_date = monthly_sorted["date"].iloc[-1] + relativedelta(months=1)
    end_date = monthly_year["year_month_start"].max() if "year_month_start" in monthly_year.columns else monthly_year["date"].max()
    
    monthly_df = monthly_train_df.copy()
    count = 0
    while current_date <= end_date and count < 15:
        print(f"  Predicting month {current_date.strftime('%Y-%m')}...")
        
        try:
            pred = _predict_next(
                models["monthly"][0],
                models["monthly"][1],
                monthly_df.rename(columns={"date": "date"}),
                "date",
                f"month_{current_date}",
                freq="M",
                target_date=current_date,
            )
            
            # Get actual value
            if "year_month_start" in monthly_year.columns:
                actual_row = monthly_year[monthly_year["year_month_start"] == current_date]
            else:
                actual_row = monthly_year[monthly_year["date"] == current_date]
            
            if not actual_row.empty:
                monthly_preds.append(pred["prediction"].iloc[0])
                monthly_actuals.append(actual_row[TARGET_COL].iloc[0])
                monthly_dates.append(current_date)
            
            # Update dataframe
            new_row = monthly_df.iloc[-1:].copy()
            new_row["date"] = current_date
            new_row[TARGET_COL] = pred["prediction"].iloc[0]
            monthly_df = pd.concat([monthly_df, new_row], ignore_index=True)
            
            current_date += relativedelta(months=1)
            count += 1
        except Exception as e:
            print(f"  Error at {current_date}: {e}")
            break
    
    if monthly_preds:
        predictions["monthly"] = pd.DataFrame({
            "date": monthly_dates,
            "prediction": monthly_preds,
            "actual": monthly_actuals
        })
    
    return predictions


def calculate_metrics(predictions: dict) -> dict:
    """Calculate comprehensive evaluation metrics."""
    print("\n" + "="*60)
    print("Calculating Evaluation Metrics")
    print("="*60)
    
    all_metrics = {}
    
    for granularity, df in predictions.items():
        if df.empty:
            continue
        
        actual = df["actual"].values
        pred = df["prediction"].values
        
        # Calculate metrics
        mae = mean_absolute_error(actual, pred)
        mse = mean_squared_error(actual, pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(actual, pred) * 100  # Convert to percentage
        r2 = r2_score(actual, pred)
        
        # Additional metrics
        mean_actual = np.mean(actual)
        mean_pred = np.mean(pred)
        median_actual = np.median(actual)
        median_pred = np.median(pred)
        
        # Percentage errors
        mape_clipped = np.mean(np.abs((actual - pred) / np.maximum(np.abs(actual), 1e-8))) * 100
        
        metrics = {
            "n_samples": len(df),
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "r2": r2,
            "mean_actual": mean_actual,
            "mean_predicted": mean_pred,
            "median_actual": median_actual,
            "median_predicted": median_pred,
            "mean_error": mean_pred - mean_actual,
            "mean_abs_error": mae,
            "rmse_percentage": (rmse / mean_actual * 100) if mean_actual > 0 else 0,
        }
        
        all_metrics[granularity] = metrics
        
        print(f"\n{granularity.upper()} Metrics:")
        print(f"  Samples: {metrics['n_samples']}")
        print(f"  MAE (Mean Absolute Error): ${metrics['mae']:.4f}")
        print(f"  RMSE (Root Mean Squared Error): ${metrics['rmse']:.4f}")
        print(f"  MAPE (Mean Absolute Percentage Error): {metrics['mape']:.2f}%")
        print(f"  R² (Coefficient of Determination): {metrics['r2']:.4f}")
        print(f"  Mean Actual: ${metrics['mean_actual']:.2f}")
        print(f"  Mean Predicted: ${metrics['mean_predicted']:.2f}")
        print(f"  Mean Error: ${metrics['mean_error']:.2f}")
        print(f"  RMSE as % of Mean: {metrics['rmse_percentage']:.2f}%")
    
    return all_metrics


def main():
    # Determine the last complete year dynamically
    eval_year = get_last_complete_year()
    cutoff_date = f"{eval_year}-01-01"
    
    print("="*60)
    print(f"Model Evaluation: Training on data up to {eval_year-1}, Predicting {eval_year}")
    print("="*60)
    
    # Train models on data up to the start of the evaluation year
    models, train_data = train_on_historical_data(cutoff_date=cutoff_date)
    
    # Make predictions for the evaluation year
    predictions = predict_year(models, train_data, eval_year)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions)
    
    # Save results with dynamic year in directory name
    output_dir = RESULTS_DIR / f"evaluation_{eval_year}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for granularity, df in predictions.items():
        if not df.empty:
            df.to_csv(output_dir / f"{granularity}_predictions_vs_actual.csv", index=False)
            print(f"\nSaved {granularity} predictions to {output_dir / f'{granularity}_predictions_vs_actual.csv'}")
    
    # Save metrics summary
    import json
    with open(output_dir / "metrics_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics summary to {output_dir / 'metrics_summary.json'}")
    
    print("\n" + "="*60)
    print(f"Evaluation Complete for {eval_year}!")
    print("="*60)


if __name__ == "__main__":
    main()

