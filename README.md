# California Residential Energy Spending Prediction

Hayley Chang and Madeleine Marangoz

## Overview

This project addresses the challenge of predicting and optimizing residential electricity costs in California. The region experiences some of the highest electricity demand in the United States due to widespread air-conditioning use, growing electric vehicle adoption, and increasing residential energy consumption. During hot summer months, cooling loads drive peak demand in the late afternoon and evening, while electric vehicle charging and household activities further elevate nighttime consumption.

**The Problem:** Hourly electricity prices are not always publicly available for California, and consumers rarely have access to clear or actionable guidance about when electricity is most affordable.

**The Solution:** This project estimates hourly electricity costs using available demand, generation, and weather data, and forecasts prices for future periods. With these predictions, the system provides users with intuitive recommendations about the best times to use electricity.

## The Prediction Problem

The core prediction problem is to forecast residential electricity spending at multiple time granularities (hourly, daily, weekly, monthly) using historical patterns and external features. Specifically:

**Input Features:**

- Historical CAISO system load data (total electricity demand)
- Energy generation mix (solar, wind, natural gas, hydro, etc.)
- Weather data (temperature, heating/cooling degree days)
- Historical electricity prices and costs
- Temporal patterns (hour of day, day of week, month, season)
- Lag features (previous hours/days patterns)

**Target Variable:**

- `Estimated_Hourly_Cost_USD`: The estimated residential electricity cost per hour, calculated as:
  ```
  (CAISO Total Load / mean_load) × RESIDENTIAL_KWH_PER_HOUR × (price / 100.0)
  ```

**Prediction Challenges:**

1. **Temporal Dependencies**: Energy costs exhibit strong temporal patterns (daily cycles, weekly patterns, seasonal variations)
2. **Non-linear Relationships**: Complex interactions between load, generation mix, weather, and prices
3. **Feature Engineering**: Need to capture lag effects, rolling statistics, and aggregated patterns
4. **Multi-scale Predictions**: Different models needed for different time granularities (hourly vs. monthly)
5. **Systematic Bias**: Models tend to underpredict, requiring calibration factors
6. **Data Quality**: Handling missing data, outliers, and ensuring feature consistency across time periods

**Solution Approach:**

- Use ensemble tree-based models (Random Forest and Histogram-based Gradient Boosting) that can capture non-linear relationships
- Engineer temporal features (lags, rolling means, day-of-week patterns)
- Train separate models for each time granularity to optimize performance
- Apply calibration factors to correct systematic underprediction
- Use time-based train/test splits to ensure realistic evaluation

## What This Project Does

This project provides:

1. **Machine Learning Models** that predict residential energy spending at multiple time granularities:

   - **Hourly predictions** - Forecast energy costs for each hour of the day
   - **Daily predictions** - Predict total daily spending
   - **Weekly predictions** - Forecast weekly energy costs
   - **Monthly predictions** - Predict monthly spending totals

2. **Feature Engineering** that combines multiple data sources:

   - CAISO system load data
   - Energy generation mix (solar, wind, natural gas, etc.)
   - Weather data (temperature, degree days)
   - Historical electricity prices
   - Lag features (previous hours/days patterns)

3. **Interactive Dashboard** that visualizes:

   - Historical and predicted energy spending trends
   - Best times to use electricity (cost optimization)
   - Detailed data breakdowns with filtering and sorting
   - Model performance evaluations
   - AI-powered energy usage recommendations

4. **Automated Pipeline** that:
   - Refreshes feature data (load, energy mix, weather, prices)
   - Trains models on updated data
   - Generates predictions for future periods
   - Evaluates model performance
   - Deploys updated dashboard automatically

## Technologies Overview

This project leverages a modern technology stack for data processing, machine learning, and web deployment:

### Machine Learning & Data Science

- **Python 3.11+**: Core programming language
- **Pandas**: Data manipulation, aggregation, and time-series processing
- **NumPy**: Numerical computations and array operations
- **Scikit-learn**: Machine learning models and evaluation metrics
  - `HistGradientBoostingRegressor`: For hourly and daily predictions
  - `RandomForestRegressor`: For weekly and monthly predictions
- **Joblib**: Model serialization and persistence

### Data Sources & APIs

- **CAISO (California Independent System Operator)**: System load and generation data
- **EIA (U.S. Energy Information Administration) API**: Historical electricity prices
- **Weather APIs**: Temperature and weather data for California

### Web Technologies

- **HTML5/CSS3**: Dashboard structure and styling
- **JavaScript (ES6+)**: Interactive dashboard functionality
- **Plotly.js**: Interactive data visualization and charting
- **Bootstrap 5**: Responsive UI framework
- **Flatpickr**: Date range selection component

### Deployment & Automation

- **GitHub Actions**: CI/CD pipeline for automated training and deployment
- **GitHub Pages**: Static site hosting for the dashboard
- **Python Scripts**: Automated data fetching, model training, and dashboard generation

### Development Tools

- **Static Site Generation**: Python-based HTML generation (`build_static.py`)
- **JSON**: Data serialization for dashboard data files
- **CSV**: Historical data storage and exchange

### Architecture Patterns

- **Static Site Generation**: Pre-rendered HTML with embedded data
- **Lazy Loading**: On-demand data fetching for performance optimization
- **Model Artifacts**: Serialized models stored as artifacts for reuse
- **Time-series Processing**: Temporal feature engineering and aggregation

## Model Architecture

The project uses ensemble tree-based models:

- **Hourly Model**: `HistGradientBoostingRegressor` - Optimized for capturing hourly patterns and variations
- **Daily Model**: `HistGradientBoostingRegressor` - Handles daily spending patterns with improved accuracy
- **Weekly Model**: `RandomForestRegressor` - Aggregates daily predictions to provide weekly forecasts
- **Monthly Model**: `RandomForestRegressor` - Provides monthly spending predictions

All models include:

- **Calibration factors** to correct systematic underprediction
- **Feature engineering** with temporal patterns, lag features, and aggregated statistics
- **Time-based train/test splits** to ensure realistic evaluation

## Training Schedule

The models are automatically trained and updated via GitHub Actions:

- **Weekly Full Retrain**: Every Monday at 08:00 UTC

  - Models are retrained on all available historical data
  - Feature data is refreshed (load, energy mix, weather, prices)
  - New predictions are generated
  - Dashboard is rebuilt and deployed

- **Daily Inference Refresh**: Every day at 07:00 UTC

  - New predictions are generated using existing models
  - Dashboard is updated with latest forecasts
  - Models are only retrained if they're missing (e.g., after artifact expiration)

- **On Code Push**: When code is pushed to the `main` branch
  - If models are missing, they are automatically trained
  - New predictions are generated
  - Dashboard is rebuilt and deployed

Model artifacts are retained for 7 days. If models expire and are not found, they will be automatically retrained on the next push or weekly schedule.

## Dashboard

The interactive dashboard is available at:

**https://haysho2260.github.io/scalable-final-project/**

### Dashboard Features

- **Multi-granularity Views**: Switch between hourly, daily, weekly, monthly, and yearly views
- **Interactive Charts**: Plotly.js charts with zoom, pan, and hover details
- **Date Range Selection**: Filter data by specific date ranges
- **Navigation Controls**: Navigate between periods with next/previous buttons
- **Detailed Data Tables**: Sortable and filterable tables showing all data points
- **Energy Usage Recommendations**: AI-powered suggestions for optimal electricity usage times
- **Model Evaluation**: Performance metrics and prediction vs. actual comparisons
- **Today Button**: Quick navigation to current hour/day/week/month

## Project Structure

```
.
├── features/          # Feature engineering scripts
│   ├── get_load.py           # CAISO load data
│   ├── get_energy_types.py   # Energy generation mix
│   ├── get_weather.py        # Weather data
│   └── get_hourly_price.py   # Price and cost estimation
├── model/            # Model training and inference
│   ├── train.py              # Model training script
│   ├── inference.py          # Prediction generation
│   └── evaluate.py           # Model evaluation (backtesting)
├── dashboard/        # Dashboard generation
│   └── build_static.py       # Static HTML dashboard builder
├── results/          # Output files
│   ├── predictions.csv       # Generated predictions
│   ├── hourly_history.csv    # Historical hourly data
│   ├── daily_history.csv     # Historical daily data
│   ├── monthly_history.csv   # Historical monthly data
│   └── evaluation_YYYY/      # Model evaluation results
└── site/            # Deployed dashboard files
    └── index.html            # Main dashboard page
```

## Key Benefits

By helping consumers shift demand to lower-cost hours, this project supports:

- **Economic Savings**: Reduce household energy bills by using electricity during cheaper hours
- **Grid Reliability**: Distribute demand more evenly across the day
- **Renewable Integration**: Encourage usage during periods of high renewable generation
- **Sustainable Behavior**: Promote energy-conscious decision making

## Technical Details

### Model Features

The models use a combination of:

- **Temporal features**: Hour of day, day of week, month, year
- **Load features**: CAISO total load, load lags
- **Price features**: Monthly price per kWh, price lags
- **Weather features**: Temperature, heating/cooling degree days
- **Energy mix features**: Solar, wind, natural gas generation percentages
- **Aggregated statistics**: Rolling means, standard deviations

### Evaluation

Models are evaluated using:

- **Mean Absolute Error (MAE)**: Average prediction error
- **Root Mean Squared Error (RMSE)**: Penalizes larger errors
- **Mean Absolute Percentage Error (MAPE)**: Percentage-based error metric
- **R² Score**: Coefficient of determination
- **Backtesting**: Models are trained on historical data and evaluated on held-out years

## Requirements

- Python 3.11+
- See `requirements.txt` for dependencies
- EIA API key (for price data) - set as `EIA_API_KEY` environment variable

## License

This project is part of a scalable final project for academic purposes.
