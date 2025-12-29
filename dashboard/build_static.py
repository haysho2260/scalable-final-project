"""Build a static dashboard from results files focused on residential spending.

Outputs a standalone HTML at `site/index.html` showing:
- Residential spending at hourly, daily, weekly, monthly, yearly levels
- Peak spending period analysis to help users focus on cost reduction
- Actionable insights and recommendations
"""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
SITE_DIR = ROOT / "site"
FEATURES_DIR = ROOT / "features"

# California has approximately 13 million households (2020-2024 estimate)
CA_HOUSEHOLDS = 13_000_000


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _load_hourly_data() -> pd.DataFrame:
    """Load hourly data from hourly_price files."""
    hourly_files = sorted(
        (FEATURES_DIR / "hourly_price").glob("CAISO_Price_*.csv"))
    if not hourly_files:
        return pd.DataFrame()

    frames = []
    for f in hourly_files[-12:]:  # Last 12 months for performance
        try:
            df = pd.read_csv(f)
            if "Date" in df.columns and "HE" in df.columns and "Estimated_Hourly_Cost_USD" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df["HE"] = pd.to_numeric(df["HE"], errors="coerce")
                df = df.dropna(
                    subset=["Date", "HE", "Estimated_Hourly_Cost_USD"])
                df["timestamp"] = df["Date"] + \
                    pd.to_timedelta(df["HE"] - 1, unit="h")
                frames.append(
                    df[["timestamp", "Estimated_Hourly_Cost_USD", "Date", "HE"]])
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    hourly = pd.concat(frames, ignore_index=True)
    hourly = hourly.sort_values(
        "timestamp").drop_duplicates(subset="timestamp")
    hourly["hour"] = hourly["timestamp"].dt.hour
    # Ensure hours are in valid range 0-23
    hourly["hour"] = hourly["hour"].clip(0, 23)
    hourly["dayofweek"] = hourly["timestamp"].dt.dayofweek
    hourly["month"] = hourly["timestamp"].dt.month
    hourly["year"] = hourly["timestamp"].dt.year
    return hourly


def build():
    SITE_DIR.mkdir(parents=True, exist_ok=True)

    preds = _read_csv(RESULTS_DIR / "predictions.csv")
    hourly_history = _read_csv(RESULTS_DIR / "hourly_history.csv")
    daily = _read_csv(RESULTS_DIR / "daily_history.csv")
    monthly = _read_csv(RESULTS_DIR / "monthly_history.csv")
    hourly_features = _load_hourly_data()

    # Use hourly history if available, otherwise fall back to features
    if not hourly_history.empty and "Estimated_Hourly_Cost_USD" in hourly_history.columns:
        hourly = hourly_history.copy()
        if "timestamp" in hourly.columns:
            hourly["timestamp"] = pd.to_datetime(
                hourly["timestamp"], errors="coerce")
        elif "Date" in hourly.columns and "HE" in hourly.columns:
            hourly["Date"] = pd.to_datetime(hourly["Date"], errors="coerce")
            hourly["HE"] = pd.to_numeric(hourly["HE"], errors="coerce")
            hourly = hourly.dropna(
                subset=["Date", "HE", "Estimated_Hourly_Cost_USD"])
            hourly["timestamp"] = hourly["Date"] + \
                pd.to_timedelta(hourly["HE"] - 1, unit="h")
        else:
            hourly = hourly_features
        if "timestamp" in hourly.columns:
            hourly["hour"] = hourly["timestamp"].dt.hour
            # Ensure hours are in valid range 0-23
            hourly["hour"] = hourly["hour"].clip(0, 23)
            hourly["dayofweek"] = hourly["timestamp"].dt.dayofweek
            hourly["month"] = hourly["timestamp"].dt.month
            hourly["year"] = hourly["timestamp"].dt.year
            if "Date" not in hourly.columns:
                hourly["Date"] = hourly["timestamp"].dt.date
    else:
        hourly = hourly_features

    # Process data for unified view
    unified_data = []

    # 1. Historical Data
    # Hourly
    if not hourly.empty:
        for _, row in hourly.iterrows():
            ts = row["timestamp"]
            unified_data.append({
                "date": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "year": int(ts.year),
                "month": ts.strftime("%B"),  # Full month name
                "day": int(ts.day),
                "hour": int(ts.hour),
                "val": float(row["Estimated_Hourly_Cost_USD"]),
                "for": "hourly",
                "type": "historical"
            })

    # Daily
    if not daily.empty:
        for _, row in daily.iterrows():
            dt = pd.to_datetime(row["date"])
            unified_data.append({
                "date": dt.strftime("%Y-%m-%d"),
                "year": int(dt.year),
                "month": dt.strftime("%B"),  # Full month name
                "day": int(dt.day),
                "val": float(row["Estimated_Hourly_Cost_USD"]),
                "for": "daily",
                "type": "historical"
            })

    # Weekly
    if not daily.empty:
        daily["date_dt"] = pd.to_datetime(daily["date"])
        daily["week_start"] = daily["date_dt"] - \
            pd.to_timedelta(daily["date_dt"].dt.dayofweek, unit="d")
        weekly = daily.groupby("week_start")[
            "Estimated_Hourly_Cost_USD"].sum().reset_index()
        for _, row in weekly.iterrows():
            week_start = row["week_start"]
            week_end = week_start + pd.Timedelta(days=6)
            # Format as "Jan 1-7, 2025" or "Dec 29 - Jan 4, 2025" if spanning months
            if week_start.month == week_end.month:
                date_str = week_start.strftime(
                    "%b %d") + "-" + week_end.strftime("%d, %Y")
            else:
                date_str = week_start.strftime(
                    "%b %d") + " - " + week_end.strftime("%b %d, %Y")
            unified_data.append({
                # Keep ISO format for sorting/filtering
                "date": week_start.strftime("%Y-%m-%d"),
                "date_display": date_str,  # Human-readable format
                "year": int(week_start.year),
                "month": week_start.strftime("%B"),  # Full month name
                "val": float(row["Estimated_Hourly_Cost_USD"]),
                "for": "weekly",
                "type": "historical"
            })

    # Monthly
    if not monthly.empty:
        for _, row in monthly.iterrows():
            month_date = pd.to_datetime(row["year_month_start"])
            # Format as "January 2025" or "Jan 2025"
            date_str = month_date.strftime("%B %Y")  # Full month name
            unified_data.append({
                # Keep ISO format for sorting/filtering
                "date": month_date.strftime("%Y-%m-%d"),
                "date_display": date_str,  # Human-readable format
                "year": int(month_date.year),
                "month": month_date.strftime("%B"),  # Full month name
                "val": float(row["Estimated_Hourly_Cost_USD"]),
                "for": "monthly",
                "type": "historical"
            })

    # 2. Prediction Data
    if not preds.empty:
        preds["feature_date"] = pd.to_datetime(preds["feature_date"])
        for _, row in preds.iterrows():
            gran = str(row["for"]).lower()
            if "hour" in gran:
                g = "hourly"
            elif "day" in gran:
                g = "daily"
            elif "week" in gran:
                g = "weekly"
            elif "month" in gran:
                g = "monthly"
            else:
                g = "unknown"

            # Format date based on granularity for consistency
            d_val = row["feature_date"]
            if g == "monthly":
                d_str = d_val.strftime("%Y-%m-%d")
                date_display = d_val.strftime("%B %Y")  # "January 2025"
                unified_data.append({
                    "date": d_str,
                    "date_display": date_display,
                    "year": int(d_val.year),
                    "month": d_val.strftime("%B"),  # Full month name
                    "val": float(row["prediction"]),
                    "for": g,
                    "type": "prediction"
                })
            elif g == "hourly":
                d_str = d_val.strftime("%Y-%m-%d %H:%M:%S")
                date_display = d_str
                unified_data.append({
                    "date": d_str,
                    "date_display": date_display,
                    "year": int(d_val.year),
                    "month": d_val.strftime("%B"),  # Full month name
                    "day": int(d_val.day),
                    "hour": int(d_val.hour),
                    "val": float(row["prediction"]),
                    "for": g,
                    "type": "prediction"
                })
            elif g == "weekly":
                # Calculate week range
                week_start = d_val - pd.Timedelta(days=d_val.dayofweek)
                week_end = week_start + pd.Timedelta(days=6)
                d_str = week_start.strftime("%Y-%m-%d")
                if week_start.month == week_end.month:
                    date_display = week_start.strftime(
                        "%b %d") + "-" + week_end.strftime("%d, %Y")
                else:
                    date_display = week_start.strftime(
                        "%b %d") + " - " + week_end.strftime("%b %d, %Y")
                unified_data.append({
                    "date": d_str,
                    "date_display": date_display,
                    "year": int(week_start.year),
                    "month": week_start.strftime("%B"),  # Full month name
                    "val": float(row["prediction"]),
                    "for": g,
                    "type": "prediction"
                })
            else:
                d_str = d_val.strftime("%Y-%m-%d")
                date_display = d_str
                unified_data.append({
                    "date": d_str,
                    "date_display": date_display,
                    "year": int(d_val.year),
                    "month": d_val.strftime("%B"),  # Full month name
                    "day": int(d_val.day),
                    "val": float(row["prediction"]),
                    "for": g,
                    "type": "prediction"
                })

    html_parts = [
        "<html>",
        "<head>",
        "<title>California Residential Energy Spending: History & Predictions</title>",
        "<link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css'>",
        "<link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/flatpickr/dist/flatpickr.min.css'>",
        "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
        "<script src='https://cdn.jsdelivr.net/npm/flatpickr'></script>",
        "<style>",
        "body { font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f8f9fa; padding-top: 56px; }",
        ".navbar { z-index: 1030; margin-bottom: 0 !important; width: 100%; }",
        ".sidebar { background-color: #212529; color: #e5e7eb; min-height: calc(100vh - 56px); padding: 1rem; position: fixed; left: 0; top: 56px; z-index: 1020; transition: transform 0.3s ease, margin-left 0.3s ease; width: 250px; border-top: 1px solid rgba(255,255,255,0.1); }",
        ".sidebar h6 { font-size: 0.75rem; letter-spacing: .08em; text-transform: uppercase; color: #9ca3af; margin-bottom: 1rem; }",
        ".sidebar .form-label { font-size: 0.8rem; color: #d1d5db; margin-bottom: 0.5rem; }",
        ".sidebar .form-select, .sidebar .form-control { background-color: #343a40; border-color: #495057; color: #e5e7eb; }",
        ".sidebar .form-select:focus, .sidebar .form-control:focus { background-color: #343a40; border-color: #6c757d; color: #e5e7eb; }",
        ".stat-card { background: white; border-radius: 8px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }",
        ".sidebar.hidden { transform: translateX(-250px); }",
        ".hamburger-btn { background: none; border: none; color: white; font-size: 1.5rem; padding: 0.5rem; cursor: pointer; margin-right: 0.5rem; }",
        ".hamburger-btn:hover { opacity: 0.8; }",
        ".sidebar-overlay { display: none; position: fixed; top: 56px; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); z-index: 999; }",
        ".sidebar-overlay.show { display: block; }",
        "@media (min-width: 768px) { .sidebar-overlay { display: none !important; } }",
        "@media (min-width: 768px) { .sidebar { position: fixed; transform: none; } .sidebar.hidden { transform: translateX(-250px); } }",
        "main { transition: margin-left 0.3s ease; margin-left: 250px; }",
        "main.full-width { margin-left: 0; }",
        "@media (max-width: 767px) { main { margin-left: 0; } }",
        ".sidebar .nav-link { color: #d1d5db; padding: 0.5rem 1rem; border-radius: 4px; margin-bottom: 0.25rem; font-size: 0.9rem; border: 1px solid transparent; transition: all 0.2s; cursor: pointer; }",
        ".sidebar .nav-link:hover { background-color: #343a40; color: white; }",
        ".sidebar .nav-link.active { background-color: #3b82f6; color: white; border-color: #3b82f6; }",
        ".sidebar .nav-link i { margin-right: 0.5rem; }",
        ".section-hidden { display: none; }",
        ".about-content { line-height: 1.7; color: #374151; }",
        ".about-content h2 { color: #111827; margin-top: 1.5rem; }",
        ".about-content p { margin-bottom: 1.25rem; }",
        ".sub-header { border-left: 4px solid #3b82f6; padding-left: 0.75rem; margin-bottom: 1.5rem; font-weight: 600; color: #1f2937; }",
        "th.sortable { cursor: pointer; position: relative; padding-right: 1.5rem !important; }",
        "th.sortable::after { content: '↕'; position: absolute; right: 0.5rem; opacity: 0.3; }",
        "th.sortable.asc::after { content: '↑'; opacity: 1; color: #3b82f6; }",
        "th.sortable.desc::after { content: '↓'; opacity: 1; color: #3b82f6; }",
        ".pagination-controls { display: flex; justify-content: space-between; align-items: center; margin-top: 1rem; flex-wrap: wrap; gap: 0.5rem; }",
        ".btn:disabled, .btn.disabled { opacity: 0.5; cursor: not-allowed; pointer-events: none; }",
        ".flatpickr-day.disabled, .flatpickr-day.not-allowed { opacity: 0.3; cursor: not-allowed; }",
        ".flatpickr-day.disabled:hover, .flatpickr-day.not-allowed:hover { background: transparent; }",
        ".chatbot-container { position: fixed; bottom: 20px; right: 20px; width: 380px; max-height: 600px; z-index: 1000; box-shadow: 0 4px 12px rgba(0,0,0,0.15); border-radius: 12px; overflow: hidden; background: white; display: none; }",
        ".chatbot-container.open { display: flex; flex-direction: column; }",
        ".chatbot-header { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; padding: 1rem; display: flex; justify-content: space-between; align-items: center; cursor: pointer; }",
        ".chatbot-header h6 { margin: 0; font-weight: 600; }",
        ".chatbot-body { flex: 1; overflow-y: auto; padding: 1rem; background: #f8f9fa; max-height: 450px; }",
        ".chatbot-message { margin-bottom: 1rem; padding: 0.75rem; border-radius: 8px; }",
        ".chatbot-message.user { background: #e3f2fd; margin-left: 2rem; }",
        ".chatbot-message.assistant { background: white; margin-right: 2rem; border-left: 3px solid #3b82f6; }",
        ".chatbot-input-container { padding: 1rem; background: white; border-top: 1px solid #e5e7eb; display: flex; gap: 0.5rem; }",
        ".chatbot-input { flex: 1; border: 1px solid #d1d5db; border-radius: 6px; padding: 0.5rem; font-size: 0.9rem; }",
        ".chatbot-toggle { position: fixed; bottom: 20px; right: 20px; width: 60px; height: 60px; border-radius: 50%; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: white; border: none; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4); cursor: pointer; z-index: 999; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; transition: transform 0.2s; }",
        ".chatbot-toggle:hover { transform: scale(1.1); }",
        ".chatbot-toggle.hidden { display: none; }",
        ".recommendation-badge { display: inline-block; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; margin-left: 0.5rem; }",
        ".recommendation-badge.best { background: #10b981; color: white; }",
        ".recommendation-badge.good { background: #3b82f6; color: white; }",
        ".recommendation-badge.avoid { background: #ef4444; color: white; }",
        "</style>",
        "</head>",
        "<body>",
        "<nav class='navbar navbar-dark bg-dark px-3 fixed-top'>",
        "<button class='hamburger-btn' id='sidebar-toggle' aria-label='Toggle menu'>☰</button>",
        "<span class='navbar-brand ms-2'>California Residential Energy Spending</span>",
        "</nav>",
        "<div class='sidebar-overlay' id='sidebar-overlay'></div>",
        "<div class='container-fluid'>",
        "<div class='row'>",
        "<aside class='col-md-3 col-lg-2 sidebar' id='sidebar'>",
        "<div class='mb-4'>",
        "<h6>Navigation</h6>",
        "<div class='nav flex-column'>",
        "<div class='nav-link active' id='nav-dashboard'>Dashboard</div>",
        "<div class='nav-link' id='nav-about'>About</div>",
        "</div>",
        "</div>",
        "<div id='sidebar-filters'>",
        "<h6>Time Filtering</h6>",
        "<div class='mb-3'>",
        "<label for='granularity' class='form-label'>View</label>",
        "<select id='granularity' class='form-select form-select-sm'>",
        "<option value='hourly'>Hourly</option>",
        "<option value='daily'>Daily</option>",
        "<option value='weekly'>Weekly</option>",
        "<option value='monthly' selected>Monthly</option>",
        "</select>",
        "</div>",
        "<div class='mb-3'>",
        "<label for='date-range' class='form-label'>Date Range</label>",
        "<input type='text' id='date-range' class='form-control form-control-sm' placeholder='Select range...'>",
        "</div>",
        "<div class='mb-3' id='period-navigation' style='display:none;'>",
        "<label for='period-select' class='form-label' id='period-label'>Navigate to</label>",
        "<select id='period-select' class='form-select form-select-sm'>",
        "<option value=''>Select period...</option>",
        "</select>",
        "</div>",
        "</div>",
        "</aside>",
        "<main class='col-md-9 col-lg-10 py-3'>",
        "<div id='section-dashboard'>",
        "    <div class='card mb-3'>",
        "        <div class='card-header fw-semibold text-primary d-flex justify-content-between align-items-center'>",
        "            <span>Visual Trends</span>",
        "            <div class='btn-group btn-group-sm' role='group'>",
        "                <button type='button' class='btn btn-outline-primary' id='chart-prev-btn' title='Previous period'>",
        "                    <svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='currentColor' class='bi bi-chevron-left' viewBox='0 0 16 16'><path fill-rule='evenodd' d='M11.354 1.646a.5.5 0 0 1 0 .708L5.707 8l5.647 5.646a.5.5 0 0 1-.708.708l-6-6a.5.5 0 0 1 0-.708l6-6a.5.5 0 0 1 .708 0z'/></svg>",
        "                </button>",
        "                <button type='button' class='btn btn-outline-primary' id='chart-next-btn' title='Next period'>",
        "                    <svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='currentColor' class='bi bi-chevron-right' viewBox='0 0 16 16'><path fill-rule='evenodd' d='M4.646 1.646a.5.5 0 0 1 .708 0l6 6a.5.5 0 0 1 0 .708l-6 6a.5.5 0 0 1-.708-.708L10.293 8 4.646 2.354a.5.5 0 0 1 0-.708z'/></svg>",
        "                </button>",
        "            </div>",
        "        </div>",
        "        <div class='card-body'>",
        "            <div id='energy-chart-container' class='mb-4'></div>",
        "        </div>",
        "    </div>",
        "    <div class='card mb-3'>",
        "        <div class='card-header fw-semibold text-primary'>Detailed Data Breakdown</div>",
        "        <div class='card-body'>",
        "            <div class='row g-3 mb-3'>",
        "                <div class='col-auto'>",
        "                    <label class='form-label small fw-bold text-muted mb-1'>Filter by Type</label>",
        "                    <select id='table-type-filter' class='form-select form-select-sm' style='width: 150px;'>",
        "                        <option value='all'>All Types</option>",
        "                        <option value='historical'>Historical</option>",
        "                        <option value='prediction'>Prediction</option>",
        "                    </select>",
        "                </div>",
        "                <div class='col-auto ms-auto d-flex align-items-end'>",
        "                    <label class='form-label small fw-bold text-muted me-2 mb-2'>Show</label>",
        "                    <select id='table-page-size' class='form-select form-select-sm' style='width: 70px;'>",
        "                        <option value='10'>10</option>",
        "                        <option value='25'>25</option>",
        "                        <option value='50'>50</option>",
        "                    </select>",
        "                </div>",
        "            </div>",
        "            <div class='table-responsive'>",
        "                <table id='energy-table' class='table table-sm table-hover table-striped mb-0'>",
        "                    <thead>",
        "                        <tr>",
        "                            <th>Type</th>",
        "                            <th class='sortable' data-sort='val' id='sort-cost'>Cost</th>",
        "                            <th class='sortable date-col' data-sort='year' id='col-year' style='display:none;'>Year</th>",
        "                            <th class='sortable date-col' data-sort='month' id='col-month' style='display:none;'>Month</th>",
        "                            <th class='sortable date-col' data-sort='day' id='col-day' style='display:none;'>Day</th>",
        "                            <th class='sortable date-col' data-sort='hour' id='col-hour' style='display:none;'>Hour</th>",
        "                        </tr>",
        "                    </thead>",
        "                    <tbody id='energy-table-body'></tbody>",
        "                </table>",
        "            </div>",
        "            <div class='pagination-controls'>",
        "                <div class='small text-muted' id='pagination-info'>Showing 1 to 10 of 0 entries</div>",
        "                <nav aria-label='Table navigation'>",
        "                    <ul class='pagination pagination-sm mb-0' id='pagination-list'></ul>",
        "                </nav>",
        "            </div>",
        "        </div>",
        "    </div>",
        "    <div class='card mb-3'>",
        "        <div class='card-header fw-semibold text-primary'>Energy Usage Recommendations</div>",
        "        <div class='card-body'>",
        "            <div id='recommendations-container'></div>",
        "        </div>",
        "    </div>",
        "</div>",
    ]

    # About Section
    html_parts.append(f"""
<div id='section-about' class='section-hidden'>
    <div class='card mb-3'>
        <div class='card-header fw-semibold text-primary'>About This Project</div>
        <div class='card-body about-content'>
            <h2 class="h4 mb-3">Motivation</h2>
            <p>Southern California experiences some of the highest electricity demand in the United States due to a combination of factors such as widespread air-conditioning use, a growing number of electric vehicles, and increasing residential and commercial energy consumption. During hot summer months, cooling loads drive peak demand in the late afternoon and evening, while electric vehicle charging and household activities further elevate nighttime consumption. These demand patterns often lead to periods of high electricity prices, even when consumers are unaware of the cost differences throughout the day.</p>
            
            <p>At the same time, many energy-intensive activities—such as EV charging, running laundry, or operating large appliances—can be shifted to hours when electricity is cheaper. Identifying these “optimal usage windows” has the potential to reduce household energy bills, ease stress on the electric grid, and support more efficient use of renewable generation.</p>
            
            <p>However, hourly electricity prices are not always publicly available for Southern California, and consumers rarely have access to clear or actionable guidance about when electricity is most affordable. This project addresses this gap by estimating hourly electricity costs using available demand, generation, and weather data, and by forecasting prices for the next day. With these predictions, the system provides users with intuitive recommendations about the best times to use electricity.</p>
            
            <p>By helping consumers shift demand to lower-cost hours, the project supports both economic savings and grid reliability, while also encouraging more sustainable energy behavior in a region where electricity demand continues to rise.</p>
        </div>
    </div>
</div>
""")

    # Extract valid dates for flatpickr enable list
    all_valid_dates = sorted(
        list(set([d["date"].split(' ')[0] for d in unified_data])))

    # Unified Dashboard Script (Historical + Predictions)
    html_parts.append(f"""
<script>
const allData = {json.dumps(unified_data)};
const validDates = {json.dumps(all_valid_dates)};
let fp; 

// Table State
let tableState = {{
    data: [], // Currently filtered and windowed data
    displayData: [], // After type filter and sort
    typeFilter: 'all',
    sortCol: 'date',
    sortDir: 'desc',
    pageSize: 10,
    currentPage: 1
}};

function updateView() {{
    const granularityIdx = document.getElementById('granularity');
    if (!granularityIdx) return;
    const granularity = granularityIdx.value;
    const selectedDateStr = document.getElementById('date-range').value || (fp ? fp.formatDate(fp.selectedDates[0], 'Y-m-d') : '');
    
    if (!selectedDateStr) return;
    
    const selectedDate = new Date(selectedDateStr + 'T00:00:00');
    let start, end;

    if (granularity === 'hourly') {{
        start = new Date(selectedDate);
        end = new Date(selectedDate);
        end.setHours(23, 59, 59);
    }} else if (granularity === 'daily') {{
        start = new Date(selectedDate);
        const day = start.getDay();
        const diff = start.getDate() - day + (day === 0 ? -6 : 1);
        start.setDate(diff);
        start.setHours(0,0,0,0);
        end = new Date(start);
        end.setDate(start.getDate() + 6);
        end.setHours(23, 59, 59);
    }} else if (granularity === 'weekly') {{
        start = new Date(selectedDate.getFullYear(), selectedDate.getMonth(), 1);
        end = new Date(selectedDate.getFullYear(), selectedDate.getMonth() + 1, 0);
        end.setHours(23, 59, 59);
    }} else if (granularity === 'monthly') {{
        start = new Date(selectedDate.getFullYear(), 0, 1);
        end = new Date(selectedDate.getFullYear(), 11, 31);
        end.setHours(23, 59, 59);
    }}

    // Chart still uses windowed data
    const chartData = allData.filter(d => {{
        if (d.for !== granularity) return false;
        if (granularity === 'monthly') {{
            // For monthly, compare by year only
            const dDate = new Date(d.date);
            const dYear = dDate.getFullYear();
            return dYear === selectedDate.getFullYear();
        }} else {{
            const dDate = new Date(d.date);
            return dDate >= start && dDate <= end;
        }}
    }});
    
    // Table shows all data for this granularity
    tableState.data = allData.filter(d => d.for === granularity);
    
    updateChart(chartData, granularity, start, end);
    applyTableState(selectedDateStr);
    updateNavigationButtons();
    
    // Trigger chatbot recommendations if available
    if (window.chatbotGenerateRecommendations) {{
        setTimeout(() => window.chatbotGenerateRecommendations(), 1000);
    }}
}}

function applyTableState(targetDateStr) {{
    const {{ typeFilter, sortCol, sortDir, pageSize }} = tableState;
    
    // 1. Filter
    tableState.displayData = tableState.data.filter(d => 
        typeFilter === 'all' || d.type === typeFilter
    );
    
    // 2. Sort
    tableState.displayData.sort((a, b) => {{
        let valA = a[sortCol];
        let valB = b[sortCol];
        if (sortCol === 'date') {{
            valA = new Date(valA);
            valB = new Date(valB);
        }} else if (sortCol === 'month') {{
            // Sort months by their numeric value
            const monthNames = ['January', 'February', 'March', 'April', 'May', 'June', 
                              'July', 'August', 'September', 'October', 'November', 'December'];
            valA = monthNames.indexOf(valA || '');
            valB = monthNames.indexOf(valB || '');
        }}
        if (valA < valB) return sortDir === 'asc' ? -1 : 1;
        if (valA > valB) return sortDir === 'asc' ? 1 : -1;
        return 0;
    }});
    
    // 3. Auto-navigate to target date if provided
    if (targetDateStr) {{
        const idx = tableState.displayData.findIndex(d => {{
            if (d.for === 'hourly') return d.date.startsWith(targetDateStr);
            if (d.for === 'monthly') return d.date.slice(0, 7) === targetDateStr.slice(0, 7);
            return d.date === targetDateStr;
        }});
        if (idx !== -1) {{
            tableState.currentPage = Math.floor(idx / pageSize) + 1;
        }}
    }}
    
    renderTable();
}}

function updateTableColumns(granularity) {{
    // Show/hide columns based on granularity
    const yearCol = document.getElementById('col-year');
    const monthCol = document.getElementById('col-month');
    const dayCol = document.getElementById('col-day');
    const hourCol = document.getElementById('col-hour');
    
    if (granularity === 'hourly') {{
        yearCol.style.display = '';
        monthCol.style.display = '';
        dayCol.style.display = '';
        hourCol.style.display = '';
    }} else if (granularity === 'daily') {{
        yearCol.style.display = '';
        monthCol.style.display = '';
        dayCol.style.display = '';
        hourCol.style.display = 'none';
    }} else if (granularity === 'weekly') {{
        yearCol.style.display = '';
        monthCol.style.display = '';
        dayCol.style.display = 'none';
        hourCol.style.display = 'none';
    }} else if (granularity === 'monthly') {{
        yearCol.style.display = '';
        monthCol.style.display = '';
        dayCol.style.display = 'none';
        hourCol.style.display = 'none';
    }}
}}

function renderTable() {{
    const tbody = document.getElementById('energy-table-body');
    const {{ pageSize, currentPage, displayData }} = tableState;
    
    // Get current granularity from the select
    const granularity = document.getElementById('granularity')?.value || 'monthly';
    updateTableColumns(granularity);
    
    const totalEntries = displayData.length;
    const totalPages = Math.ceil(totalEntries / pageSize) || 1;
    const normalizedPage = Math.min(currentPage, totalPages);
    tableState.currentPage = normalizedPage;
    
    const startIdx = (normalizedPage - 1) * pageSize;
    const endIdx = Math.min(startIdx + pageSize, totalEntries);
    const splitData = displayData.slice(startIdx, endIdx);
    
    tbody.innerHTML = splitData.map(d => {{
        const year = d.year || '';
        const month = d.month || '';
        const day = d.day !== undefined ? d.day : '';
        const hour = d.hour !== undefined ? d.hour : '';
        
        let cells = `
            <td><span class="badge ${{d.type === 'historical' ? 'bg-secondary' : 'bg-primary'}}">${{d.type.charAt(0).toUpperCase() + d.type.slice(1)}}</span></td>
            <td><strong>$${{d.val.toFixed(d.for === 'hourly' ? 4 : 2)}}</strong></td>
        `;
        
        if (granularity === 'hourly') {{
            cells += `
                <td class="text-muted small">${{year}}</td>
                <td class="text-muted small">${{month}}</td>
                <td class="text-muted small">${{day}}</td>
                <td class="text-muted small">${{hour}}</td>
            `;
        }} else if (granularity === 'daily') {{
            cells += `
                <td class="text-muted small">${{year}}</td>
                <td class="text-muted small">${{month}}</td>
                <td class="text-muted small">${{day}}</td>
            `;
        }} else if (granularity === 'weekly' || granularity === 'monthly') {{
            cells += `
                <td class="text-muted small">${{year}}</td>
                <td class="text-muted small">${{month}}</td>
            `;
        }}
        
        return `<tr>${{cells}}</tr>`;
    }}).join('');
    
    if (displayData.length === 0) {{
        const colCount = granularity === 'hourly' ? 6 : granularity === 'daily' ? 5 : 4;
        tbody.innerHTML = `<tr><td colspan="${{colCount}}" class="text-center text-muted py-4">No data available for this selection</td></tr>`;
    }}
    
    // Update Pagination UI
    document.getElementById('pagination-info').innerText = totalEntries > 0 
        ? `Showing ${{startIdx + 1}} to ${{endIdx}} of ${{totalEntries}} entries`
        : `Showing 0 to 0 of 0 entries`;
        
    const pagList = document.getElementById('pagination-list');
    pagList.innerHTML = '';
    
    // Prev
    const prevLi = document.createElement('li');
    prevLi.className = `page-item ${{normalizedPage === 1 ? 'disabled' : ''}}`;
    prevLi.innerHTML = `<a class="page-link" href="#" onclick="changePage(${{normalizedPage - 1}})">Previous</a>`;
    pagList.appendChild(prevLi);
    
    // Pages (Show 5 around current)
    for (let i = 1; i <= totalPages; i++) {{
        if (totalPages > 5) {{
            if (i > normalizedPage + 2 || i < normalizedPage - 2) continue;
        }}
        const li = document.createElement('li');
        li.className = `page-item ${{i === normalizedPage ? 'active' : ''}}`;
        li.innerHTML = `<a class="page-link" href="#" onclick="changePage(${{i}})">${{i}}</a>`;
        pagList.appendChild(li);
    }}
    
    // Next
    const nextLi = document.createElement('li');
    nextLi.className = `page-item ${{normalizedPage === totalPages || totalEntries === 0 ? 'disabled' : ''}}`;
    nextLi.innerHTML = `<a class="page-link" href="#" onclick="changePage(${{normalizedPage + 1}})">Next</a>`;
    pagList.appendChild(nextLi);
}}

function changePage(p) {{
    if (p < 1) return;
    tableState.currentPage = p;
    renderTable();
}}

function handleSort(col) {{
    if (tableState.sortCol === col) {{
        tableState.sortDir = tableState.sortDir === 'asc' ? 'desc' : 'asc';
    }} else {{
        tableState.sortCol = col;
        tableState.sortDir = 'asc';
    }}
    
    // Update Header Icons
    document.querySelectorAll('th.sortable').forEach(th => th.classList.remove('asc', 'desc'));
    let activeTh;
    if (col === 'val') {{
        activeTh = document.getElementById('sort-cost');
    }} else if (col === 'year') {{
        activeTh = document.getElementById('col-year');
    }} else if (col === 'month') {{
        activeTh = document.getElementById('col-month');
    }} else if (col === 'day') {{
        activeTh = document.getElementById('col-day');
    }} else if (col === 'hour') {{
        activeTh = document.getElementById('col-hour');
    }}
    if (activeTh) activeTh.classList.add(tableState.sortDir);
    
    applyTableState();
}}

function updateChart(data, granularity, start, end) {{
    const container = document.getElementById('energy-chart-container');
    if (!container) return;
    
    const hist = data.filter(d => d.type === 'historical');
    const pred = data.filter(d => d.type === 'prediction');
    const traces = [];
    
    if (hist.length > 0) {{
        traces.push({{
            x: hist.map(d => d.date_display || d.date),
            y: hist.map(d => d.val),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Historical Cost',
            line: {{ color: '#64748b', width: 2 }},
            marker: {{ size: 6 }},
            hovertemplate: '<b>%{{x}}</b><br>Cost: $%{{y:.2f}}<extra></extra>'
        }});
    }}
    
    if (pred.length > 0) {{
        traces.push({{
            x: pred.map(d => d.date_display || d.date),
            y: pred.map(d => d.val),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Predicted Cost',
            line: {{ color: '#3b82f6', width: 3, dash: 'dash' }},
            marker: {{ size: 8 }},
            hovertemplate: '<b>%{{x}}</b><br>Cost: $%{{y:.2f}}<extra></extra>'
        }});
    }}
    
    const titleDate = start.toLocaleDateString(undefined, {{ 
        year: 'numeric', 
        month: granularity === 'monthly' ? undefined : 'short', 
        day: (granularity === 'hourly' || granularity === 'daily') ? 'numeric' : undefined 
    }});
                
                const layout = {{
        title: `${{granularity.charAt(0).toUpperCase() + granularity.slice(1)}} Spending (${{titleDate}}${{granularity !== 'hourly' ? ' context' : ''}})`,
        xaxis: {{ title: 'Time' }},
        yaxis: {{ title: 'Cost (USD)' }},
        margin: {{ t: 40, b: 40, l: 60, r: 20 }},
        height: 450,
        legend: {{ orientation: 'h', y: -0.2 }}
    }};
    
    Plotly.newPlot(container, traces, layout, {{responsive: true}});
}}

function updateNavigationButtons() {{
    const granularity = document.getElementById('granularity')?.value || 'monthly';
    const currentDateStr = document.getElementById('date-range').value || (fp ? fp.formatDate(fp.selectedDates[0], 'Y-m-d') : '');
    const prevBtn = document.getElementById('chart-prev-btn');
    const nextBtn = document.getElementById('chart-next-btn');
    
    if (!currentDateStr || !prevBtn || !nextBtn) {{
        if (prevBtn) {{
            prevBtn.disabled = true;
            prevBtn.classList.add('disabled');
        }}
        if (nextBtn) {{
            nextBtn.disabled = true;
            nextBtn.classList.add('disabled');
        }}
        return;
    }}
    
    const currentDate = new Date(currentDateStr + 'T00:00:00');
    const sortedDates = [...validDates].sort();
    
    // Check if there's data for previous period
    let hasPrevData = false;
    if (granularity === 'hourly') {{
        const prevDate = new Date(currentDate);
        prevDate.setDate(currentDate.getDate() - 1);
        const prevDateStr = prevDate.toISOString().split('T')[0];
        hasPrevData = sortedDates.some(d => d <= prevDateStr && d < currentDateStr);
    }} else if (granularity === 'daily') {{
        const prevDate = new Date(currentDate);
        prevDate.setDate(currentDate.getDate() - 7);
        const prevDateStr = prevDate.toISOString().split('T')[0];
        hasPrevData = sortedDates.some(d => d <= prevDateStr && d < currentDateStr);
    }} else if (granularity === 'weekly') {{
        const prevDate = new Date(currentDate);
        prevDate.setMonth(currentDate.getMonth() - 1);
        const prevDateStr = prevDate.toISOString().split('T')[0];
        hasPrevData = sortedDates.some(d => d <= prevDateStr && d < currentDateStr);
    }} else if (granularity === 'monthly') {{
        const prevDate = new Date(currentDate);
        prevDate.setFullYear(currentDate.getFullYear() - 1);
        const prevYear = prevDate.getFullYear();
        hasPrevData = sortedDates.some(d => {{
            const dYear = new Date(d + 'T00:00:00').getFullYear();
            return dYear < currentDate.getFullYear();
        }});
    }}
    
    // Check if there's data for next period
    let hasNextData = false;
    if (granularity === 'hourly') {{
        const nextDate = new Date(currentDate);
        nextDate.setDate(currentDate.getDate() + 1);
        const nextDateStr = nextDate.toISOString().split('T')[0];
        hasNextData = sortedDates.some(d => d >= nextDateStr && d > currentDateStr);
    }} else if (granularity === 'daily') {{
        const nextDate = new Date(currentDate);
        nextDate.setDate(currentDate.getDate() + 7);
        const nextDateStr = nextDate.toISOString().split('T')[0];
        hasNextData = sortedDates.some(d => d >= nextDateStr && d > currentDateStr);
    }} else if (granularity === 'weekly') {{
        const nextDate = new Date(currentDate);
        nextDate.setMonth(currentDate.getMonth() + 1);
        const nextDateStr = nextDate.toISOString().split('T')[0];
        hasNextData = sortedDates.some(d => d >= nextDateStr && d > currentDateStr);
    }} else if (granularity === 'monthly') {{
        const nextYear = currentDate.getFullYear() + 1;
        hasNextData = sortedDates.some(d => {{
            const dYear = new Date(d + 'T00:00:00').getFullYear();
            return dYear > currentDate.getFullYear();
        }});
    }}
    
    // Update button states
    prevBtn.disabled = !hasPrevData;
    nextBtn.disabled = !hasNextData;
    
    // Add/remove disabled class for styling
    if (hasPrevData) {{
        prevBtn.classList.remove('disabled');
    }} else {{
        prevBtn.classList.add('disabled');
    }}
    
    if (hasNextData) {{
        nextBtn.classList.remove('disabled');
    }} else {{
        nextBtn.classList.add('disabled');
    }}
}}

function navigateChart(direction) {{
    const granularity = document.getElementById('granularity')?.value || 'monthly';
    const currentDateStr = document.getElementById('date-range').value || (fp ? fp.formatDate(fp.selectedDates[0], 'Y-m-d') : '');
    if (!currentDateStr) return;
    
    // Prevent navigation if button is disabled
    const btn = direction === 'prev' ? document.getElementById('chart-prev-btn') : document.getElementById('chart-next-btn');
    if (btn && btn.disabled) return;
    
    const currentDate = new Date(currentDateStr + 'T00:00:00');
    let newDate = new Date(currentDate);
    
    // Navigate based on granularity
    if (granularity === 'hourly') {{
        newDate.setDate(currentDate.getDate() + (direction === 'next' ? 1 : -1));
    }} else if (granularity === 'daily') {{
        // Navigate by week
        newDate.setDate(currentDate.getDate() + (direction === 'next' ? 7 : -7));
    }} else if (granularity === 'weekly') {{
        // Navigate by month
        newDate.setMonth(currentDate.getMonth() + (direction === 'next' ? 1 : -1));
    }} else if (granularity === 'monthly') {{
        // Navigate by year
        newDate.setFullYear(currentDate.getFullYear() + (direction === 'next' ? 1 : -1));
    }}
    
    // Format new date and check if it's valid
    const newDateStr = newDate.toISOString().split('T')[0];
    
    // Find the closest valid date
    let targetDate = newDateStr;
    if (!validDates.includes(newDateStr)) {{
        // Find closest valid date
        const sortedDates = [...validDates].sort();
        if (direction === 'next') {{
            targetDate = sortedDates.find(d => d > newDateStr) || sortedDates[sortedDates.length - 1];
        }} else {{
            const reversedDates = [...sortedDates].reverse();
            targetDate = reversedDates.find(d => d < newDateStr) || sortedDates[0];
        }}
    }}
    
    // Update date picker and view
    if (fp) {{
        fp.setDate(targetDate, false); // false = don't trigger onChange
        updateView(); // Manually trigger updateView
    }} else {{
        document.getElementById('date-range').value = targetDate;
        updateView();
    }}
}}

// Event Listeners
document.getElementById('table-type-filter').addEventListener('change', (e) => {{
    tableState.typeFilter = e.target.value;
    applyTableState();
}});

document.getElementById('table-page-size').addEventListener('change', (e) => {{
    tableState.pageSize = parseInt(e.target.value);
    applyTableState();
}});

document.querySelectorAll('th.sortable').forEach(th => {{
    th.addEventListener('click', () => handleSort(th.dataset.sort));
}});

function updatePeriodNavigation() {{
    const granularity = document.getElementById('granularity')?.value || 'monthly';
    const periodNav = document.getElementById('period-navigation');
    const periodSelect = document.getElementById('period-select');
    const periodLabel = document.getElementById('period-label');
    
    if (!periodNav || !periodSelect) return;
    
    // Filter data by granularity
    const filteredData = allData.filter(d => d.for === granularity);
    if (filteredData.length === 0) {{
        periodNav.style.display = 'none';
        return;
    }}
    
    periodNav.style.display = 'block';
    periodSelect.innerHTML = '<option value="">Select period...</option>';
    
    if (granularity === 'hourly') {{
        // Show all available days
        periodLabel.textContent = 'Navigate to Day';
        const days = [...new Set(filteredData.map(d => d.date.split(' ')[0]))].sort().reverse();
        days.forEach(day => {{
            const date = new Date(day);
            const option = document.createElement('option');
            option.value = day;
            option.textContent = date.toLocaleDateString('en-US', {{ weekday: 'short', year: 'numeric', month: 'short', day: 'numeric' }});
            periodSelect.appendChild(option);
        }});
    }} else if (granularity === 'daily') {{
        // Show all available weeks
        periodLabel.textContent = 'Navigate to Week';
        const weeks = new Map();
        filteredData.forEach(d => {{
            const date = new Date(d.date);
            const weekStart = new Date(date);
            const day = weekStart.getDay();
            const diff = weekStart.getDate() - day + (day === 0 ? -6 : 1);
            weekStart.setDate(diff);
            weekStart.setHours(0, 0, 0, 0);
            const weekEnd = new Date(weekStart);
            weekEnd.setDate(weekStart.getDate() + 6);
            
            const weekKey = weekStart.toISOString().split('T')[0];
            if (!weeks.has(weekKey)) {{
                const weekStr = weekStart.toLocaleDateString('en-US', {{ month: 'short', day: 'numeric' }}) + 
                              ' - ' + weekEnd.toLocaleDateString('en-US', {{ month: 'short', day: 'numeric', year: 'numeric' }});
                weeks.set(weekKey, weekStr);
            }}
        }});
        const sortedWeeks = Array.from(weeks.entries()).sort((a, b) => b[0].localeCompare(a[0]));
        sortedWeeks.forEach(([date, label]) => {{
            const option = document.createElement('option');
            option.value = date;
            option.textContent = label;
            periodSelect.appendChild(option);
        }});
    }} else if (granularity === 'weekly') {{
        // Show all available months
        periodLabel.textContent = 'Navigate to Month';
        const months = new Map();
        filteredData.forEach(d => {{
            const date = new Date(d.date);
            const monthKey = `${{date.getFullYear()}}-${{String(date.getMonth() + 1).padStart(2, '0')}}`;
            if (!months.has(monthKey)) {{
                const monthStr = date.toLocaleDateString('en-US', {{ year: 'numeric', month: 'long' }});
                months.set(monthKey, monthStr);
            }}
        }});
        const sortedMonths = Array.from(months.entries()).sort((a, b) => b[0].localeCompare(a[0]));
        sortedMonths.forEach(([date, label]) => {{
            const option = document.createElement('option');
            option.value = date + '-01';
            option.textContent = label;
            periodSelect.appendChild(option);
        }});
    }} else if (granularity === 'monthly') {{
        // Show all available years
        periodLabel.textContent = 'Navigate to Year';
        const years = [...new Set(filteredData.map(d => {{
            const date = new Date(d.date);
            return date.getFullYear();
        }}))].sort((a, b) => b - a);
        years.forEach(year => {{
            const option = document.createElement('option');
            option.value = `${{year}}-01-01`;
            option.textContent = year;
            periodSelect.appendChild(option);
        }});
    }}
}}

function handlePeriodNavigation() {{
    const periodSelect = document.getElementById('period-select');
    if (!periodSelect || !periodSelect.value) return;
    
    const targetDate = periodSelect.value;
    if (fp) {{
        fp.setDate(targetDate, false);
        updateView();
    }} else {{
        document.getElementById('date-range').value = targetDate;
        updateView();
    }}
    periodSelect.value = ''; // Reset selection
}}

// Chart navigation buttons
document.getElementById('chart-prev-btn').addEventListener('click', () => navigateChart('prev'));
document.getElementById('chart-next-btn').addEventListener('click', () => navigateChart('next'));

// Period navigation dropdown
document.getElementById('period-select').addEventListener('change', handlePeriodNavigation);

// Update period navigation when granularity changes
document.getElementById('granularity').addEventListener('change', () => {{
    updatePeriodNavigation();
    updateView();
}});

// Initialize flatpickr
fp = flatpickr('#date-range', {{ 
    mode: 'single', 
    dateFormat: 'Y-m-d', 
    enable: validDates,
    defaultDate: validDates[validDates.length - 1],
    onChange: updateView 
}});

// Initial call
updatePeriodNavigation();
updateView();

// Navigation Logic
(function() {{
    const navDashboard = document.getElementById('nav-dashboard');
    const navAbout = document.getElementById('nav-about');
    const sectionDashboard = document.getElementById('section-dashboard');
    const sectionAbout = document.getElementById('section-about');
    const sidebarFilters = document.getElementById('sidebar-filters');

    function switchSection(section) {{
        if (section === 'dashboard') {{
            sectionDashboard.classList.remove('section-hidden');
            sectionAbout.classList.add('section-hidden');
            navDashboard.classList.add('active');
            navAbout.classList.remove('active');
            sidebarFilters.style.display = 'block';
            updateView();
        }} else {{
            sectionDashboard.classList.add('section-hidden');
            sectionAbout.classList.remove('section-hidden');
            navDashboard.classList.remove('active');
            navAbout.classList.add('active');
            sidebarFilters.style.display = 'none';
        }}
    }}

    if (navDashboard) navDashboard.addEventListener('click', () => switchSection('dashboard'));
    if (navAbout) navAbout.addEventListener('click', () => switchSection('about'));
    
    updateView();
}})();
</script>
""")

    # Sidebar Toggle Logic
    html_parts.append("""
<script>
  (function() {
  const sidebarToggle = document.getElementById('sidebar-toggle');
  const sidebar = document.getElementById('sidebar');
  const sidebarOverlay = document.getElementById('sidebar-overlay');
      const mainContent = document.querySelector('main');
  
  function toggleSidebar() {
        const isHidden = sidebar.classList.toggle('hidden');
        if (window.innerWidth >= 768) {
          if (isHidden) {
            mainContent.classList.add('full-width');
          } else {
            mainContent.classList.remove('full-width');
          }
        } else {
      sidebarOverlay.classList.toggle('show');
    }
  }
  
      if (sidebarToggle) sidebarToggle.addEventListener('click', toggleSidebar);
      if (sidebarOverlay) sidebarOverlay.addEventListener('click', toggleSidebar);
      
  function updateSidebarState() {
    if (window.innerWidth >= 768) {
      sidebar.classList.remove('hidden');
          mainContent.classList.remove('full-width');
      if (sidebarOverlay) sidebarOverlay.classList.remove('show');
    } else {
      sidebar.classList.add('hidden');
          mainContent.classList.add('full-width');
      if (sidebarOverlay) sidebarOverlay.classList.remove('show');
    }
  }
  
  updateSidebarState();
  window.addEventListener('resize', updateSidebarState);
  })();
</script>
""")

    # Chatbot component
    html_parts.append("""
    <!-- Chatbot Toggle Button -->
    <button class="chatbot-toggle" id="chatbot-toggle" title="Get energy usage recommendations">
        💬
    </button>
    
    <!-- Chatbot Container -->
    <div class="chatbot-container" id="chatbot-container">
        <div class="chatbot-header" id="chatbot-header">
            <h6>Energy Usage Assistant</h6>
            <span id="chatbot-close" style="cursor: pointer; font-size: 1.2rem;">×</span>
        </div>
        <div class="chatbot-body" id="chatbot-messages">
            <div class="chatbot-message assistant">
                <strong>Assistant:</strong> Hi! I'm your energy usage assistant. I analyze predictions and historical data to help you save money. Ask me about the best times to use electricity, or I can automatically analyze today's data for you.
            </div>
        </div>
        <div class="chatbot-input-container">
            <input type="text" class="chatbot-input" id="chatbot-input" placeholder="Ask about energy usage..." />
            <button class="btn btn-primary btn-sm" id="chatbot-send">Send</button>
        </div>
    </div>
    </main></div></div>
    
    <script>
    // Chatbot functionality
    (function() {
        const chatbotToggle = document.getElementById('chatbot-toggle');
        const chatbotContainer = document.getElementById('chatbot-container');
        const chatbotClose = document.getElementById('chatbot-close');
        const chatbotInput = document.getElementById('chatbot-input');
        const chatbotSend = document.getElementById('chatbot-send');
        const chatbotMessages = document.getElementById('chatbot-messages');
        
        function toggleChatbot() {
            chatbotContainer.classList.toggle('open');
            chatbotToggle.classList.toggle('hidden');
            if (chatbotContainer.classList.contains('open')) {
                chatbotInput.focus();
            }
        }
        
        chatbotToggle.addEventListener('click', toggleChatbot);
        chatbotClose.addEventListener('click', toggleChatbot);
        
        function addMessage(text, isUser = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `chatbot-message ${isUser ? 'user' : 'assistant'}`;
            messageDiv.innerHTML = `<strong>${isUser ? 'You' : 'Assistant'}:</strong> ${text}`;
            chatbotMessages.appendChild(messageDiv);
            chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
        }
        
        function generateRecommendations(data) {
            if (!data || data.length === 0) {
                return "I don't have enough data to provide recommendations. Please select a granularity with available data.";
            }
            
            const predictions = data.filter(d => d.type === 'prediction');
            const historical = data.filter(d => d.type === 'historical');
            
            if (predictions.length === 0 && historical.length === 0) {
                return "No data available for analysis.";
            }
            
            let recommendations = [];
            
            // Analyze hourly data for best times
            const hourlyData = data.filter(d => d.for === 'hourly');
            if (hourlyData.length > 0) {
                const hourlyPreds = hourlyData.filter(d => d.type === 'prediction');
                if (hourlyPreds.length > 0) {
                    // Find cheapest and most expensive hours
                    const sortedByCost = [...hourlyPreds].sort((a, b) => a.val - b.val);
                    const cheapestHours = sortedByCost.slice(0, 5);
                    const expensiveHours = sortedByCost.slice(-5).reverse();
                    
                    if (cheapestHours.length > 0) {
                        const avgCheap = cheapestHours.reduce((sum, d) => sum + d.val, 0) / cheapestHours.length;
                        const avgExpensive = expensiveHours.reduce((sum, d) => sum + d.val, 0) / expensiveHours.length;
                        const savings = ((avgExpensive - avgCheap) / avgExpensive * 100).toFixed(1);
                        
                        recommendations.push(`<strong>Best Hours to Use Electricity:</strong><br>`);
                        recommendations.push(`The cheapest hours are: ${cheapestHours.map(d => {
                            const date = new Date(d.date);
                            return date.toLocaleTimeString('en-US', { hour: 'numeric', hour12: true });
                        }).join(', ')}<br>`);
                        recommendations.push(`Average cost: $${avgCheap.toFixed(4)}/hour<br>`);
                        recommendations.push(`<strong>Avoid these expensive hours:</strong> ${expensiveHours.slice(0, 3).map(d => {
                            const date = new Date(d.date);
                            return date.toLocaleTimeString('en-US', { hour: 'numeric', hour12: true });
                        }).join(', ')} ($${avgExpensive.toFixed(4)}/hour)<br>`);
                        recommendations.push(`<span class="recommendation-badge best">Potential Savings: ${savings}%</span><br><br>`);
                    }
                }
            }
            
            // Analyze daily patterns
            const dailyData = data.filter(d => d.for === 'daily');
            if (dailyData.length > 0) {
                const dailyPreds = dailyData.filter(d => d.type === 'prediction');
                if (dailyPreds.length > 0) {
                    const sortedDaily = [...dailyPreds].sort((a, b) => a.val - b.val);
                    const cheapestDay = sortedDaily[0];
                    const expensiveDay = sortedDaily[sortedDaily.length - 1];
                    
                    if (cheapestDay && expensiveDay) {
                        const date1 = new Date(cheapestDay.date);
                        const date2 = new Date(expensiveDay.date);
                        recommendations.push(`<strong>Daily Recommendations:</strong><br>`);
                        recommendations.push(`Best day: ${date1.toLocaleDateString('en-US', { weekday: 'long', month: 'short', day: 'numeric' })} - $${cheapestDay.val.toFixed(2)}<br>`);
                        recommendations.push(`Most expensive day: ${date2.toLocaleDateString('en-US', { weekday: 'long', month: 'short', day: 'numeric' })} - $${expensiveDay.val.toFixed(2)}<br><br>`);
                    }
                }
            }
            
            // Analyze weekly patterns
            const weeklyData = data.filter(d => d.for === 'weekly');
            if (weeklyData.length > 0) {
                const weeklyPreds = weeklyData.filter(d => d.type === 'prediction');
                if (weeklyPreds.length > 0) {
                    const avgWeekly = weeklyPreds.reduce((sum, d) => sum + d.val, 0) / weeklyPreds.length;
                    recommendations.push(`<strong>Weekly Outlook:</strong><br>`);
                    recommendations.push(`Average weekly cost: $${avgWeekly.toFixed(2)}<br>`);
                    recommendations.push(`Plan major energy-intensive tasks (laundry, EV charging) during cheaper weeks.<br><br>`);
                }
            }
            
            // Monthly insights
            const monthlyData = data.filter(d => d.for === 'monthly');
            if (monthlyData.length > 0) {
                const monthlyPreds = monthlyData.filter(d => d.type === 'prediction');
                if (monthlyPreds.length > 0) {
                    const sortedMonthly = [...monthlyPreds].sort((a, b) => a.val - b.val);
                    recommendations.push(`<strong>Monthly Insights:</strong><br>`);
                    recommendations.push(`Upcoming months show ${sortedMonthly.length > 1 ? 'varying' : 'consistent'} costs.<br>`);
                    if (sortedMonthly.length > 1) {
                        const cheapestMonth = sortedMonthly[0];
                        const date = new Date(cheapestMonth.date);
                        recommendations.push(`Best month: ${date.toLocaleDateString('en-US', { month: 'long', year: 'numeric' })} - $${cheapestMonth.val.toFixed(2)}<br>`);
                    }
                }
            }
            
            // General tips
            recommendations.push(`<strong>💡 Tips:</strong><br>`);
            recommendations.push(`• Schedule EV charging during off-peak hours (typically late night/early morning)<br>`);
            recommendations.push(`• Run dishwashers and washing machines during cheaper hours<br>`);
            recommendations.push(`• Pre-cool your home before peak hours in summer<br>`);
            recommendations.push(`• Use timers for major appliances to take advantage of lower rates<br>`);
            
            return recommendations.join('');
        }
        
        function handleChatbotQuery(query) {
            const lowerQuery = query.toLowerCase();
            
            // Get current data - use allData if available, otherwise try dataCache
            const granularity = document.getElementById('granularity')?.value || 'monthly';
            let currentData = [];
            if (typeof allData !== 'undefined' && allData.length > 0) {
                currentData = allData.filter(d => d.for === granularity);
            } else if (typeof dataCache !== 'undefined' && dataCache[granularity]) {
                currentData = dataCache[granularity];
            }
            
            if (lowerQuery.includes('best time') || lowerQuery.includes('cheapest') || lowerQuery.includes('when should')) {
                if (currentData.length === 0) {
                    addMessage("Please select a granularity first to load data, then ask again.");
                    return;
                }
                const recommendations = generateRecommendations(currentData);
                addMessage(recommendations);
            } else if (lowerQuery.includes('analyze') || lowerQuery.includes('recommend') || lowerQuery.includes('suggest')) {
                if (currentData.length === 0) {
                    addMessage("Please select a granularity first to load data, then ask again.");
                    return;
                }
                const recommendations = generateRecommendations(currentData);
                addMessage(recommendations);
            } else if (lowerQuery.includes('hello') || lowerQuery.includes('hi') || lowerQuery === '') {
                addMessage("Hello! I can help you find the best times to use electricity based on predictions and historical data. Try asking: 'What are the best times to use electricity?' or 'Analyze today's data'");
            } else {
                addMessage("I can help you with energy usage recommendations. Try asking: 'What are the best times to use electricity?' or 'Analyze the current data for recommendations'");
            }
        }
        
        chatbotSend.addEventListener('click', () => {
            const query = chatbotInput.value.trim();
            if (query) {
                addMessage(query, true);
                chatbotInput.value = '';
                setTimeout(() => handleChatbotQuery(query), 500);
            }
        });
        
        chatbotInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                chatbotSend.click();
            }
        });
        
        // Function to generate recommendations when data is available
        window.chatbotGenerateRecommendations = function() {
            if (chatbotContainer.classList.contains('open')) {
                const granularity = document.getElementById('granularity')?.value || 'monthly';
                let currentData = [];
                if (typeof allData !== 'undefined' && allData.length > 0) {
                    currentData = allData.filter(d => d.for === granularity);
                } else if (typeof dataCache !== 'undefined' && dataCache[granularity]) {
                    currentData = dataCache[granularity];
                }
                if (currentData.length > 0) {
                    const recommendations = generateRecommendations(currentData);
                    addMessage(`<strong>Automatic Analysis:</strong><br>${recommendations}`);
                }
            }
        };
    })();
    </script>
    </body></html>""")
    out_path = SITE_DIR / "index.html"
    out_path.write_text("\n".join(html_parts), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    path = build()
    print(f"Built dashboard at {path}")
