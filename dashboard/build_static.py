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


def _normalize_to_per_household(df: pd.DataFrame, cost_col: str = "Estimated_Hourly_Cost_USD") -> pd.DataFrame:
    """Convert system-wide costs to per-household costs."""
    if cost_col in df.columns:
        df = df.copy()
        df[cost_col] = df[cost_col] / CA_HOUSEHOLDS
    return df


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
    hourly["dayofweek"] = hourly["timestamp"].dt.dayofweek
    hourly["month"] = hourly["timestamp"].dt.month
    hourly["year"] = hourly["timestamp"].dt.year
    return hourly


def build():
    SITE_DIR.mkdir(parents=True, exist_ok=True)

    preds = _read_csv(RESULTS_DIR / "predictions.csv")
    daily = _read_csv(RESULTS_DIR / "daily_history.csv")
    monthly = _read_csv(RESULTS_DIR / "monthly_history.csv")
    hourly = _load_hourly_data()

    # Normalize all costs to per-household
    hourly = _normalize_to_per_household(hourly)
    daily = _normalize_to_per_household(daily)
    monthly = _normalize_to_per_household(monthly)
    if not preds.empty and "prediction" in preds.columns:
        preds = preds.copy()
        preds["prediction"] = preds["prediction"] / CA_HOUSEHOLDS

    # Calculate weekly from daily - ensure proper week boundaries and seasonal variation
    weekly = pd.DataFrame()
    if not daily.empty and "Estimated_Hourly_Cost_USD" in daily:
        daily["date"] = pd.to_datetime(daily["date"])
        # Get the Monday of each week (week start)
        daily["week_start"] = daily["date"] - \
            pd.to_timedelta(daily["date"].dt.dayofweek, unit="d")
        # Group by week_start to get weekly totals (not cumulative)
        weekly = (
            daily.groupby("week_start")
            .agg({"Estimated_Hourly_Cost_USD": "sum"})
            .reset_index()
            .sort_values("week_start")
        )

    # Calculate insights
    insights = {}
    if not hourly.empty:
        # Peak hour of day
        hourly_avg = hourly.groupby("hour")["Estimated_Hourly_Cost_USD"].mean()
        insights["peak_hour"] = hourly_avg.idxmax()
        insights["peak_hour_cost"] = hourly_avg.max()
        insights["off_peak_hour"] = hourly_avg.idxmin()
        insights["off_peak_cost"] = hourly_avg.min()

        # Peak day of week
        daily_avg = hourly.groupby("dayofweek")[
            "Estimated_Hourly_Cost_USD"].sum()
        day_names = ["Monday", "Tuesday", "Wednesday",
                     "Thursday", "Friday", "Saturday", "Sunday"]
        insights["peak_day"] = day_names[daily_avg.idxmax()]
        insights["peak_day_cost"] = daily_avg.max()

    if not monthly.empty:
        monthly["year_month_start"] = pd.to_datetime(
            monthly["year_month_start"])
        monthly_avg = monthly.groupby(monthly["year_month_start"].dt.month)[
            "Estimated_Hourly_Cost_USD"].mean()
        month_names = ["Jan", "Feb", "Mar", "Apr", "May",
                       "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        insights["peak_month"] = month_names[monthly_avg.idxmax() - 1]
        insights["peak_month_cost"] = monthly_avg.max()
        insights["avg_monthly"] = monthly["Estimated_Hourly_Cost_USD"].mean()
        insights["total_yearly"] = monthly["Estimated_Hourly_Cost_USD"].sum() if len(
            monthly) >= 12 else None

    html_parts = [
        "<html>",
        "<head>",
        "<title>Residential Energy Spending per Household Dashboard</title>",
        "<link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css'>",
        "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
        "<style>",
        "body { font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f8f9fa; }",
        ".sidebar { background-color: #111827; color: #e5e7eb; min-height: 100vh; padding: 1rem; }",
        ".sidebar h6 { font-size: 0.75rem; letter-spacing: .08em; text-transform: uppercase; color: #9ca3af; margin-bottom: 1rem; }",
        ".sidebar .form-label { font-size: 0.8rem; color: #d1d5db; margin-bottom: 0.5rem; }",
        ".sidebar .form-select { background-color: #1f2937; border-color: #374151; color: #e5e7eb; }",
        ".sidebar .form-select:focus { background-color: #1f2937; border-color: #4b5563; color: #e5e7eb; }",
        ".insight-card { border-left: 4px solid #3b82f6; }",
        ".insight-value { font-size: 1.5rem; font-weight: 600; color: #3b82f6; }",
        "table { font-size: 0.8rem; }",
        ".stat-card { background: white; border-radius: 8px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }",
        ".sidebar { position: fixed; left: 0; top: 56px; z-index: 1000; transition: transform 0.3s ease; width: 250px; }",
        ".sidebar.hidden { transform: translateX(-100%); }",
        ".hamburger-btn { background: none; border: none; color: white; font-size: 1.5rem; padding: 0.5rem; cursor: pointer; margin-right: 0.5rem; }",
        ".hamburger-btn:hover { opacity: 0.8; }",
        ".sidebar-overlay { display: none; position: fixed; top: 56px; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); z-index: 999; }",
        ".sidebar-overlay.show { display: block; }",
        "@media (min-width: 768px) { .sidebar-overlay { display: none !important; } }",
        "main { transition: margin-left 0.3s ease; }",
        "@media (min-width: 768px) { .sidebar:not(.hidden) + * { margin-left: 250px; } }",
        "</style>",
        "</head>",
        "<body>",
        "<nav class='navbar navbar-dark bg-dark px-3 mb-3'>",
        "<button class='hamburger-btn' id='sidebar-toggle' aria-label='Toggle menu'>☰</button>",
        "<span class='navbar-brand ms-2'>Residential Energy Spending per Household</span>",
        "</nav>",
        "<div class='sidebar-overlay' id='sidebar-overlay'></div>",
        "<div class='container-fluid'>",
        "<div class='row'>",
        "<aside class='col-md-3 col-lg-2 sidebar' id='sidebar'>",
        "<h6>Time Period</h6>",
        "<div class='mb-3'>",
        "<label for='granularity' class='form-label'>View</label>",
        "<select id='granularity' class='form-select form-select-sm'>"
        "<option value='hourly'>Hourly</option>"
        "<option value='daily'>Daily</option>"
        "<option value='weekly'>Weekly</option>"
        "<option value='monthly' selected>Monthly</option>"
        "<option value='yearly'>Yearly</option>"
        "</select>",
        "</div>",
        "</aside>",
        "<main class='col-md-9 col-lg-10 py-3'>",
    ]

    # Summary stats cards
    if insights:
        html_parts.append("<div class='row mb-4'>")
        if "avg_monthly" in insights:
            html_parts.append(
                f"<div class='col-md-3 mb-3'><div class='stat-card'>"
                f"<div class='text-muted small'>Avg Monthly (per Household)</div>"
                f"<div class='insight-value'>${insights['avg_monthly']:.2f}</div>"
                f"</div></div>"
            )
        if "total_yearly" in insights and insights["total_yearly"]:
            html_parts.append(
                f"<div class='col-md-3 mb-3'><div class='stat-card'>"
                f"<div class='text-muted small'>Total Yearly (per Household)</div>"
                f"<div class='insight-value'>${insights['total_yearly']:.2f}</div>"
                f"</div></div>"
            )
        if "peak_hour" in insights:
            html_parts.append(
                f"<div class='col-md-3 mb-3'><div class='stat-card'>"
                f"<div class='text-muted small'>Peak Hour</div>"
                f"<div class='insight-value'>{insights['peak_hour']}:00</div>"
                f"<div class='small text-muted'>${insights['peak_hour_cost']:.4f}/hr per household</div>"
                f"</div></div>"
            )
        if "peak_month" in insights:
            html_parts.append(
                f"<div class='col-md-3 mb-3'><div class='stat-card'>"
                f"<div class='text-muted small'>Peak Month</div>"
                f"<div class='insight-value'>{insights['peak_month']}</div>"
                f"<div class='small text-muted'>${insights['peak_month_cost']:.2f} per household</div>"
                f"</div></div>"
            )
        html_parts.append("</div>")

    # Predictions
    if not preds.empty:
        html_parts.append(
            "<div class='card mb-3'><div class='card-header fw-semibold'>Upcoming Predictions</div><div class='card-body'>"
        )
        preds_display = preds.copy()
        preds_display["prediction"] = preds_display["prediction"].apply(
            lambda x: f"${x:,.0f}")
        html_parts.append(
            preds_display.to_html(
                index=False, classes="table table-sm table-striped mb-0")
        )
        html_parts.append("</div></div>")

    # Hourly section
    html_parts.append("<div id='hourly-section' style='display:none;'>")
    if not hourly.empty:
        # Hour of day heatmap
        try:
            hourly_pivot = hourly.groupby(["Date", "hour"])[
                "Estimated_Hourly_Cost_USD"].mean().reset_index()
            hourly_pivot = hourly_pivot.pivot_table(
                index="Date", columns="hour", values="Estimated_Hourly_Cost_USD", aggfunc="mean")

            if not hourly_pivot.empty:
                fig_hourly = go.Figure(data=go.Heatmap(
                    z=hourly_pivot.values,
                    x=hourly_pivot.columns.tolist(),
                    y=[str(d.date()) for d in hourly_pivot.index],
                    colorscale="Viridis",
                    colorbar=dict(title="Cost per Household (USD)")
                ))
                fig_hourly.update_layout(
                    title="Hourly Residential Spending per Household (Cost per Hour)",
                    xaxis_title="Hour of Day",
                    yaxis_title="Date",
                    height=600,
                    xaxis=dict(dtick=1)
                )
                html_parts.append(
                    "<div class='card mb-3'><div class='card-header fw-semibold'>Hourly Spending Pattern (per Household)</div><div class='card-body'>"
                )
                html_parts.append(fig_hourly.to_html(
                    full_html=False, include_plotlyjs=False))
                html_parts.append("</div></div>")
        except Exception:
            pass  # Skip heatmap if pivot fails

        # Average by hour of day
        hourly_avg = hourly.groupby(
            "hour")["Estimated_Hourly_Cost_USD"].mean().reset_index()
        fig_hour_avg = px.line(
            hourly_avg, x="hour", y="Estimated_Hourly_Cost_USD",
            title="Average Residential Cost per Household per Hour of Day",
            labels={"hour": "Hour of Day",
                    "Estimated_Hourly_Cost_USD": "Cost per Household (USD)"},
            markers=True
        )
        fig_hour_avg.update_xaxes(dtick=1)
        html_parts.append(
            "<div class='card mb-3'><div class='card-header fw-semibold'>Average Cost by Hour</div><div class='card-body'>"
        )
        html_parts.append(fig_hour_avg.to_html(
            full_html=False, include_plotlyjs=False))
        html_parts.append("</div></div>")

        # Insights
        if "peak_hour" in insights:
            html_parts.append(
                f"<div class='card mb-3 insight-card'><div class='card-body'>"
                f"<h6 class='text-primary'>💡 Focus Area: Peak Hours</h6>"
                f"<p>Peak spending per household occurs at <strong>{insights['peak_hour']}:00</strong> "
                f"(${insights['peak_hour_cost']:.4f}/hr) vs off-peak at <strong>{insights['off_peak_hour']}:00</strong> "
                f"(${insights['off_peak_cost']:.4f}/hr).</p>"
                f"<p class='mb-0'><strong>Action:</strong> Shift high-energy activities (laundry, charging) to off-peak hours to save up to ${(insights['peak_hour_cost'] - insights['off_peak_cost']) * 24 * 30:,.2f}/month per household.</p>"
                f"</div></div>"
            )
    else:
        html_parts.append(
            "<p class='text-muted'>No hourly data available.</p>")
    html_parts.append("</div>")

    # Daily section - show hourly data for a selected day
    html_parts.append("<div id='daily-section' style='display:none;'>")
    if not hourly.empty and "Estimated_Hourly_Cost_USD" in hourly:
        # Get available dates for the selector
        hourly["date_only"] = hourly["timestamp"].dt.date
        available_dates = sorted(hourly["date_only"].unique(), reverse=True)
        default_date = available_dates[0] if len(available_dates) > 0 else None

        # Create initial chart for default date
        if default_date:
            day_data = hourly[hourly["date_only"] == default_date].copy()
            day_data = day_data.sort_values("timestamp")
            day_data["hour_label"] = day_data["timestamp"].dt.strftime("%H:00")

            fig_daily = px.line(
                day_data, x="hour_label", y="Estimated_Hourly_Cost_USD",
                title=f"Hourly Residential Spending per Household - {default_date}",
                labels={"hour_label": "Hour of Day",
                        "Estimated_Hourly_Cost_USD": "Cost per Household per Hour (USD)"},
                markers=True
            )
            fig_daily.update_xaxes(dtick=2)
            fig_daily.update_traces(line=dict(width=2))

            html_parts.append(
                "<div class='card mb-3'><div class='card-header fw-semibold'>Daily Spending by Hour</div><div class='card-body'>"
            )
            html_parts.append(
                "<div class='mb-3'><label for='daily-date-select' class='form-label'>Select Date:</label>"
                "<select id='daily-date-select' class='form-select form-select-sm' style='max-width: 200px;'>"
            )
            for date in available_dates:
                selected = "selected" if date == default_date else ""
                html_parts.append(
                    f"<option value='{date}' {selected}>{date}</option>")
            html_parts.append("</select></div>")
            html_parts.append("<div id='daily-chart-container'>")
            html_parts.append(fig_daily.to_html(
                full_html=False, include_plotlyjs=False))
            html_parts.append("</div></div></div>")

            # Prepare data for JavaScript (properly format as JSON)
            daily_data_json = {}
            for date in available_dates:
                day_data = hourly[hourly["date_only"] == date].copy()
                day_data = day_data.sort_values("timestamp")
                daily_data_json[str(date)] = {
                    "hours": day_data["timestamp"].dt.strftime("%H:00").tolist(),
                    "costs": [float(x) for x in day_data["Estimated_Hourly_Cost_USD"].tolist()]
                }

            # Add JavaScript to update chart when date changes
            html_parts.append(f"""
            <script>
            const dailyData = {json.dumps(daily_data_json)};
            const dateSelect = document.getElementById('daily-date-select');
            const chartContainer = document.getElementById('daily-chart-container');
            
            dateSelect.addEventListener('change', function() {{
                const selectedDate = dateSelect.value;
                const data = dailyData[selectedDate];
                
                if (!data) {{
                    chartContainer.innerHTML = '<p>No data available for this date.</p>';
                    return;
                }}
                
                const trace = {{
                    x: data.hours,
                    y: data.costs,
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: {{ width: 2 }},
                    marker: {{ size: 6 }}
                }};
                
                const layout = {{
                    title: `Hourly Residential Spending per Household - ${{selectedDate}}`,
                    xaxis: {{ title: 'Hour of Day', dtick: 2 }},
                    yaxis: {{ title: 'Cost per Household per Hour (USD)' }},
                    height: 400
                }};
                
                Plotly.newPlot(chartContainer, [trace], layout, {{responsive: true}});
            }});
            </script>
            """)
        else:
            html_parts.append(
                "<p class='text-muted'>No hourly data available for daily view.</p>")
    elif not daily.empty and "Estimated_Hourly_Cost_USD" in daily:
        # Fallback to daily aggregated data if hourly not available
        daily["date"] = pd.to_datetime(daily["date"])
        available_dates = sorted(daily["date"].dt.date.unique(), reverse=True)
        default_date = available_dates[0] if len(available_dates) > 0 else None

        if default_date:
            html_parts.append(
                "<div class='card mb-3'><div class='card-header fw-semibold'>Daily Spending</div><div class='card-body'>"
            )
            html_parts.append(
                "<div class='mb-3'><label for='daily-date-select' class='form-label'>Select Date:</label>"
                "<select id='daily-date-select' class='form-select form-select-sm' style='max-width: 200px;'>"
            )
            for date in available_dates:
                selected = "selected" if date == default_date else ""
                html_parts.append(
                    f"<option value='{date}' {selected}>{date}</option>")
            html_parts.append("</select></div>")
            html_parts.append(
                f"<p>Total cost for {default_date}: ${daily[daily['date'].dt.date == default_date]['Estimated_Hourly_Cost_USD'].iloc[0]:.2f}</p>"
            )
            html_parts.append("</div></div>")
        else:
            html_parts.append(
                "<p class='text-muted'>No daily data available.</p>")
    else:
        html_parts.append("<p class='text-muted'>No daily data available.</p>")

    # Average by day of week (show if we have daily data)
    if not daily.empty and "Estimated_Hourly_Cost_USD" in daily:
        daily["date"] = pd.to_datetime(daily["date"])
        daily["day_name"] = daily["date"].dt.day_name()
        daily_avg = daily.groupby("day_name")[
            "Estimated_Hourly_Cost_USD"].mean().reset_index()
        day_order = ["Monday", "Tuesday", "Wednesday",
                     "Thursday", "Friday", "Saturday", "Sunday"]
        daily_avg["day_name"] = pd.Categorical(
            daily_avg["day_name"], categories=day_order, ordered=True)
        daily_avg = daily_avg.sort_values("day_name")
        fig_day_avg = px.line(
            daily_avg, x="day_name", y="Estimated_Hourly_Cost_USD",
            title="Average Residential Cost per Household by Day of Week",
            labels={"day_name": "Day",
                    "Estimated_Hourly_Cost_USD": "Cost per Household per Day (USD)"},
            markers=True
        )
        html_parts.append(
            "<div class='card mb-3'><div class='card-header fw-semibold'>Average Cost by Day of Week</div><div class='card-body'>"
        )
        html_parts.append(fig_day_avg.to_html(
            full_html=False, include_plotlyjs=False))
        html_parts.append("</div></div>")

        # Insights for daily
        if "peak_day" in insights:
            html_parts.append(
                f"<div class='card mb-3 insight-card'><div class='card-body'>"
                f"<h6 class='text-primary'>💡 Focus Area: Peak Days</h6>"
                f"<p>Peak spending per household occurs on <strong>{insights['peak_day']}s</strong> "
                f"(${insights['peak_day_cost']:.2f}/day).</p>"
                f"<p class='mb-0'><strong>Action:</strong> Schedule energy-intensive activities on lower-cost days when possible.</p>"
                f"</div></div>"
            )
    html_parts.append("</div>")

    # Weekly section
    html_parts.append("<div id='weekly-section' style='display:none;'>")
    if not weekly.empty:
        fig_weekly = px.line(
            weekly.sort_values("week_start"), x="week_start", y="Estimated_Hourly_Cost_USD",
            title="Weekly Residential Spending per Household",
            labels={"week_start": "Week Starting",
                    "Estimated_Hourly_Cost_USD": "Cost per Household per Week (USD)"},
            markers=True
        )
        html_parts.append(
            "<div class='card mb-3'><div class='card-header fw-semibold'>Weekly Spending</div><div class='card-body'>"
        )
        html_parts.append(fig_weekly.to_html(
            full_html=False, include_plotlyjs=False))
        html_parts.append("</div></div>")

        html_parts.append(
            "<div class='card mb-3'><div class='card-header fw-semibold'>Weekly Spending (all available)</div><div class='card-body table-responsive'>"
        )
        weekly_display = weekly[["week_start",
                                 "Estimated_Hourly_Cost_USD"]].copy()
        weekly_display.columns = ["Week Starting",
                                  "Total Cost per Household (USD)"]
        weekly_display["Total Cost per Household (USD)"] = weekly_display["Total Cost per Household (USD)"].apply(
            lambda x: f"${x:.2f}")
        html_parts.append(
            weekly_display.sort_values("Week Starting", ascending=False).to_html(
                index=False, classes="table table-sm table-striped table-hover mb-0"
            )
        )
        html_parts.append("</div></div>")
    else:
        html_parts.append(
            "<p class='text-muted'>No weekly data available.</p>")
    html_parts.append("</div>")

    # Monthly section
    html_parts.append("<div id='monthly-section' style='display:block;'>")
    if not monthly.empty and "Estimated_Hourly_Cost_USD" in monthly:
        monthly["year_month_start"] = pd.to_datetime(
            monthly["year_month_start"])
        fig_month = px.line(
            monthly.sort_values("year_month_start"), x="year_month_start", y="Estimated_Hourly_Cost_USD",
            title="Monthly Residential Spending per Household",
            labels={"year_month_start": "Month",
                    "Estimated_Hourly_Cost_USD": "Cost per Household per Month (USD)"},
            markers=True
        )
        fig_month.update_xaxes(rangeslider_visible=True)
        html_parts.append(
            "<div class='card mb-3'><div class='card-header fw-semibold'>Monthly Spending</div><div class='card-body'>"
        )
        html_parts.append(fig_month.to_html(
            full_html=False, include_plotlyjs=False))
        html_parts.append("</div></div>")

        # Average by month
        monthly["month_name"] = monthly["year_month_start"].dt.strftime("%B")
        monthly_avg = monthly.groupby("month_name")[
            "Estimated_Hourly_Cost_USD"].mean().reset_index()
        month_order = ["January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"]
        monthly_avg["month_name"] = pd.Categorical(
            monthly_avg["month_name"], categories=month_order, ordered=True)
        monthly_avg = monthly_avg.sort_values("month_name")
        fig_month_avg = px.line(
            monthly_avg, x="month_name", y="Estimated_Hourly_Cost_USD",
            title="Average Residential Cost per Household by Month",
            labels={"month_name": "Month",
                    "Estimated_Hourly_Cost_USD": "Cost per Household per Month (USD)"},
            markers=True
        )
        html_parts.append(
            "<div class='card mb-3'><div class='card-header fw-semibold'>Average Cost by Month</div><div class='card-body'>"
        )
        html_parts.append(fig_month_avg.to_html(
            full_html=False, include_plotlyjs=False))
        html_parts.append("</div></div>")

        # Insights
        if "peak_month" in insights:
            html_parts.append(
                f"<div class='card mb-3 insight-card'><div class='card-body'>"
                f"<h6 class='text-primary'>💡 Focus Area: Peak Months</h6>"
                f"<p>Peak spending per household occurs in <strong>{insights['peak_month']}</strong> "
                f"(${insights['peak_month_cost']:.2f}/month).</p>"
                f"<p class='mb-0'><strong>Action:</strong> Plan major energy-intensive activities (HVAC maintenance, insulation upgrades) before peak months to reduce costs.</p>"
                f"</div></div>"
            )
    else:
        html_parts.append(
            "<p class='text-muted'>No monthly data available.</p>")
    html_parts.append("</div>")

    # Yearly section
    html_parts.append("<div id='yearly-section' style='display:none;'>")
    if not monthly.empty and "Estimated_Hourly_Cost_USD" in monthly:
        monthly["year_month_start"] = pd.to_datetime(
            monthly["year_month_start"])
        monthly["year"] = monthly["year_month_start"].dt.year
        monthly["month"] = monthly["year_month_start"].dt.month

        month_counts = monthly.groupby("year")["month"].agg(
            ["count", "min", "max"]).reset_index()
        month_counts.columns = [
            "year", "month_count", "min_month", "max_month"]

        yearly = (
            monthly.groupby("year")
            .agg({"Estimated_Hourly_Cost_USD": "sum", "CAISO Total": "mean", "Monthly_Price_Cents_per_kWh": "mean"})
            .reset_index()
        )
        yearly = yearly.merge(month_counts, on="year")

        def make_year_label(row):
            if row["month_count"] == 12:
                return str(int(row["year"]))
            else:
                month_names = ["Jan", "Feb", "Mar", "Apr", "May",
                               "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                min_name = month_names[int(row["min_month"]) - 1]
                max_name = month_names[int(row["max_month"]) - 1]
                return f"{int(row['year'])} ({min_name}-{max_name})"

        yearly["year_label"] = yearly.apply(make_year_label, axis=1)
        yearly = yearly[yearly["month_count"] >= 6].copy()

        fig_year = px.line(
            yearly, x="year_label", y="Estimated_Hourly_Cost_USD",
            title="Yearly Residential Spending per Household",
            labels={"year_label": "Year",
                    "Estimated_Hourly_Cost_USD": "Cost per Household per Year (USD)"},
            markers=True
        )
        html_parts.append(
            "<div class='card mb-3'><div class='card-header fw-semibold'>Yearly Spending</div><div class='card-body'>"
        )
        html_parts.append(fig_year.to_html(
            full_html=False, include_plotlyjs=False))
        html_parts.append("</div></div>")

        yearly_sorted = yearly.sort_values("year", ascending=False)
        display_yearly = yearly_sorted[[
            "year_label", "Estimated_Hourly_Cost_USD", "Monthly_Price_Cents_per_kWh", "month_count"]].copy()
        display_yearly.columns = [
            "Year", "Total Cost per Household (USD)", "Avg Price (cents/kWh)", "Months"]
        display_yearly["Total Cost per Household (USD)"] = display_yearly["Total Cost per Household (USD)"].apply(
            lambda x: f"${x:,.0f}")
        html_parts.append(
            "<div class='card mb-3'><div class='card-header fw-semibold'>Yearly Summary</div><div class='card-body table-responsive'>"
        )
        html_parts.append(display_yearly.to_html(
            index=False, classes="table table-sm table-striped table-hover mb-0"))
        html_parts.append("</div></div>")
    else:
        html_parts.append(
            "<p class='text-muted'>No yearly data available.</p>")
    html_parts.append("</div>")

    # JavaScript for toggling
    html_parts.append("""
<script>
  // Sidebar toggle functionality
  const sidebarToggle = document.getElementById('sidebar-toggle');
  const sidebar = document.getElementById('sidebar');
  const sidebarOverlay = document.getElementById('sidebar-overlay');
  
  function toggleSidebar() {
    sidebar.classList.toggle('hidden');
    if (window.innerWidth < 768) {
      sidebarOverlay.classList.toggle('show');
    }
  }
  
  sidebarToggle.addEventListener('click', toggleSidebar);
  if (sidebarOverlay) {
    sidebarOverlay.addEventListener('click', toggleSidebar);
  }
  
  // Initialize sidebar state - visible on desktop, hidden on mobile
  function updateSidebarState() {
    if (window.innerWidth >= 768) {
      sidebar.classList.remove('hidden');
      if (sidebarOverlay) sidebarOverlay.classList.remove('show');
    } else {
      sidebar.classList.add('hidden');
      if (sidebarOverlay) sidebarOverlay.classList.remove('show');
    }
  }
  
  updateSidebarState();
  window.addEventListener('resize', updateSidebarState);
  
  // Time period view toggling
  const select = document.getElementById('granularity');
  const sections = {
    hourly: document.getElementById('hourly-section'),
    daily: document.getElementById('daily-section'),
    weekly: document.getElementById('weekly-section'),
    monthly: document.getElementById('monthly-section'),
    yearly: document.getElementById('yearly-section'),
  };
  function updateView() {
    const val = select.value;
    for (const key in sections) {
      sections[key].style.display = key === val ? 'block' : 'none';
    }
  }
  select.addEventListener('change', updateView);
  updateView();
</script>
""")

    html_parts.append("</main></div></div></body></html>")
    out_path = SITE_DIR / "index.html"
    out_path.write_text("\n".join(html_parts), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    path = build()
    print(f"Built dashboard at {path}")
