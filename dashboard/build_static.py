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
        ".navbar { z-index: 1030; }",
        ".sidebar { background-color: #212529; color: #e5e7eb; min-height: calc(100vh - 56px); padding: 1rem; position: fixed; left: 0; top: 56px; z-index: 1020; transition: transform 0.3s ease; width: 250px; border-top: 1px solid rgba(255,255,255,0.1); }",
        ".sidebar h6 { font-size: 0.75rem; letter-spacing: .08em; text-transform: uppercase; color: #9ca3af; margin-bottom: 1rem; }",
        ".sidebar .form-label { font-size: 0.8rem; color: #d1d5db; margin-bottom: 0.5rem; }",
        ".sidebar .form-select { background-color: #343a40; border-color: #495057; color: #e5e7eb; }",
        ".sidebar .form-select:focus { background-color: #343a40; border-color: #6c757d; color: #e5e7eb; }",
        ".insight-card { border-left: 4px solid #3b82f6; }",
        ".insight-value { font-size: 1.5rem; font-weight: 600; color: #3b82f6; }",
        "table { font-size: 0.8rem; }",
        ".stat-card { background: white; border-radius: 8px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }",
        ".sidebar.hidden { transform: translateX(-100%); }",
        ".hamburger-btn { background: none; border: none; color: white; font-size: 1.5rem; padding: 0.5rem; cursor: pointer; margin-right: 0.5rem; }",
        ".hamburger-btn:hover { opacity: 0.8; }",
        ".sidebar-overlay { display: none; position: fixed; top: 56px; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); z-index: 999; }",
        ".sidebar-overlay.show { display: block; }",
        "@media (min-width: 768px) { .sidebar-overlay { display: none !important; } }",
        "@media (min-width: 768px) { .sidebar { position: relative; top: 0; transform: none !important; width: auto; min-height: 100vh; border-top: none; } }",
        "main { transition: margin-left 0.3s ease; }",
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

        def format_prediction(row):
            pred_val = row["prediction"]
            if row["for"] == "next_hour":
                return f"${pred_val:.4f}"
            elif row["for"] == "next_day":
                return f"${pred_val:.2f}"
            else:
                return f"${pred_val:,.0f}"
        preds_display["prediction"] = preds_display.apply(
            format_prediction, axis=1)
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

        # Average Cost by Hour - with date selector
        # Get available dates
        if "Date" in hourly.columns:
            hourly["Date"] = pd.to_datetime(hourly["Date"], errors="coerce")
            available_dates = sorted(
                hourly["Date"].dt.date.unique(), reverse=True)
        elif "timestamp" in hourly.columns:
            available_dates = sorted(
                hourly["timestamp"].dt.date.unique(), reverse=True)
        else:
            available_dates = []

        default_date = available_dates[0] if len(available_dates) > 0 else None

        html_parts.append(
            "<div class='card mb-3'><div class='card-header fw-semibold'>Average Cost by Hour</div><div class='card-body'>"
        )

        if default_date:
            # Date selector and time format toggle
            html_parts.append(
                f"<div class='mb-3 d-flex gap-3 align-items-center flex-wrap'>"
                f"<label for='hourlyDateSelect' class='form-label mb-0'><strong>Select Date:</strong></label>"
                f"<select id='hourlyDateSelect' class='form-select' style='width: auto; min-width: 200px;'>"
            )
            for date in available_dates:
                selected = "selected" if date == default_date else ""
                html_parts.append(
                    f"<option value='{date}' {selected}>{date}</option>")
            html_parts.append("</select>")

            html_parts.append(
                f"<label for='hourlyTimeFormat' class='form-label mb-0'><strong>Time Format:</strong></label>"
                f"<select id='hourlyTimeFormat' class='form-select' style='width: auto; min-width: 150px;'>"
                f"<option value='24'>24-Hour (00:00)</option>"
                f"<option value='12'>12-Hour (12:00 AM)</option>"
                f"</select>"
                f"</div>"
            )

            # Prepare hourly data for JavaScript
            hourly_data_list = []
            date_col = "Date" if "Date" in hourly.columns else "timestamp"
            for _, row in hourly.iterrows():
                try:
                    date_val = row[date_col]
                    if pd.isna(date_val):
                        continue
                    if hasattr(date_val, 'date'):
                        date_str = str(date_val.date())
                    else:
                        date_str = str(date_val)
                    hour_val = int(row["hour"]) if not pd.isna(
                        row["hour"]) else 0
                    # Ensure hour is in valid range 0-23
                    hour_val = max(0, min(23, hour_val))
                    hourly_data_list.append({
                        "date": date_str,
                        "hour": hour_val,
                        "cost": float(row["Estimated_Hourly_Cost_USD"]) if not pd.isna(row["Estimated_Hourly_Cost_USD"]) else 0.0
                    })
                except Exception:
                    continue

            hourly_data_json = json.dumps(hourly_data_list)

            # Generate initial chart for default date
            selected_data = hourly[hourly[date_col].dt.date ==
                                   default_date] if date_col in hourly.columns else pd.DataFrame()
            if not selected_data.empty:
                selected_data = selected_data.sort_values("hour")
                # Format hours for 24-hour display
                hour_labels_24 = [
                    f"{int(h):02d}:00" for h in selected_data["hour"]]

                fig_hour_avg = go.Figure()
                fig_hour_avg.add_trace(go.Scatter(
                    x=selected_data["hour"],
                    y=selected_data["Estimated_Hourly_Cost_USD"],
                    mode='lines+markers',
                    name='Cost per Hour',
                    hovertemplate='<b>%{text}</b><br>Cost: $%{y:.4f}<extra></extra>',
                    text=hour_labels_24
                ))
                fig_hour_avg.update_layout(
                    title=f"Hourly Residential Cost per Household - {default_date}",
                    xaxis_title="Hour of Day",
                    yaxis_title="Cost per Household (USD)",
                    xaxis=dict(
                        tickmode='array',
                        tickvals=selected_data["hour"].tolist(),
                        ticktext=hour_labels_24,
                        dtick=1
                    ),
                    height=500
                )
                html_parts.append(fig_hour_avg.to_html(
                    full_html=False, include_plotlyjs=False, div_id="hourlyChartContainer"))

            # JavaScript to handle date selection and time format toggle
            html_parts.append(f"""
            <script>
            (function() {{
                const dateSelect = document.getElementById('hourlyDateSelect');
                const timeFormatSelect = document.getElementById('hourlyTimeFormat');
                const chartContainer = document.getElementById('hourlyChartContainer');
                const hourlyData = {hourly_data_json};
                
                function formatHour24(hour) {{
                    return String(Math.floor(hour)).padStart(2, '0') + ':00';
                }}
                
                function formatHour12(hour) {{
                    const h = Math.floor(hour);
                    const period = h >= 12 ? 'PM' : 'AM';
                    const displayHour = h === 0 ? 12 : (h > 12 ? h - 12 : h);
                    return displayHour + ':00 ' + period;
                }}
                
                function updateHourlyChart() {{
                    const selectedDate = dateSelect.value;
                    const timeFormat = timeFormatSelect.value;
                    const is24Hour = timeFormat === '24';
                    
                    // Filter data for selected date
                    const filtered = hourlyData.filter(d => d.date === selectedDate);
                    filtered.sort((a, b) => a.hour - b.hour);
                    
                    if (filtered.length === 0) {{
                        chartContainer.innerHTML = '<p class="text-muted">No data available for this date.</p>';
                        return;
                    }}
                    
                    const hours = filtered.map(d => d.hour);
                    const costs = filtered.map(d => d.cost);
                    const hourLabels = hours.map(h => is24Hour ? formatHour24(h) : formatHour12(h));
                    
                    const trace = {{
                        x: hours,
                        y: costs,
                        mode: 'lines+markers',
                        type: 'scatter',
                        name: 'Cost per Hour',
                        hovertemplate: '<b>%{{text}}</b><br>Cost: $%{{y:.4f}}<extra></extra>',
                        text: hourLabels
                    }};
                    
                    const layout = {{
                        title: `Hourly Residential Cost per Household - ${{selectedDate}}`,
                        xaxis: {{
                            title: 'Hour of Day',
                            tickmode: 'array',
                            tickvals: hours,
                            ticktext: hourLabels,
                            dtick: 1
                        }},
                        yaxis: {{ title: 'Cost per Household (USD)' }},
                        height: 500
                    }};
                    
                    Plotly.newPlot('hourlyChartContainer', [trace], layout, {{responsive: true}});
                }}
                
                dateSelect.addEventListener('change', updateHourlyChart);
                timeFormatSelect.addEventListener('change', updateHourlyChart);
            }})();
            </script>
            """)
        else:
            html_parts.append(
                "<p class='text-muted'>No hourly data available.</p>")

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

    # Daily section - show average hourly cost for a selected day
    html_parts.append("<div id='daily-section' style='display:none;'>")
    if not daily.empty and "Estimated_Hourly_Cost_USD" in daily:
        daily["date"] = pd.to_datetime(daily["date"])

        # Calculate average hourly cost per day (daily total / 24 hours)
        daily["avg_hourly_cost"] = daily["Estimated_Hourly_Cost_USD"] / 24.0

        available_dates = sorted(daily["date"].dt.date.unique(), reverse=True)
        default_date = available_dates[0] if len(available_dates) > 0 else None

        if default_date:
            default_avg = daily[daily["date"].dt.date ==
                                default_date]["avg_hourly_cost"].iloc[0]

            # Create a bar chart showing average hourly cost for the selected day
            fig_daily = go.Figure(data=[
                go.Bar(x=[str(default_date)], y=[default_avg],
                       marker_color='#3b82f6', text=[f"${default_avg:.4f}/hr"],
                       textposition='outside')
            ])
            fig_daily.update_layout(
                title=f"Average Hourly Cost per Household - {default_date}",
                xaxis_title="Date",
                yaxis_title="Average Cost per Hour (USD)",
                height=400,
                showlegend=False
            )

            html_parts.append(
                "<div class='card mb-3'><div class='card-header fw-semibold'>Daily Average Hourly Cost</div><div class='card-body'>"
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

            # Prepare data for JavaScript
            daily_avg_json = {}
            for date in available_dates:
                avg_cost = daily[daily["date"].dt.date ==
                                 date]["avg_hourly_cost"].iloc[0]
                daily_avg_json[str(date)] = float(avg_cost)

            # Add JavaScript to update chart when date changes
            html_parts.append(f"""
            <script>
            const dailyAvgData = {json.dumps(daily_avg_json)};
            const dateSelect = document.getElementById('daily-date-select');
            const chartContainer = document.getElementById('daily-chart-container');
            
            dateSelect.addEventListener('change', function() {{
                const selectedDate = dateSelect.value;
                const avgCost = dailyAvgData[selectedDate];
                
                if (avgCost === undefined) {{
                    chartContainer.innerHTML = '<p>No data available for this date.</p>';
                    return;
                }}
                
                const trace = {{
                    x: [selectedDate],
                    y: [avgCost],
                    type: 'bar',
                    marker: {{ color: '#3b82f6' }},
                    text: [`$${{avgCost.toFixed(4)}}/hr`],
                    textposition: 'outside'
                }};
                
                const layout = {{
                    title: `Average Hourly Cost per Household - ${{selectedDate}}`,
                    xaxis: {{ title: 'Date' }},
                    yaxis: {{ title: 'Average Cost per Hour (USD)' }},
                    height: 400,
                    showlegend: false
                }};
                
                Plotly.newPlot(chartContainer, [trace], layout, {{responsive: true}});
            }});
            </script>
            """)
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
