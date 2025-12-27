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

    html_parts = [
        "<html>",
        "<head>",
        "<title>Residential Energy Spending Predictions Dashboard</title>",
        "<link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css'>",
        "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
        "<style>",
        "body { font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f8f9fa; padding-top: 56px; }",
        ".navbar { z-index: 1030; margin-bottom: 0 !important; width: 100%; }",
        ".sidebar { background-color: #212529; color: #e5e7eb; min-height: calc(100vh - 56px); padding: 1rem; position: fixed; left: 0; top: 56px; z-index: 1020; transition: transform 0.3s ease, margin-left 0.3s ease; width: 250px; border-top: 1px solid rgba(255,255,255,0.1); }",
        ".sidebar h6 { font-size: 0.75rem; letter-spacing: .08em; text-transform: uppercase; color: #9ca3af; margin-bottom: 1rem; }",
        ".sidebar .form-label { font-size: 0.8rem; color: #d1d5db; margin-bottom: 0.5rem; }",
        ".sidebar .form-select { background-color: #343a40; border-color: #495057; color: #e5e7eb; }",
        ".sidebar .form-select:focus { background-color: #343a40; border-color: #6c757d; color: #e5e7eb; }",
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
        "</style>",
        "</head>",
        "<body>",
        "<nav class='navbar navbar-dark bg-dark px-3 fixed-top'>",
        "<button class='hamburger-btn' id='sidebar-toggle' aria-label='Toggle menu'>☰</button>",
        "<span class='navbar-brand ms-2'>Residential Energy Spending Predictions</span>",
        "</nav>",
        "<div class='sidebar-overlay' id='sidebar-overlay'></div>",
        "<div class='container-fluid'>",
        "<div class='row'>",
        "<aside class='col-md-3 col-lg-2 sidebar' id='sidebar'>",
        "<h6>Prediction Period</h6>",
        "<div class='mb-3'>",
        "<label for='granularity' class='form-label'>View</label>",
        "<select id='granularity' class='form-select form-select-sm'>"
        "<option value='hourly'>Hourly</option>"
        "<option value='daily'>Daily</option>"
        "<option value='weekly'>Weekly</option>"
        "<option value='monthly' selected>Monthly</option>"
        "</select>",
        "</div>",
        "</aside>",
        "<main class='col-md-9 col-lg-10 py-3'>",
    ]

    # Upcoming Predictions Section (Core)
    if not preds.empty:
        html_parts.append(
            "<div class='card mb-3'><div class='card-header fw-semibold'>Upcoming Predictions</div><div class='card-body'>"
        )
        preds_display = preds.copy()
        
        # Ensure feature_date is datetime
        preds_display["feature_date"] = pd.to_datetime(preds_display["feature_date"])
        
        def format_prediction(row):
            pred_val = row["prediction"]
            if "hour" in str(row["for"]):
                return f"${pred_val:.4f}"
            elif "day" in str(row["for"]) or "week" in str(row["for"]):
                return f"${pred_val:.2f}"
            else:
                return f"${pred_val:,.2f}"
                
        preds_display["prediction_text"] = preds_display.apply(format_prediction, axis=1)
        
        # Add a chart for upcoming predictions
        html_parts.append("<div id='predictions-chart-container' class='mb-4'></div>")
        
        # Prepare JSON for JS-based charting of predictions
        preds_json = []
        for _, row in preds_display.iterrows():
            preds_json.append({
                "date": row["feature_date"].strftime("%Y-%m-%d %H:%M:%S"),
                "val": float(row["prediction"]),
                "for": str(row["for"]),
                "target": str(row["target"])
            })
        
        html_parts.append(f"""
<script>
const upcomingPreds = {json.dumps(preds_json)};
function updatePredictionsChart(granularity) {{
    const container = document.getElementById('predictions-chart-container');
    if (!container) return;
    
    const filtered = upcomingPreds.filter(p => {{
        if (granularity === 'hourly') return p.for.includes('hour');
        if (granularity === 'daily') return p.for.includes('day');
        if (granularity === 'weekly') return p.for.includes('week');
        if (granularity === 'monthly') return p.for.includes('month');
        return false;
    }});
    
    if (filtered.length === 0) {{
        container.innerHTML = '<p class=\"text-muted\">No upcoming predictions for this view.</p>';
        return;
    }}
    
    const trace = {{
        x: filtered.map(p => p.date),
        y: filtered.map(p => p.val),
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Predicted Cost',
        line: {{ color: '#3b82f6', width: 3 }},
        marker: {{ size: 8 }}
    }};
    
    const layout = {{
        title: `Upcoming ${{granularity.charAt(0).toUpperCase() + granularity.slice(1)}} Predictions`,
        xaxis: {{ title: 'Time' }},
        yaxis: {{ title: 'Predicted Cost (USD)' }},
        margin: {{ t: 40, b: 40, l: 60, r: 20 }},
        height: 450
    }};
    
    Plotly.newPlot(container, [trace], layout, {{responsive: true}});
}}
</script>
""")

        # Add the table with a specific ID and custom row attributes for filtering
        html_parts.append("<div class='table-responsive'>")
        html_parts.append("<table id='predictions-table' class='table table-sm table-striped mb-0'>")
        html_parts.append("<thead><tr><th>Prediction</th><th>Time</th></tr></thead>")
        html_parts.append("<tbody>")
        
        for _, row in preds_display.iterrows():
            f_date = row["feature_date"].strftime("%Y-%m-%d")
            f_month = row["feature_date"].strftime("%Y-%m")
            f_for = str(row["for"])
            
            row_class = "prediction-row"
            html_parts.append(f"<tr class='{row_class}' data-date='{f_date}' data-month='{f_month}' data-for='{f_for}'>")
            html_parts.append(f"<td>{row['prediction_text']}</td>")
            html_parts.append(f"<td>{row['feature_date']}</td>")
            html_parts.append("</tr>")
            
        html_parts.append("</tbody></table></div>")
        html_parts.append("</div></div>")

    # JavaScript for toggling
    html_parts.append("""
<script>
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
  
  sidebarToggle.addEventListener('click', toggleSidebar);
  if (sidebarOverlay) {
    sidebarOverlay.addEventListener('click', toggleSidebar);
  }
  
  // Initialize sidebar state - visible on desktop, hidden on mobile
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
  
  // Time period view toggling
  const select = document.getElementById('granularity');
  
  function updateTableFiltering(granularity) {
    const rows = document.querySelectorAll('.prediction-row');
    rows.forEach(row => {
      const type = row.getAttribute('data-for') || "";
      let show = false;
      if (granularity === 'hourly' && type.includes('hour')) show = true;
      else if (granularity === 'daily' && type.includes('day')) show = true;
      else if (granularity === 'weekly' && type.includes('week')) show = true;
      else if (granularity === 'monthly' && type.includes('month')) show = true;
      
      row.style.display = show ? '' : 'none';
    });
  }

  function runFilter() {
    const granularity = select.value;
    updateTableFiltering(granularity);
    updatePredictionsChart(granularity);
  }

  select.addEventListener('change', runFilter);

  // Initial update
  runFilter();
</script>
""")

    html_parts.append("</main></div></div></body></html>")
    out_path = SITE_DIR / "index.html"
    out_path.write_text("\n".join(html_parts), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    path = build()
    print(f"Built dashboard at {path}")
