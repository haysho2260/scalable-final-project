"""Build a static dashboard from results files.

Outputs a standalone HTML at `site/index.html` showing:
- Next-day and next-month predictions.
- Daily and monthly spend history with interactive navigation (range sliders).
- Recent tables for quick lookback.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
SITE_DIR = ROOT / "site"


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def build():
    SITE_DIR.mkdir(parents=True, exist_ok=True)

    preds = _read_csv(RESULTS_DIR / "predictions.csv")
    daily = _read_csv(RESULTS_DIR / "daily_history.csv")
    monthly = _read_csv(RESULTS_DIR / "monthly_history.csv")

    html_parts = [
        "<html>",
        "<head>",
        "<title>Energy Spend Dashboard</title>",
        "<link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap@5.3.0/dist/css/bootstrap.min.css'>",
        "<style>",
        "body { font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }",
        ".sidebar { background-color: #111827; color: #e5e7eb; min-height: 100vh; }",
        ".sidebar h6 { font-size: 0.75rem; letter-spacing: .08em; text-transform: uppercase; color: #9ca3af; }",
        ".sidebar .form-label { font-size: 0.8rem; color: #d1d5db; }",
        "table { font-size: 0.8rem; }",
        "</style>",
        "</head>",
        "<body class='bg-light'>",
        "<nav class='navbar navbar-dark bg-dark px-3 mb-3'>",
        "<span class='navbar-brand'>Energy Spend Dashboard</span>",
        "</nav>",
        "<div class='container-fluid'>",
        "<div class='row'>",
        "<aside class='col-md-3 col-lg-2 sidebar py-3 d-none d-md-block'>",
        "<h6 class='mb-3'>Controls</h6>",
        "<div class='mb-3'>",
        "<label for='granularity' class='form-label mb-1'>Granularity</label>",
        "<select id='granularity' class='form-select form-select-sm bg-dark text-light border-secondary'>"
        "<option value='daily'>Daily</option>"
        "<option value='monthly' selected>Monthly</option>"
        "<option value='yearly'>Yearly</option>"
        "</select>",
        "</div>",
        "</aside>",
        "<main class='col-md-9 col-lg-10 py-3'>",
    ]

    if not preds.empty:
        html_parts.append(
            "<div class='card mb-3'><div class='card-header fw-semibold'>Next Predictions</div><div class='card-body'>"
        )
        html_parts.append(
            preds.to_html(
                index=False, classes="table table-sm table-striped mb-0")
        )
        html_parts.append("</div></div>")
    else:
        html_parts.append(
            "<p class='text-muted'>No predictions available.</p>")

    # Daily section
    html_parts.append("<div id='daily-section' style='display:none;'>")
    if not daily.empty and "Estimated_Hourly_Cost_USD" in daily:
        daily["date"] = pd.to_datetime(daily["date"])
        fig_daily = px.line(
            daily.sort_values("date"),
            x="date",
            y="Estimated_Hourly_Cost_USD",
            title="Daily Spend (USD)",
        )
        fig_daily.update_xaxes(rangeslider_visible=True)
        html_parts.append(
            "<div class='card mb-3'><div class='card-header fw-semibold'>Daily Spend</div><div class='card-body'>"
        )
        html_parts.append(
            fig_daily.to_html(full_html=False, include_plotlyjs="cdn")
        )
        html_parts.append("</div></div>")
        # Daily table (all rows)
        html_parts.append(
            "<div class='card mb-3'><div class='card-header fw-semibold'>Daily Spend (all available)</div><div class='card-body table-responsive'>"
        )
        html_parts.append(
            daily.sort_values("date", ascending=False).to_html(
                index=False, classes="table table-sm table-striped table-hover mb-0"
            )
        )
        html_parts.append("</div></div>")
    else:
        html_parts.append(
            "<p class='text-muted'>No daily history available.</p>")
    html_parts.append("</div>")

    # Monthly section
    html_parts.append("<div id='monthly-section' style='display:block;'>")
    if not monthly.empty and "Estimated_Hourly_Cost_USD" in monthly:
        monthly["year_month_start"] = pd.to_datetime(
            monthly["year_month_start"])
        fig_month = px.bar(
            monthly.sort_values("year_month_start"),
            x="year_month_start",
            y="Estimated_Hourly_Cost_USD",
            title="Monthly Spend (USD)",
        )
        fig_month.update_xaxes(rangeslider_visible=True)
        html_parts.append(
            "<div class='card mb-3'><div class='card-header fw-semibold'>Monthly Spend</div><div class='card-body'>"
        )
        html_parts.append(
            fig_month.to_html(full_html=False, include_plotlyjs="cdn")
        )
        html_parts.append("</div></div>")
        html_parts.append(
            "<div class='card mb-3'><div class='card-header fw-semibold'>Monthly Spend (all available)</div><div class='card-body table-responsive'>"
        )
        html_parts.append(
            monthly.sort_values("year_month_start", ascending=False)
            .to_html(index=False, classes="table table-sm table-striped table-hover mb-0")
        )
        html_parts.append("</div></div>")
    else:
        html_parts.append(
            "<p class='text-muted'>No monthly history available.</p>")
    html_parts.append("</div>")

    # Yearly section (aggregated from monthly)
    html_parts.append("<div id='yearly-section' style='display:none;'>")
    if not monthly.empty and "Estimated_Hourly_Cost_USD" in monthly:
        monthly["year_month_start"] = pd.to_datetime(
            monthly["year_month_start"])
        monthly["year"] = monthly["year_month_start"].dt.year
        yearly = (
            monthly.groupby("year")
            .agg(
                {
                    "Estimated_Hourly_Cost_USD": "sum",
                    "CAISO Total": "mean",
                    "Monthly_Price_Cents_per_kWh": "mean",
                }
            )
            .reset_index()
        )
        fig_year = px.bar(
            yearly,
            x="year",
            y="Estimated_Hourly_Cost_USD",
            title="Yearly Spend (USD)",
        )
        html_parts.append(
            "<div class='card mb-3'><div class='card-header fw-semibold'>Yearly Spend</div><div class='card-body'>"
        )
        html_parts.append(
            fig_year.to_html(full_html=False, include_plotlyjs="cdn")
        )
        html_parts.append("</div></div>")
        html_parts.append(
            "<div class='card mb-3'><div class='card-header fw-semibold'>Yearly Spend (all available)</div><div class='card-body table-responsive'>"
        )
        html_parts.append(
            yearly.sort_values("year", ascending=False).to_html(
                index=False, classes="table table-sm table-striped table-hover mb-0"
            )
        )
        html_parts.append("</div></div>")
    else:
        html_parts.append(
            "<p class='text-muted'>No yearly history available.</p>")
    html_parts.append("</div>")

    # Simple JS to toggle sections based on dropdown
    html_parts.append(
        """
<script>
  const select = document.getElementById('granularity');
  const sections = {
    daily: document.getElementById('daily-section'),
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
"""
    )

    html_parts.append("</main></div></div></body></html>")

    out_path = SITE_DIR / "index.html"
    out_path.write_text("\n".join(html_parts), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    path = build()
    print(f"Built dashboard at {path}")
