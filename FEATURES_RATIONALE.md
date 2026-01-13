# Features Rationale

This document outlines the data features used in the energy cost prediction model, their sources, and the technical reasoning behind their inclusion.

## Overview

The model predicts the `Estimated_Hourly_Cost_USD` for a typical residential household in California. To achieve high accuracy, we incorporate a diverse set of features that capture meteorological conditions, grid-level demand, market pricing, and historical patterns.

---

## 1. Weather Features
**Source:** [Open-Meteo Historical Archive](https://open-meteo.com/)

Weather is a primary driver of residential energy consumption, particularly for heating and cooling.

### Acquisition Method
We use the `openmeteo-requests` library to interface with the **Open-Meteo Archive API**. Data is fetched for Los Angeles coordinates (34.05, -118.24). The script (`get_weather.py`) implements a local cache to minimize redundant API calls and performs linear interpolation to fill any meteorological data gaps.

### Key Variables
- **Temperature (Mean, Max, Min):** These are the primary drivers of HVAC (Heating, Ventilation, and Air Conditioning) load. Extreme highs and lows force climate control systems to work harder and longer, which accounts for the largest portion of residential energy spend.
- **Apparent Temperature:** Also known as the "Heat Index" or "Wind Chill," it captures how humans actually perceive weather. This is often a more accurate predictor of behavioral changes—like switching on the AC—than raw dry-bulb temperature alone.
- **Dew Point & Humidity:** High humidity levels (high dew point) impede the body's natural cooling and make air conditioning systems significantly less efficient, as they must expend energy to dehumidify the air as well as cool it.
- **Shortwave Radiation (Solar Irradiance):** This measures the amount of solar energy (visible light and ultraviolet) reaching the Earth's surface. It is a critical predictor for two reasons:
    - **Thermal Gain:** High radiation significantly increases the "solar gain" of buildings, heating up interiors even when air temperatures are moderate, which drives an earlier or stronger demand for air conditioning.
    - **Distributed Generation:** In California, solar radiation is a direct proxy for behind-the-meter (residential) solar production. High radiation reduces a household's net demand on the grid, while sudden drops (due to cloud cover) can cause sharp spikes in residential grid dependency.
- **Precipitation & Cloud Cover:** Beyond their cooling effect, these impact lifestyle behaviors. Rainy or overcast days increase the use of indoor lighting and electronics, and often result in residents staying home longer, consistently elevating the residential baseline load.

### Technical Rationale: Degree Days (CDD & HDD)
We derive **Cooling Degree Days (CDD)** and **Heating Degree Days (HDD)** from the mean temperature.
- **CDD:** `max(0, Temperature - 18°C)`
- **HDD:** `max(0, 18°C - Temperature)`
These features provide a non-linear representation of energy demand relative to human comfort thresholds, making it easier for the model to learn seasonal variations.

---

## 2. Grid Load Features
**Source:** [CAISO Historical EMS Hourly Load](https://www.caiso.com/library/historical-ems-hourly-load)

Grid-level demand serves as a high-fidelity proxy for general energy activity patterns.

### Acquisition Method
This data is retrieved via **Web Scraping**. The script (`get_load.py`) uses `BeautifulSoup` to crawl the **CAISO Historical Library**. It identifies and downloads historical EMS hourly load files (Excel or CSV), standardizes column headers across different years, and handles the conversion of legacy `.xls` files into clean CSV format.

### Key Variables
- **CAISO Total Load:** The total system-wide demand in the California Independent System Operator region.

### Technical Rationale
While we predict individual household cost, the system-wide load captures broader economic and social cycles (e.g., weekends vs. weekdays, holidays) that affect energy usage across the state.

---

## 3. Electricity Pricing
**Source:** [EIA API (Retail Sales)](https://api.eia.gov/v2/electricity/retail-sales/data/)

To convert energy consumption into a monetary value, we require accurate retail rate data.

### Acquisition Method
We query the **EIA Retail Sales API v2**. The request specifically targets the `monthly` frequency, filtering for `stateid: CA` and `sectorid: RES` (Residential). This provides a reliable baseline for what consumers are actually billed per kWh in California.

### Key Variables
- **Monthly Retail Price (Cents/kWh):** The average residential rate for California.

### Technical Rationale: Cost Estimation
Since residential electricity prices for most consumers are billed at a flat monthly rate rather than changing hourly, we estimate the **Hourly Cost** to provide a high-frequency target for the model.

We use the following formula:
$$\text{Hourly Cost (USD)} = \left( \frac{\text{Current Hourly Grid Load}}{\text{Mean Monthly Grid Load}} \right) \times \text{Avg. Residential kWh/hour} \times \left( \frac{\text{Monthly Price (Cents)}}{100} \right)$$

This approach accomplishes several goals:
1.  **Load Shaping**: It assumes a household's usage patterns follow the general statewide grid demand (e.g., higher costs during evening peaks).
2.  **Monetary Mapping**: It converts raw energy usage into a dollar value using real historical rate data from the EIA.
3.  **Configurable Baselines**: It scales based on a configurable `RESIDENTIAL_KWH_PER_HOUR` (defaulted to 0.8 kWh), allowing the model to adapt to different household sizes.

---

## 4. Energy Mix (Fuel Type)
**Source:** [EIA API (RTO Fuel Type Data)](https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/)

The composition of the energy grid (the "fuel mix") provides insights into the available supply and operational constraints.

### Acquisition Method
Data is fetched from the **EIA Real-Time Operating Grid API**. We request hourly generation by fuel type for the California balancing authority (`respondent: CISO`). The script (`get_energy_types.py`) pivots the raw long-format API response into a wide format (one column per fuel type) for model compatibility.

### Key Variables
- **Natural Gas, Solar, Wind, Hydro, Nuclear, Coal, etc.**

### Technical Rationale
The fuel mix is a high-fidelity signal for both the **cost** and **environmental impact** of energy:
- **Marginal Cost Economics:** Renewables like Solar and Wind have near-zero marginal costs. When they dominate the grid (usually midday), wholesale prices tend to drop. Conversely, a high percentage of Natural Gas or Imports typically signals more expensive generation is being dispatched to meet high demand.
- **The "Duck Curve" Signal:** In California, the transition from midday solar abundance to evening natural gas ramp-up creates the famous "Duck Curve." This transition is a direct predictor of when the grid enters its most expensive and carbon-intensive "peak" windows.
- **Supply Constraints:** Monitoring the balance of dispatchable fuels (Natural Gas, Nuclear) vs. intermittent ones (Wind, Solar) helps the model understand grid stability and the likelihood of demand-response events or price spikes.

---

## 5. Temporal & Lag Features
**Logic-based Generation**

Energy usage is highly seasonal and exhibits strong "memory" or autocorrelation.

### Key Variables
- **Lags (1, 7, 15, 30 days):** Rolling means and standard deviations for both `Load` and `Cost`.
- **Hour of Day (HE):** Captures daily cycles (peak evening usage vs. overnight lows).

### Technical Rationale
- **1-Day Lag:** Captures immediate persistence (was it expensive yesterday?).
- **7-Day Lag:** Captures weekly seasonality (what happened last Monday?).
- **Rolling Statistics:** Help the model distinguish between a sudden spike and a sustained trend in demand or pricing.

---

## Data Quality & Preprocessing
To ensure model robustness, the following steps are applied to all features:
- **Linear Interpolation:** Fills gaps in time-series data to ensure hourly continuity.
- **Clipping:** Prevents outliers (e.g., negative humidity or impossible radiation values).
- **Forward Filling:** Ensures that if a source API is temporarily unavailable, the model has the most recent valid state to work with.
