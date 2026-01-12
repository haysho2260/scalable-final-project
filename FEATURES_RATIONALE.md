# Features Rationale

This document outlines the data features used in the energy cost prediction model, their sources, and the technical reasoning behind their inclusion.

## Overview

The model predicts the `Estimated_Hourly_Cost_USD` for a typical residential household in California. To achieve high accuracy, we incorporate a diverse set of features that capture meteorological conditions, grid-level demand, market pricing, and historical patterns.

---

## 1. Weather Features
**Source:** [Open-Meteo Historical Archive](https://open-meteo.com/)

Weather is a primary driver of residential energy consumption, particularly for heating and cooling.

### Key Variables
- **Temperature (Mean, Max, Min):** Direct impact on HVAC usage.
- **Apparent Temperature:** Captures "feels like" conditions, which often correlate better with human behavior than raw temperature.
- **Dew Point & Humidity:** High humidity increases the cooling load on air conditioning systems.
- **Shortwave Radiation:** Influences both solar gain in buildings and potential local solar generation.
- **Precipitation & Cloud Cover:** Affects outdoor activity and lighting needs.

### Technical Rationale: Degree Days (CDD & HDD)
We derive **Cooling Degree Days (CDD)** and **Heating Degree Days (HDD)** from the mean temperature.
- **CDD:** `max(0, Temperature - 18°C)`
- **HDD:** `max(0, 18°C - Temperature)`
These features provide a non-linear representation of energy demand relative to human comfort thresholds, making it easier for the model to learn seasonal variations.

---

## 2. Grid Load Features
**Source:** [CAISO Historical EMS Hourly Load](https://www.caiso.com/library/historical-ems-hourly-load)

Grid-level demand serves as a high-fidelity proxy for general energy activity patterns.

### Key Variables
- **CAISO Total Load:** The total system-wide demand in the California Independent System Operator region.

### Technical Rationale
While we predict individual household cost, the system-wide load captures broader economic and social cycles (e.g., weekends vs. weekdays, holidays) that affect energy usage across the state.

---

## 3. Electricity Pricing
**Source:** [EIA API (Retail Sales)](https://api.eia.gov/v2/electricity/retail-sales/data/)

To convert energy consumption into a monetary value, we require accurate retail rate data.

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

### Key Variables
- **Natural Gas, Solar, Wind, Hydro, Nuclear, Coal, etc.**

### Technical Rationale
The availability of renewables (Solar/Wind) vs. dispatchable fossil fuels (Natural Gas) can correlate with grid stability and potential future pricing tiers. Including the fuel mix helps the model understand the underlying state of the energy market.

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
