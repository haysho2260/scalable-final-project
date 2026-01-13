# Price Estimation Methodology

Since direct hourly or daily residential pricing data is typically unavailable (as household billing is usually based on flat monthly rates), this project estimates a high-frequency target called `Estimated_Hourly_Cost_USD`.

## Overview of Estimation Logic

The estimation uses **statewide grid demand (CAISO Load)** as a proxy for individual household behavior. The core assumption is that a household's usage patterns roughly follow broader system-wide demand—meaning energy usage (and thus cost) is higher during evening grid peaks than in the middle of the night.

By taking a typical monthly energy spend and "shaping" it according to the grid's hourly load, we create a synthetic but realistic hourly cost variable for the model to learn from.

### Formula
$$\text{Hourly Cost (USD)} = \left( \frac{\text{Current Hourly Grid Load}}{\text{Mean Monthly Grid Load}} \right) \times \text{Avg. Residential kWh/hour} \times \left( \frac{\text{Monthly Price (Cents)}}{100} \right)$$

---

## Variable List

The following variables are used to generate the estimated price and serve as features for the predictive model:

### 1. Core Estimation Variables
*   **`CAISO Total`**: The hourly system-wide demand in California (Source: CAISO).
*   **Mean Monthly Grid Load**: The average CAISO load for the specific month in question.
*   **Monthly Price (Cents/kWh)**: The average residential rate for California (Source: EIA API).
*   **`RESIDENTIAL_KWH_PER_HOUR`**: A configurable baseline for a typical household (standardized at **0.8 kWh**, roughly 700 kWh/month).

### 2. Derived Feature Variables
*   **`Estimated_Hourly_Cost_USD`**: The target variable calculated by the formula above.
*   **Daily Stats**: `daily_mean_cost` and `daily_std_cost`.
*   **Lag Features**: Rolling means and standard deviations for cost at **1, 7, 15, and 30-day** intervals.
*   **Temporal Features**: `HE` (Hour Ending), `hour` (0-23), `dayofweek`, and `month`.
*   **Weather Features**: `Temperature`, `CDD` (Cooling Degree Days), and `HDD` (Heating Degree Days) from Los Angeles weather station data.
