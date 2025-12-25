import requests
import pandas as pd
import os
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def fetch_fuel_type_data(start_year, output_dir):
    """
    Fetches hourly fuel type data for California from EIA API and saves monthly CSVs in wide format.
    """
    api_key = os.getenv("EIA_API_KEY")
    if not api_key:
        print("Error: EIA_API_KEY not found in environment variables.")
        return
    api_key = api_key.strip()  # Remove any whitespace/newlines

    url = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
    os.makedirs(output_dir, exist_ok=True)

    current_date = datetime.now()
    start_date = datetime(start_year, 1, 1)

    # Iterate by month to manage pagination and file storage efficiency
    # We go from start_year up to current month

    # Helper to generate month ranges
    iter_date = start_date
    while iter_date <= current_date:
        year = iter_date.year
        month = iter_date.month

        # Next month for loop
        if month == 12:
            next_iter = datetime(year + 1, 1, 1)
        else:
            next_iter = datetime(year, month + 1, 1)

        # Define month start/end for API
        # API strings are typically YYYY-MM-DDTHH
        # We fetch the whole month
        month_start_str = f"{year}-{month:02d}-01T00"

        # End is start of next month (exclusive in logic, but API inclusive/exclusive depends)
        # Safest is to just fetch chunk and filter.
        # But let's check file existence first.

        filename = f"Fuel_Type_{year}_{month:02d}.csv"
        filepath = os.path.join(output_dir, filename)

        is_current_month = (
            year == current_date.year and month == current_date.month)

        if os.path.exists(filepath) and not is_current_month:
            # Skip historical completed months
            # print(f"Skipping {filename} (already exists)")
            iter_date = next_iter
            continue

        print(f"Processing {year}-{month:02d}...")

        # Calculate end date string for this month (start of next month)
        month_end_str = f"{next_iter.year}-{next_iter.month:02d}-01T00"

        # Fetch data for this specific month range
        # Pagination loop might be needed within the month if > 5000 rows?
        # 1 month = 30 * 24 = 720 hours.
        # CAISO has ~15-20 fuel types? 720 * 20 = 14,400 rows.
        # So yes, we WILL hit the 5000 limit per month.
        # We need to page through the month.

        all_rows = []
        offset = 0
        page_length = 5000

        while True:
            params = {
                "api_key": api_key,
                "frequency": "hourly",
                "data": ["value"],
                "facets": {
                    # Accept both CISO and CAL respondents, restricted to California BA name
                    "respondent": ["CISO", "CAL"],
                    "respondent-name": ["California"],
                },
                "start": month_start_str,
                "end": month_end_str,
                "sort": [{"column": "period", "direction": "asc"}],
                "offset": offset,
                "length": page_length
            }

            headers = {
                'X-Params': json.dumps(params)
            }

            try:
                # Need to use params trick for API Key if headers fail, but header is standard v2
                # Passing api_key in query params usually safer
                req_params = {'api_key': api_key}
                response = requests.get(
                    url, params=req_params, headers=headers)
                response.raise_for_status()
                data = response.json()

                rows = []
                if 'response' in data and 'data' in data['response']:
                    rows = data['response']['data']
                elif 'data' in data:
                    rows = data['data']

                if not rows:
                    break

                all_rows.extend(rows)

                if len(rows) < page_length:
                    break

                offset += page_length

            except Exception as e:
                print(f"  Error fetching page: {e}")

                # If 'CAL' is wrong/empty, we might error out or get empty.
                break

        if not all_rows:
            print(
                f"  No data found for {year}-{month:02d}. (Check respondent/facets or EIA coverage for this period.)")
        else:
            # Process DataFrame
            df = pd.DataFrame(all_rows)

            # Pivot
            # Columns needed: period, type-name, value
            if 'period' in df.columns and 'type-name' in df.columns and 'value' in df.columns:
                # Ensure value is float
                df['value'] = pd.to_numeric(
                    df['value'], errors='coerce').fillna(0)

                # Pivot: Index=period, Columns=type-name, Values=value
                df_pivot = df.pivot_table(
                    index='period', columns='type-name', values='value', aggfunc='sum').reset_index()

                # --- Cleaning Start ---
                # 1. Enforce Hourly Continuity
                if not df_pivot.empty:
                    # Normalize period to datetime, accepting both "YYYY-MM-DDTHH" and "YYYY-MM-DD HH:MM:SS"
                    df_pivot['period'] = pd.to_datetime(
                        df_pivot['period'], errors='coerce', utc=False, format="ISO8601")
                    missing = df_pivot['period'].isna()
                    if missing.any():
                        df_pivot.loc[missing, 'period'] = pd.to_datetime(
                            df_pivot.loc[missing, 'period'], errors='coerce')
                    df_pivot = df_pivot.dropna(subset=['period'])
                    df_pivot = df_pivot.sort_values('period')

                    # Create full range for this month (or the range covered by data)
                    # We want strictly hourly freq
                    min_dt = df_pivot['period'].min()
                    max_dt = df_pivot['period'].max()
                    full_idx = pd.date_range(
                        start=min_dt, end=max_dt, freq='h')

                    df_pivot = df_pivot.set_index('period').reindex(full_idx)
                    df_pivot.index.name = 'period'
                    df_pivot = df_pivot.reset_index()

                    # 2. Interpolate
                    # Interpolate numeric columns (fuel types)
                    numeric_cols = df_pivot.select_dtypes(
                        include=['float64', 'int64']).columns
                    df_pivot[numeric_cols] = df_pivot[numeric_cols].interpolate(
                        method='linear', limit_direction='both')
                # --- Cleaning End ---

                # Fill NaNs with 0
                df_pivot = df_pivot.fillna(0)

                # Standardize period format to ISO hour string for downstream parsing
                df_pivot['period'] = pd.to_datetime(
                    df_pivot['period'], errors='coerce')
                df_pivot = df_pivot.dropna(subset=['period'])
                df_pivot['period'] = df_pivot['period'].dt.strftime(
                    "%Y-%m-%dT%H")

                # Save
                df_pivot.to_csv(filepath, index=False)
                print(f"  -> Saved {filename} ({len(df_pivot)} hours)")
            else:
                print(f"  Unexpected columns: {df.columns}")

        iter_date = next_iter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_year", type=int, default=2019,
                        help="Start year for fetching data")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'energy_types')

    fetch_fuel_type_data(args.start_year, output_dir)
