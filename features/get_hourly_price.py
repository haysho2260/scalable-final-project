import requests
import pandas as pd
import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Typical CA residential consumption: ~700 kWh/month = ~0.8 kWh/hour average
# Can be configured via environment variable for different household sizes
RESIDENTIAL_KWH_PER_HOUR = float(os.getenv('RESIDENTIAL_KWH_PER_HOUR', '0.8'))


def fetch_eia_prices():
    """
    Fetches monthly retail electricity prices from EIA API.
    Returns a DataFrame with 'Year', 'Month', 'Price' (cents/kWh).
    """
    api_key = os.getenv("EIA_API_KEY")
    if not api_key:
        print("Error: EIA_API_KEY not found in environment variables.")
        return None
    api_key = api_key.strip()  # Remove any whitespace/newlines

    url = "https://api.eia.gov/v2/electricity/retail-sales/data/"
    params = {
        "api_key": api_key,
        "frequency": "monthly",
        "data": ["price"],
        "facets": {
            "stateid": ["CA"],
            "sectorid": ["RES"]  # Residential
        },
        "start": None,
        "end": None,
        "sort": [{"column": "period", "direction": "desc"}],
        "offset": 0,
        "length": 5000
    }

    headers = {
        'X-Params': json.dumps(params)
    }

    print(f"Fetching data from EIA API with headers...")
    try:
        request_params = {
            'api_key': api_key
        }

        response = requests.get(url, params=request_params, headers=headers)
        response.raise_for_status()
        data = response.json()

        if 'response' in data and 'data' in data['response']:
            rows = data['response']['data']
        elif 'data' in data:
            rows = data['data']
        else:
            print("Unknown API response structure:", data.keys())
            return None

        df = pd.DataFrame(rows)

        # No need to filter stateid/sectorid locally if facets worked, but safety check doesn't hurt.
        if 'stateid' in df.columns:
            df = df[df['stateid'] == 'CA']

        # Parse period (YYYY-MM)
        df['period'] = pd.to_datetime(df['period'])
        df['Year'] = df['period'].dt.year
        df['Month'] = df['period'].dt.month

        # Rename price column and ensure float
        df = df.rename(columns={'price': 'Monthly_Price_Cents_per_kWh'})
        df['Monthly_Price_Cents_per_kWh'] = pd.to_numeric(
            df['Monthly_Price_Cents_per_kWh'], errors='coerce')

        df = df.sort_values(['Year', 'Month'])

        # 1. Robust Price: Forward Fill
        # If a month is missing, use the previous month's price.
        # First, ensure we have a continuous monthly index
        min_date = df['period'].min()
        max_date = df['period'].max()
        full_idx = pd.date_range(
            start=min_date, end=max_date, freq='MS')  # Month Start

        df = df.set_index('period').reindex(full_idx)
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        df['Monthly_Price_Cents_per_kWh'] = df['Monthly_Price_Cents_per_kWh'].ffill()

        return df[['Year', 'Month', 'Monthly_Price_Cents_per_kWh']].reset_index(drop=True)

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def process_hourly_prices():
    """
    Loads lag_load files, merges with price data, computes cost, saves to hourly_price.
    Optimized to be incremental: checks if source file has newer data (latest hour) than target.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(base_dir, 'lag_load')
    target_dir = os.path.join(base_dir, 'hourly_price')
    os.makedirs(target_dir, exist_ok=True)

    source_files = [f for f in os.listdir(source_dir) if f.startswith(
        "CAISO_Load_") and f.endswith(".csv")]

    # Identify files that need processing based on content check (Latest Hour)
    files_to_process = []

    for f in source_files:
        parts = f.replace("CAISO_Load_", "").replace(".csv", "").split("_")
        year, month = int(parts[0]), int(parts[1])
        out_name = f"CAISO_Price_{year}_{int(month):02d}.csv"
        out_path = os.path.join(target_dir, out_name)

        needs_update = False

        if not os.path.exists(out_path):
            needs_update = True
        else:
            try:
                # Check Latest Hour in Source vs Target
                # We only need the last few lines, but reading full file is safer for sorting
                # Optimization: Read only tail if large, but files are monthly and smallish (<10MB)
                src_df = pd.read_csv(os.path.join(source_dir, f))
                tgt_df = pd.read_csv(out_path)

                # Check for Date/HE columns.
                # "Date" and "HE" (Hour Ending). HE 24 -> next day 00:00 technically, but just sorting by Date, HE works.
                date_col_src = next(
                    (c for c in src_df.columns if 'date' in c.lower()), 'Date')
                date_col_tgt = next(
                    (c for c in tgt_df.columns if 'date' in c.lower()), 'Date')

                if date_col_src in src_df.columns and date_col_tgt in tgt_df.columns and 'HE' in src_df.columns and 'HE' in tgt_df.columns:
                    src_df[date_col_src] = pd.to_datetime(
                        src_df[date_col_src], errors='coerce')
                    tgt_df[date_col_tgt] = pd.to_datetime(
                        tgt_df[date_col_tgt], errors='coerce')

                    # Drop NA
                    src_df = src_df.dropna(subset=[date_col_src, 'HE'])
                    tgt_df = tgt_df.dropna(subset=[date_col_tgt, 'HE'])

                    if not src_df.empty and not tgt_df.empty:
                        # Get max timestamp tuple
                        src_max = src_df.sort_values(
                            [date_col_src, 'HE']).iloc[-1]
                        tgt_max = tgt_df.sort_values(
                            [date_col_tgt, 'HE']).iloc[-1]

                        src_ts = (src_max[date_col_src], src_max['HE'])
                        tgt_ts = (tgt_max[date_col_tgt], tgt_max['HE'])

                        if src_ts > tgt_ts:
                            needs_update = True
                            print(
                                f"Update needed for {f}: Source latest {src_ts} > Target latest {tgt_ts}")
                    elif not src_df.empty and tgt_df.empty:
                        needs_update = True  # Target broken
                else:
                    # Columns missing, force update
                    needs_update = True
            except Exception as e:
                print(f"Error checking {f}, forcing update: {e}")
                needs_update = True

        if needs_update:
            files_to_process.append(f)

    if not files_to_process:
        print("All hourly price files up to date (latest hour check passed). Skipping API fetch.")
    else:
        print(
            f"Found {len(files_to_process)} files needing update. Fetching prices...")

        # 1. Fetch Prices (Once)
        prices_df = fetch_eia_prices()
        if prices_df is None or prices_df.empty:
            print("No price data retrieved. Aborting.")
            return

        # 2. Process Files
        for f in files_to_process:
            load_path = os.path.join(source_dir, f)
            try:
                df = pd.read_csv(load_path)
                # Normalize Date/HE for downstream consistency
                date_col = next(
                    (c for c in df.columns if 'date' in c.lower()), 'Date')
                if date_col in df.columns:
                    df[date_col] = pd.to_datetime(
                        df[date_col], errors='coerce').dt.date
                if 'HE' in df.columns:
                    df['HE'] = pd.to_numeric(df['HE'], errors='coerce')
                df = df.dropna(
                    subset=[date_col, 'HE']) if date_col in df.columns and 'HE' in df.columns else df

                parts = f.replace("CAISO_Load_", "").replace(
                    ".csv", "").split("_")
                year = int(parts[0])
                month = int(parts[1])

                price_row = prices_df[(prices_df['Year'] == year) & (
                    prices_df['Month'] == month)]

                if not price_row.empty:
                    price = price_row.iloc[0]['Monthly_Price_Cents_per_kWh']
                    df['Monthly_Price_Cents_per_kWh'] = price

                    # Calculate residential household cost (not system-wide)
                    # Using configurable kWh/hour consumption rate
                    df['Estimated_Hourly_Cost_USD'] = RESIDENTIAL_KWH_PER_HOUR * (price / 100.0)
                else:
                    print(f"  Warning: No price found for {year}-{month}")
                    df['Monthly_Price_Cents_per_kWh'] = pd.NA
                    df['Estimated_Hourly_Cost_USD'] = pd.NA

                out_name = f"CAISO_Price_{year}_{int(month):02d}.csv"
                out_path = os.path.join(target_dir, out_name)
                # Write Date back as ISO string for consistency
                if date_col in df.columns:
                    df[date_col] = pd.to_datetime(
                        df[date_col]).dt.strftime("%Y-%m-%d")
                df.to_csv(out_path, index=False)
                print(f"  -> Saved {out_name}")

            except Exception as e:
                print(f"Error processing {f}: {e}")


def process_lag_prices():
    """
    Generates lag features for Cost data.
    Source: features/hourly_price (contains Estimated_Hourly_Cost_USD)
    Target: features/lag_prices
    """
    print("Processing lag prices...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(base_dir, 'hourly_price')
    target_dir = os.path.join(base_dir, 'lag_prices')
    os.makedirs(target_dir, exist_ok=True)

    if not os.path.exists(source_dir):
        print(
            "Source directory hourly_price does not exist. Run process_hourly_prices first.")
        return

    source_files = [f for f in os.listdir(source_dir) if f.startswith(
        "CAISO_Price_") and f.endswith(".csv")]
    target_files = set(os.listdir(target_dir))

    # Identify missing or outdated files in lag_prices
    # We apply the same Content Check strategy for robust incremental updates
    missing_files = []

    # Pre-read metadata to sorting
    file_metadata = []
    for f in source_files:
        try:
            parts = f.replace("CAISO_Price_", "").replace(
                ".csv", "").split("_")
            year, month = int(parts[0]), int(parts[1])
            file_metadata.append({'file': f, 'year': year, 'month': month})
        except:
            continue

    file_metadata.sort(key=lambda x: (x['year'], x['month']))

    for meta in file_metadata:
        f = meta['file']
        if f not in target_files:
            missing_files.append(f)
        else:
            # Check content age
            # Similar logic: Source (hourly_price) vs Target (lag_prices)
            try:
                # Check logic: if Source Last TS > Target Last TS -> Needs update
                # Using modification time is faster for this step if we trust process_hourly_prices updated the file recently
                # But sticking to content check is safest.

                src_path = os.path.join(source_dir, f)
                tgt_path = os.path.join(target_dir, f)  # Same name

                src_mtime = os.path.getmtime(src_path)
                tgt_mtime = os.path.getmtime(tgt_path)

                if src_mtime > tgt_mtime:
                    missing_files.append(f)
            except:
                missing_files.append(f)

    if not missing_files:
        print("All lag price files up to date.")
        return

    print(f"Found {len(missing_files)} files to update/create in lag_prices.")

    # Resolve dependencies for context
    # Need to process each missing file. But for lags, we need context.
    # Approach: Iterate missing files. For each, load context (prev 2 months).

    for meta in file_metadata:
        f = meta['file']
        if f not in missing_files:
            continue

        print(f"Generating features for {f}...")

        # Determine context files
        # Find index in sorted metadata
        current_idx = next(i for i, m in enumerate(
            file_metadata) if m['file'] == f)
        start_idx = max(0, current_idx - 2)
        # Include current
        context_files = file_metadata[start_idx: current_idx + 1]

        df_list = []
        for cf in context_files:
            path = os.path.join(source_dir, cf['file'])
            try:
                df = pd.read_csv(path)
                df_list.append(df)
            except Exception as e:
                print(f"Error reading {path}: {e}")

        if not df_list:
            continue

        full_df = pd.concat(df_list, ignore_index=True)

        # Prepare Data
        date_col = next(
            (c for c in full_df.columns if 'date' in c.lower()), None)
        if not date_col:
            continue

        full_df[date_col] = pd.to_datetime(full_df[date_col], errors='coerce')
        full_df = full_df.dropna(subset=[date_col])
        full_df = full_df.sort_values(date_col)
        full_df['Daily_Date'] = full_df[date_col].dt.date

        target_col = 'Estimated_Hourly_Cost_USD'
        if target_col not in full_df.columns:
            continue

        # Compute Daily Stats
        daily_stats = full_df.groupby('Daily_Date')[target_col].agg([
            'mean', 'std']).reset_index()
        daily_stats.columns = ['Daily_Date',
                               'daily_mean_cost', 'daily_std_cost']
        daily_stats['Daily_Date'] = pd.to_datetime(daily_stats['Daily_Date'])

        # Lags
        lags = [1, 7, 15, 30]
        for lag in lags:
            daily_stats[f'daily_mean_cost_lag_{lag}'] = daily_stats['daily_mean_cost'].shift(
                lag)
            daily_stats[f'daily_std_cost_lag_{lag}'] = daily_stats['daily_std_cost'].shift(
                lag)

        full_df['Daily_Date'] = pd.to_datetime(full_df['Daily_Date'])

        # Clean existing lags
        cols_to_drop = [
            c for c in full_df.columns if 'daily_mean_cost' in c or 'daily_std_cost' in c]
        if cols_to_drop:
            full_df = full_df.drop(columns=cols_to_drop)

        # Merge
        full_df = pd.merge(full_df, daily_stats, on='Daily_Date', how='left')

        # Filter back to only the target month (Current File)
        # Using temp year/month
        full_df['TempYear'] = full_df[date_col].dt.year
        full_df['TempMonth'] = full_df[date_col].dt.month

        target_year = meta['year']
        target_month = meta['month']

        final_df = full_df[(full_df['TempYear'] == target_year) & (
            full_df['TempMonth'] == target_month)].copy()

        # Drop temp columns
        drop_cols = ['Daily_Date', 'TempYear', 'TempMonth',
                     'daily_mean_cost', 'daily_std_cost']
        final_df = final_df.drop(columns=drop_cols, errors='ignore')

        # --- Cleaning Start ---
        # 2. Output Hourly Continuity (in features/lag_prices)
        if not final_df.empty:
            date_col_final = next(
                (c for c in final_df.columns if 'date' in c.lower()), None)
            if date_col_final:
                final_df = final_df.sort_values(date_col_final)
                # Drop duplicates on date_col_final before reindexing to avoid duplicate label error
                final_df = final_df.drop_duplicates(
                    subset=[date_col_final], keep='last')

                # Create strict hourly index for target month
                min_dt = datetime(target_year, target_month, 1)
                if target_month == 12:
                    next_month = datetime(target_year + 1, 1, 1)
                else:
                    next_month = datetime(target_year, target_month + 1, 1)
                # max_dt is end of month, exclusive of next month start
                # e.g. 2019-01-01 00:00 to 2019-01-31 23:00
                full_idx = pd.date_range(
                    start=min_dt, end=next_month, freq='h', inclusive='left')

                final_df = final_df.set_index(date_col_final).reindex(full_idx)
                final_df.index.name = date_col_final
                final_df = final_df.reset_index()

                # Normalize Date column to YYYY-MM-DD format and set HE from timestamp
                timestamps = pd.to_datetime(final_df[date_col_final])
                final_df[date_col_final] = timestamps.dt.strftime("%Y-%m-%d")
                # Set HE from the hour component (HE is 1-indexed: 1-24)
                if 'HE' not in final_df.columns or final_df['HE'].isna().all() or (final_df['HE'] == 0).all():
                    final_df['HE'] = timestamps.dt.hour + \
                        1  # HE is 1-indexed (1-24)

                # Interpolate numeric
                numeric_cols = final_df.select_dtypes(
                    include=['float64', 'int64']).columns
                final_df[numeric_cols] = final_df[numeric_cols].interpolate(
                    method='linear', limit_direction='both')

                # Fill remaining edge NaNs with 0
                final_df[numeric_cols] = final_df[numeric_cols].fillna(0)
        # --- Cleaning End ---

        # Save
        out_path = os.path.join(target_dir, f)
        final_df.to_csv(out_path, index=False)
        print(f"  -> Created {f} in lag_prices/")


if __name__ == "__main__":
    process_hourly_prices()
    process_lag_prices()
