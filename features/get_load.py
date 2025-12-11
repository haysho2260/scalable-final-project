import requests
from bs4 import BeautifulSoup
import os
import urllib.parse
import pandas as pd
import re
from datetime import datetime

def get_file_urls(base_url):
    """
    Scrapes the CAISO historical EMS hourly load page for file URLs.

    Args:
        base_url (str): The URL of the page to scrape.

    Returns:
        list: A list of absolute URLs for the data files found on the page.
    """
    print(f"Scraping data from {base_url}...")
    try:
        response = requests.get(base_url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching {base_url}: {e}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find matching tags based on known structure (td with class "doc-lib-name title")
    tds = soup.find_all('td', class_='doc-lib-name title')
    
    file_urls = []
    for td in tds:
        a_tag = td.find('a')
        if a_tag and a_tag.get('href'):
            file_url = a_tag.get('href')
            # Handle relative URLs
            if not file_url.startswith('http'):
                 file_url = urllib.parse.urljoin(base_url, file_url)
            file_urls.append(file_url)
            
    print(f"Found {len(file_urls)} files.")
    return file_urls

def parse_filename(file_url):
    """
    Parses the filename from a URL and determines a standardized name 
    based on the year and month found within the original filename.

    Args:
        file_url (str): The URL of the file.

    Returns:
        tuple: (original_filename, new_base_name)
            original_filename (str): The filename as it appears in the URL.
            new_base_name (str): The standardized base name (e.g. "CAISO_Load_2023_01").
    """
    original_filename = os.path.basename(urllib.parse.urlparse(file_url).path)
    lower_name = original_filename.lower()
    
    # Month mapping for detection
    month_map = {
        'january': '01', 'february': '02', 'march': '03', 'april': '04',
        'may': '05', 'june': '06', 'july': '07', 'august': '08',
        'september': '09', 'october': '10', 'november': '11', 'december': '12'
    }

    # Extract Year
    year_match = re.search(r'20\d{2}', lower_name)
    year = year_match.group(0) if year_match else None
    
    # Extract Month
    month = None
    for m_name, m_num in month_map.items():
        if m_name in lower_name:
            month = m_num
            break
    
    # Construct new base name
    new_basename = "CAISO_Load"
    if year:
        new_basename += f"_{year}"
    if month:
        new_basename += f"_{month}"
    
    # Fallback if no year/month found
    if not year and not month:
        new_basename = os.path.splitext(original_filename)[0]
        
    return original_filename, new_basename

def download_and_process_file(file_url, target_dir):
    """
    Downloads a single file, processes it (converts Excel to CSV), and saves it.

    Args:
        file_url (str): The URL of the file to download.
        target_dir (str): The local directory to save files to.
    """
    original_filename, new_basename = parse_filename(file_url)
    
    # Determine expected output filename to check existence
    lower_name = original_filename.lower()
    expected_output_path = None
    
    if lower_name.endswith(('.xls', '.xlsx')):
        csv_name = f"{new_basename}.csv"
        expected_output_path = os.path.join(target_dir, csv_name)
    else:
        ext = os.path.splitext(original_filename)[1]
        file_name = new_basename + ext
        expected_output_path = os.path.join(target_dir, file_name)
    
    # Check if file exists
    if expected_output_path and os.path.exists(expected_output_path):
        print(f"Skipping {original_filename} (already exists as {os.path.basename(expected_output_path)})")
        return

    print(f"Processing {original_filename} from {file_url}...")

    # Determine if this is likely a yearly file (Year set, Month Not set)
    is_yearly_file = False
    
    # We parse the Year and Month again to be sure (since parse_filename returns new_basename)
    temp_original, temp_basename = parse_filename(file_url) # Re-using logic implicitly or better to refactor parse_filename to return metadata
    # Quick re-parse for metadata (Year/Month)
    lower_original = original_filename.lower()
    year_match = re.search(r'20\d{2}', lower_original)
    has_year = bool(year_match)
    has_month = any(m in lower_original for m in ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'])
    
    if has_year and not has_month:
        is_yearly_file = True

    # Determine expected output filename(s) to check existence
    existing_file_check = None
    
    if is_yearly_file and has_year:
        # For yearly files, we ideally check if ALL monthly files exist. 
        # But for simplicity, let's check one or two, or just check if the year file was already processed.
        # Actually simplest is: if we are splitting, we produce CAISO_Load_YYYY_MM.csv
        # So let's check if CAISO_Load_YYYY_01.csv exists?
        year_str = year_match.group(0)
        check_path = os.path.join(target_dir, f"CAISO_Load_{year_str}_01.csv")
        if os.path.exists(check_path):
             existing_file_check = check_path
    else:
        # Monthly file check
        if lower_name.endswith(('.xls', '.xlsx')):
            csv_name = f"{new_basename}.csv"
            existing_file_check = os.path.join(target_dir, csv_name)
        else:
            ext = os.path.splitext(original_filename)[1]
            file_name = new_basename + ext
            existing_file_check = os.path.join(target_dir, file_name)
    
    # Check if file exists
    if existing_file_check and os.path.exists(existing_file_check):
        print(f"Skipping {original_filename} (seems already processed as {os.path.basename(existing_file_check)})")
        return

    print(f"Processing {original_filename} from {file_url}...")

    try:
        file_response = requests.get(file_url)
        file_response.raise_for_status()
        
        # Check if it is an Excel file for conversion
        if lower_name.endswith(('.xls', '.xlsx')):
             # Save temporarily
            temp_path = os.path.join(target_dir, original_filename)
            with open(temp_path, 'wb') as f:
                f.write(file_response.content)
            
            try:
                df = pd.read_excel(temp_path)
                
                # If yearly file, split it
                if is_yearly_file:
                    print(f"  -> Detected yearly file. Splitting into months...")
                    # Identify date column (first column containing 'Date' or just first column?)
                    # Usually it's 'Date' or 'OPR_DT' or similar. 
                    # Let's look for a column that parses to datetime
                    date_col = None
                    for col in df.columns:
                        if 'date' in str(col).lower():
                            date_col = col
                            break
                    
                    if not date_col and len(df.columns) > 0:
                         # Fallback to first column
                         date_col = df.columns[0]
                    
                    if date_col:
                        df[date_col] = pd.to_datetime(df[date_col])
                        df['TempYear'] = df[date_col].dt.year
                        df['TempMonth'] = df[date_col].dt.month
                        
                        unique_months = df['TempMonth'].unique()
                        for m in unique_months:
                            # Filter
                            month_df = df[df['TempMonth'] == m].copy()
                            # Drop temp cols
                            month_df = month_df.drop(columns=['TempYear', 'TempMonth'])
                            
                            # Construct filename
                            # Use the Year from the row, or the file year? 
                            # Safe to use the file year if available, or data year.
                            # Let's use data year to be accurate.
                            data_year = month_df[date_col].dt.year.iloc[0]
                            
                            out_name = f"CAISO_Load_{data_year}_{int(m):02d}.csv"
                            out_path = os.path.join(target_dir, out_name)
                            
                            month_df.to_csv(out_path, index=False)
                            print(f"    -> Saved {out_name}")
                    else:
                        print("    -> Could not identify Date column. Saving as yearly CSV.")
                        csv_name = f"{new_basename}.csv"
                        csv_path = os.path.join(target_dir, csv_name)
                        df.to_csv(csv_path, index=False)
                        
                else:
                    # Normal monthly file
                    csv_name = f"{new_basename}.csv"
                    csv_path = os.path.join(target_dir, csv_name)
                    df.to_csv(csv_path, index=False)
                    print(f"  -> Converted to CSV: {os.path.basename(csv_path)}")

                # Remove temporary excel file
                os.remove(temp_path)
            except Exception as conv_e:
                 print(f"  -> Failed to convert {original_filename}: {conv_e}")

        else:
            # Not an Excel file, save as is (renamed)
            ext = os.path.splitext(original_filename)[1]
            file_name = new_basename + ext
            file_path = os.path.join(target_dir, file_name)
            
            with open(file_path, 'wb') as f:
                f.write(file_response.content)
            print(f"  -> Saved: {os.path.basename(file_path)}")

    except Exception as e:
        print(f"  -> Failed to download/process {original_filename}: {e}")

def process_lag_features(source_dir):
    """
    Loads monthly CSVs from source_dir, checks for missing files in features/lag_load,
    loads necessary context (previous months), calculates lags, and saves to features/lag_load.
    """
    print("Processing lag features...")
    
    target_dir = os.path.join(os.path.dirname(source_dir), 'lag_load')
    os.makedirs(target_dir, exist_ok=True)
    
    # 1. Identify source files and missing target files
    source_files = [f for f in os.listdir(source_dir) if f.startswith("CAISO_Load_") and f.endswith(".csv")]
    target_files = set(os.listdir(target_dir))
    
    missing_files = [f for f in source_files if f not in target_files]
    
    if not missing_files:
        print("All lag feature files already exist. Skipping.")
        return

    print(f"Found {len(missing_files)} missing lag files: {missing_files[:5]}...")
    
    # 2. Determine required context
    # We need to load missing months + enough previous data (context) to calculate max lag (30 days).
    # Simplest approach: Identify the earliest missing month. Load everything from (Earliest Month - 2 Months) onwards.
    # Why 2 months? To be safe for 30 day lag calc.
    
    # Parse filenames to get (Year, Month)
    file_metadata = []
    for f in source_files:
        # Expected format: CAISO_Load_YYYY_MM.csv
        try:
            parts = f.replace("CAISO_Load_", "").replace(".csv", "").split("_")
            year = int(parts[0])
            month = int(parts[1])
            file_metadata.append({'file': f, 'year': year, 'month': month})
        except:
            continue
            
    if not file_metadata:
        return
        
    # Sort by time
    file_metadata.sort(key=lambda x: (x['year'], x['month']))
    
    # Find index of earliest missing file
    earliest_missing_idx = -1
    for i, meta in enumerate(file_metadata):
        if meta['file'] in missing_files:
            earliest_missing_idx = i
            break
            
    if earliest_missing_idx == -1:
        return # Should capture above, but safety check.
    
    # Context start index: max(0, earliest_missing_idx - 2)
    # We load 2 files back to ensure we have >30 days of history.
    context_start_idx = max(0, earliest_missing_idx - 2)
    files_to_load = file_metadata[context_start_idx:]
    
    print(f"Loading {len(files_to_load)} files (context + target) starting from {files_to_load[0]['file']}...")

    df_list = []
    for meta in files_to_load:
        f_path = os.path.join(source_dir, meta['file'])
        try:
            df = pd.read_csv(f_path)
            # Ensure Date parsing here to be safe
            # We assume source files might or might not have lags already if we ran previous version. 
            # But we are re-calculating or calculating fresh.
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {f_path}: {e}")
            
    if not df_list:
        return

    full_df = pd.concat(df_list, ignore_index=True)
    
    # 3. Prepare Date column (Robust parsing)
    date_col = None
    for col in full_df.columns:
        if 'date' in str(col).lower():
            date_col = col
            break
            
    if not date_col:
         # Fallback
         if 'Date' in full_df.columns: date_col = 'Date'
         elif 'OPR_DT' in full_df.columns: date_col = 'OPR_DT'
    
    if not date_col:
        print("Could not find Date column.")
        return

    full_df[date_col] = pd.to_datetime(full_df[date_col], errors='coerce')
    full_df = full_df.dropna(subset=[date_col])
    full_df = full_df.sort_values(date_col)
    
    full_df['Daily_Date'] = full_df[date_col].dt.date
    
    # 4. Compute Daily Stats
    load_col = 'CAISO Total'
    if load_col not in full_df.columns:
         print(f"Column '{load_col}' not found.")
         return

    # Group by Daily Date
    daily_stats = full_df.groupby('Daily_Date')[load_col].agg(['mean', 'std']).reset_index()
    daily_stats.columns = ['Daily_Date', 'daily_mean_load', 'daily_std_load']
    daily_stats['Daily_Date'] = pd.to_datetime(daily_stats['Daily_Date'])
    
    # 5. Create Lags
    lags = [1, 7, 15, 30]
    for lag in lags:
        daily_stats[f'daily_mean_load_lag_{lag}'] = daily_stats['daily_mean_load'].shift(lag)
        daily_stats[f'daily_std_load_lag_{lag}'] = daily_stats['daily_std_load'].shift(lag)
        
    full_df['Daily_Date'] = pd.to_datetime(full_df['Daily_Date'])
    
    # Drop existing lag columns if they exist to avoid _x, _y duplicates
    cols_to_drop = [c for c in full_df.columns if 'daily_mean_load' in c or 'daily_std_load' in c]
    if cols_to_drop:
        full_df = full_df.drop(columns=cols_to_drop)

    full_df = pd.merge(full_df, daily_stats, on='Daily_Date', how='left')
    
    # 6. Save relevant files
    # Only save the files that were in 'missing_files'
    
    full_df['TempYear'] = full_df[date_col].dt.year
    full_df['TempMonth'] = full_df[date_col].dt.month
    
    groups = full_df.groupby(['TempYear', 'TempMonth'])
    
    print("Saving new lag feature files...")
    for (year, month), group in groups:
        out_name = f"CAISO_Load_{year}_{int(month):02d}.csv"
        
        # Only save if it matches a missing file
        if out_name in missing_files:
            out_path = os.path.join(target_dir, out_name)
            
            # Drop temp columns
            cols_to_drop = ['Daily_Date', 'TempYear', 'TempMonth', 'daily_mean_load', 'daily_std_load']
            
            # Also drop garbage columns identified by user (Unnamed, HR, CAISO, Spring DST notes)
            # Use simple string matching for robustness
            garbage_patterns = ['unnamed', 'hr', 'caiso', 'spring dst'] 
            for c in group.columns:
                c_lower = str(c).lower()
                if any(p in c_lower for p in garbage_patterns) and c != 'CAISO Total': # Keep CAISO Total!
                     # Careful: matches 'HR' in 'CHR'? No, 'hr' in 'hr' yes.
                     # 'CAISO' matches 'CAISO Total'. Correct logic needed.
                     # Unnamed is safe.
                     # HR might be a valid column? Usually headers are PGE, SCE etc. HR might be 'Hour'?
                     # But raw data uses 'HE' (Hour Ending). 'HR' is likely extra header.
                     # 'CAISO' matches 'CAISO Total'.
                     
                     is_garbage = False
                     if 'unnamed' in c_lower: is_garbage = True
                     elif c in ['HR', 'CAISO']: is_garbage = True # Exact match for these likely garbage headers
                     elif 'spring dst' in c_lower: is_garbage = True
                     
                     elif 'spring dst' in c_lower: is_garbage = True
                     
                     if is_garbage:
                         cols_to_drop.append(c)

            # --- Cleaning Start ---
            # 1. Enforce Hourly Continuity
            date_col_final = next((c for c in group.columns if 'date' in str(c).lower()), None)
            
            # Need to ensure date_col_final is actual datetime
            if date_col_final:
                # Create strict hourly index for target month
                min_dt = datetime(year, int(month), 1)
                if int(month) == 12:
                    next_month = datetime(year + 1, 1, 1)
                else:
                    next_month = datetime(year, int(month) + 1, 1)
                
                # Full hourly range
                full_idx = pd.date_range(start=min_dt, end=next_month, freq='h', inclusive='left')
                
                # Reindex
                # Ensure group is sorted and unique on date_col
                clean_df = group.sort_values(date_col_final).drop_duplicates(subset=[date_col_final])
                clean_df = clean_df.set_index(date_col_final).reindex(full_idx)
                clean_df.index.name = date_col_final # Restore name
                clean_df = clean_df.reset_index()
                
                # 2. Interpolate
                numeric_cols = clean_df.select_dtypes(include=['float64', 'int64']).columns
                # Only interpolate columns that are not garbage
                valid_numeric = [c for c in numeric_cols if c not in cols_to_drop]
                
                if valid_numeric:
                    clean_df[valid_numeric] = clean_df[valid_numeric].interpolate(method='linear', limit_direction='both')
                    clean_df[valid_numeric] = clean_df[valid_numeric].ffill().bfill().fillna(0)
                
                # Use clean_df for saving
                save_df = clean_df.drop(columns=cols_to_drop, errors='ignore')
            else:
                 # Fallback if no date col found (unlikely)
                 save_df = group.drop(columns=cols_to_drop, errors='ignore')
            # --- Cleaning End ---
            
            save_df.to_csv(out_path, index=False)
            print(f"  -> Created {out_name} in lag_load/")

def get_hourly_load():
    """
    Main function to scrape, download, and standardizes historical EMS hourly load data.
    
    Saves data to 'features/hourly_load' directory.
    """
    base_url = "https://www.caiso.com/library/historical-ems-hourly-load"
    
    # Ensure target directory exists
    target_dir = os.path.join(os.path.dirname(__file__), 'hourly_load')
    os.makedirs(target_dir, exist_ok=True)
    
    # Get all file URLs
    file_urls = get_file_urls(base_url)
    
    # Process each file
    for url in file_urls:
        download_and_process_file(url, target_dir)
        
    # Process features
    process_lag_features(target_dir)

if __name__ == "__main__":
    get_hourly_load()