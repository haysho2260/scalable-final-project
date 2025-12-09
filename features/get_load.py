import requests
from bs4 import BeautifulSoup
import os
import urllib.parse
import pandas as pd
import re

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

if __name__ == "__main__":
    get_hourly_load()