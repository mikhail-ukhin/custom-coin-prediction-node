import os
import pathlib
import requests
import json
import pandas as pd
from datetime import date, timedelta

from requests.adapters import HTTPAdapter
from concurrent.futures import ThreadPoolExecutor
from urllib3.util import Retry

retry_strategy = Retry(
    total=4,  # Maximum number of retries
    backoff_factor=2, # Exponential backoff factor (e.g., 2 means 1, 2, 4, 8 seconds, ...)
    status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry on
)

# Create an HTTP adapter with the retry strategy and mount it to session
adapter = HTTPAdapter(max_retries=retry_strategy)

# Create a new session object
session = requests.Session()
session.mount('http://', adapter)
session.mount('https://', adapter)

files = []

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def download_url(url, download_path, name=None):
    try:
        global files

        if name:
            file_name = os.path.join(download_path, name)
        else:
            file_name = os.path.join(download_path, os.path.basename(url))
        dir_path = os.path.dirname(file_name)

        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)

        if os.path.isfile(file_name):
            # print(f"{file_name} already exists")
            return
        # Make a request using the session object
        response = session.get(url)
        if response.status_code == 404:
            print(f"File does not exist: {url}")
        elif response.status_code == 200:
            with open(file_name, 'wb') as f:
                f.write(response.content)
            # print(f"Downloaded: {url} to {file_name}")
            files.append(file_name)
            return
        else:
            print(f"Failed to download {url}")
            return
    except Exception as e:
        print(str(e))

def get_latest_data(pair, interval, limit=1000):
    limit = '1000'  # Fetch the last 1000 data points

    # API endpoint for batch request
    url = f'https://api.binance.com/api/v3/klines?symbol={pair}&interval={interval}&limit={limit}'

    # Make the API request
    response = requests.get(url)
    data = response.json()

    return data

def download_binance_monthly_data(cm_or_um, symbols, intervals, years, months, download_path):
    
    if cm_or_um not in ["cm", "um"]:
        print("CM_OR_UM can be only cm or um")
        return
    
    base_url = f"https://data.binance.vision/data/futures/{cm_or_um}/monthly/klines"

    # Main loop to iterate over all the arrays and launch child processes
    with ThreadPoolExecutor() as executor:
        for symbol in symbols:
            for interval in intervals:
                for year in years:
                    for month in months:
                        url = f"{base_url}/{symbol}/{interval}/{symbol}-{interval}-{year}-{month}.zip"
                        executor.submit(download_url, url, download_path)

def download_binance_daily_data(pair, training_days, interval, region, download_path):
    base_url = f"https://data.binance.vision/data/spot/daily/klines"

    end_date = date.today()
    start_date = end_date - timedelta(days=int(training_days))
    
    global files
    files = []

    with ThreadPoolExecutor() as executor:
        print(f"Downloading data for {pair}")
        for single_date in daterange(start_date, end_date):
            url = f"{base_url}/{pair}/{interval}/{pair}-{interval}-{single_date}.zip"
            executor.submit(download_url, url, download_path)
    
    return files

def download_binance_current_day_data(pair, region):
    limit = 1000
    base_url = f'https://api.binance.{region}/api/v3/klines?symbol={pair}&interval=1m&limit={limit}'

    # Make a request using the session object
    response = session.get(base_url)
    response.raise_for_status()
    resp = str(response.content, 'utf-8').rstrip()

    columns = ['start_time','open','high','low','close','volume','end_time','volume_usd','n_trades','taker_volume','taker_volume_usd','ignore']
    
    df = pd.DataFrame(json.loads(resp),columns=columns)
    df['date'] = [pd.to_datetime(x+1,unit='ms') for x in df['end_time']]
    df['date'] = df['date'].apply(pd.to_datetime)
    df[["volume", "taker_volume", "open", "high", "low", "close"]] = df[["volume", "taker_volume", "open", "high", "low", "close"]].apply(pd.to_numeric)

    return df.sort_index()