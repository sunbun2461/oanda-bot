import requests
import pandas as pd
import time
from config import API_KEY, ACCOUNT_ID, DEFAULT_INSTRUMENTS, DEFAULT_INTERVAL, BASE_URL

def fetch_historical_data(instrument, granularity='M5', count=5000, before=None):
    url = f"{BASE_URL}/v3/instruments/{instrument}/candles"
    params = {
        'granularity': granularity,
        'count': count,
        'price': 'M'  # mid-prices
    }
    
    if before:
        params['to'] = before  # Use the 'to' parameter to fetch data before the last timestamp
    
    headers = {
        'Authorization': f'Bearer {API_KEY}'
    }
    
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()['candles']
        return data
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def get_last_timestamp(filename):
    try:
        # Load the existing CSV
        df = pd.read_csv(filename)
        # Get the oldest timestamp to fetch data before it
        last_timestamp = df['time'].iloc[0]  # Fetch the oldest timestamp (first row)
        print(f"Oldest timestamp found: {last_timestamp}")
        return last_timestamp
    except (FileNotFoundError, IndexError):
        # If file not found or empty, return None
        print("No previous data found, starting from the current date.")
        return None

def append_to_csv(data, filename):
    try:
        # Load the existing CSV (if it exists) and append new data
        df_existing = pd.read_csv(filename)
    except FileNotFoundError:
        df_existing = pd.DataFrame()  # Create a new one if not found

    # Convert the new data into a DataFrame
    df_new = pd.DataFrame([{
        'time': candle['time'],
        'open': candle['mid']['o'],
        'high': candle['mid']['h'],
        'low': candle['mid']['l'],
        'close': candle['mid']['c'],
        'volume': candle['volume']
    } for candle in data])

    # Append the new data to the existing CSV
    df_combined = pd.concat([df_new, df_existing], ignore_index=True)
    df_combined.to_csv(filename, index=False)
    print(f"Data appended to {filename}")

def get_candle_count(filename):
    try:
        df = pd.read_csv(filename)
        return len(df)
    except FileNotFoundError:
        return 0

if __name__ == "__main__":
    instrument = 'NZD_USD'  # Define the instrument to be fetched
    filename = f'historical_data/{instrument}.csv'
    target_candles = 1000000  # Set target number of candles

    # Keep fetching data until we reach the target candle count
    while get_candle_count(filename) < target_candles:
        last_timestamp = get_last_timestamp(filename)
        
        for _ in range(5):  # Fetch multiple batches of data
            data = fetch_historical_data(instrument, before=last_timestamp)
            
            if data:
                last_timestamp = data[0]['time']  # Update with the oldest timestamp from the data
                append_to_csv(data, filename)
            else:
                break  # Stop if no data is returned
        
        current_candle_count = get_candle_count(filename)
        print(f"Current candle count: {current_candle_count}")
    
    print(f"Finished fetching data. Reached {get_candle_count(filename)} candles.")