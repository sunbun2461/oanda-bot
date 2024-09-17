# Contents of ./fetch_hist_data.py
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

# Contents of ./feature_engineering.py
import pandas as pd
import os

# Function to calculate moving averages (SMA and EMA)
def add_moving_averages(df, sma_periods=[20, 50, 200], ema_periods=[5, 9, 20, 50, 200]):
    # Calculate SMAs for given periods
    for period in sma_periods:
        df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()

    # Calculate EMAs for given periods
    for period in ema_periods:
        df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

    return df

# Function to calculate RSI
def add_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# Function to calculate ATR
def add_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=period).mean()
    return df

# Function to calculate Bollinger Bands
def add_bollinger_bands(df, period=20, std_dev=2):
    df['SMA_20'] = df['close'].rolling(window=period).mean()
    df['Bollinger_Upper'] = df['SMA_20'] + (df['close'].rolling(window=period).std() * std_dev)
    df['Bollinger_Lower'] = df['SMA_20'] - (df['close'].rolling(window=period).std() * std_dev)
    return df

# Apply feature engineering to all pairs
def feature_engineering(filepath):
    df = pd.read_csv(filepath)
    
    # Add moving averages
    df = add_moving_averages(df)
    
    # Add RSI
    df = add_rsi(df)
    
    # Add ATR
    df = add_atr(df)
    
    # Add Bollinger Bands
    df = add_bollinger_bands(df)
    
    desired_order = ['time', 'open', 'high', 'low', 'close', 'volume', 'EMA_5', 'EMA_9', 'SMA_20', 'EMA_20', 'SMA_50', 'EMA_50', 'SMA_200', 'EMA_200', 'RSI', 'ATR', 'Bollinger_Upper', 'Bollinger_Lower']
    df = df[desired_order]  # Reorder the DataFrame columns

    # Save the enhanced data
    df.to_csv(filepath, index=False)
    print(f"Features added and saved to {filepath}")

# Directory setup
input_dir = 'cleaned_data/'

# Iterate over each CSV in cleaned_data and apply feature engineering
for file in os.listdir(input_dir):
    if file.endswith(".csv"):
        filepath = os.path.join(input_dir, file)
        feature_engineering(filepath)

print("Feature engineering completed.")

# Contents of ./db.py


# Contents of ./risk_management.py


# Contents of ./api_handler.py
# api_handler.py

import requests
import time
from config import API_KEY, ACCOUNT_ID, DEFAULT_INSTRUMENTS, DEFAULT_INTERVAL, BASE_URL

def fetch_prices():
    url = f"{BASE_URL}/v3/accounts/{ACCOUNT_ID}/pricing"
    params = {
        'instruments': DEFAULT_INSTRUMENTS
    }
    headers = {
        'Authorization': f'Bearer {API_KEY}'
    }
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def start_price_updates():
    while True:
        prices = fetch_prices()
        if prices:
            print(prices)  # Handle the prices as needed
        time.sleep(DEFAULT_INTERVAL)  # Wait before fetching again

if __name__ == "__main__":
    print("Starting price updates...")
    start_price_updates()

# Contents of ./config.py
API_KEY = '80a2e02b46dc4c1eef251fcb13f44a76-4e6ae796de996f78357fd1298d8be768'
ACCOUNT_ID = '001-001-3841160-001'

# Define default settings
DEFAULT_INSTRUMENTS = 'EUR_USD,USD_CAD'
DEFAULT_TIMEFRAME = 'M5'  # e.g., M1 (1 min), M5 (5 min), H1 (1 hour)
DEFAULT_INTERVAL = 300  # in seconds, (e.g., 300 for 5 min, 900 for 15 min), but this might actually be how many candles before the trade cuts off

# Other relevant settings
BASE_URL = 'https://api-fxtrade.oanda.com'


#Backtesting settings

EMA_SHORT = 5
EMA_LONG = 20
# ATR_MULTIPLIER = 1.5


# Set the number of candles to limit in the backtest
CANDLE_LIMIT = 150000  # Adjust this value to limit the number of candles processed

# Choose whether to use EMA or SMA
USE_EMA = True  # Set to True for EMA or False for SMA

# ATR-related variables
ATR_MULTIPLIER = 4.5  # Set the ATR multiplier for take-profit
STOP_LOSS_ATR = .75   # Set the ATR multiplier for stop-loss

# Initial balance for the backtest
INITIAL_BALANCE = 1000  # Set the initial balance for the backtest
PAIR_NAME = False  # Set the pair name for the backtest

# New: Starting year and month for filtering
START_YEAR = 2022  # Specify the start year (e.g., 2015)
START_MONTH = 5    # Specify the start month (July), but use digits (e.g., 1 for January, 2 for February, etc.)

# Contents of ./strategies.py


# Contents of ./websocket_client.py
import websocket
import ssl
import certifi
import json
import threading
from config import API_KEY, ACCOUNT_ID

# Function to handle incoming messages from the WebSocket
def on_message(ws, message):
    print("Received data:", message)

# Function to handle errors
def on_error(ws, error):
    print("Error:", error)

# Function to handle connection closure
def on_close(ws, close_status_code, close_msg):
    print(f"WebSocket closed with code: {close_status_code} and message: {close_msg}")

# Function to handle WebSocket opening
def on_open(ws):
    print("WebSocket connection opened")
    # Subscribing to the heartbeat stream
    ws.send(json.dumps({
        "type": "heartbeat"
    }))

# Function to start the WebSocket connection
def start_websocket():
    # WebSocket URL for live account
    url = f"wss://stream-fxtrade.oanda.com/v3/accounts/{ACCOUNT_ID}/pricing/stream?instruments=EUR_USD"

    headers = {
        'Authorization': f'Bearer {API_KEY}'
    }

    ws = websocket.WebSocketApp(url,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close,
                                header=headers)

    # Start the WebSocket connection with SSL certificate validation
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_REQUIRED, "ca_certs": certifi.where()})

# Start WebSocket in a separate thread
if __name__ == "__main__":
    ws_thread = threading.Thread(target=start_websocket)
    ws_thread.start()

    input("Press Enter to stop...\n")

# Contents of ./test.py
import requests
from config import API_KEY, ACCOUNT_ID

# Define the headers with your API key
headers = {
    'Authorization': f'Bearer {API_KEY}',
}

# Define the API endpoint for fetching account details
url = f"https://api-fxtrade.oanda.com/v3/accounts/{ACCOUNT_ID}"

response = requests.get(url, headers=headers)

if response.status_code == 200:
    print("Account details fetched successfully!")
    print(response.json())
else:
    print(f"Error: {response.status_code} - {response.text}")

# Contents of ./combined_code.py
# Contents of ./fetch_hist_data.py
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

# Contents of ./feature_engineering.py
import pandas as pd
import os

# Function to calculate moving averages (SMA and EMA)
def add_moving_averages(df, sma_periods=[20, 50, 200], ema_periods=[5, 9, 20, 50, 200]):
    # Calculate SMAs for given periods
    for period in sma_periods:
        df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()

    # Calculate EMAs for given periods
    for period in ema_periods:
        df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

    return df

# Function to calculate RSI
def add_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# Function to calculate ATR
def add_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=period).mean()
    return df

# Function to calculate Bollinger Bands
def add_bollinger_bands(df, period=20, std_dev=2):
    df['SMA_20'] = df['close'].rolling(window=period).mean()
    df['Bollinger_Upper'] = df['SMA_20'] + (df['close'].rolling(window=period).std() * std_dev)
    df['Bollinger_Lower'] = df['SMA_20'] - (df['close'].rolling(window=period).std() * std_dev)
    return df

# Apply feature engineering to all pairs
def feature_engineering(filepath):
    df = pd.read_csv(filepath)
    
    # Add moving averages
    df = add_moving_averages(df)
    
    # Add RSI
    df = add_rsi(df)
    
    # Add ATR
    df = add_atr(df)
    
    # Add Bollinger Bands
    df = add_bollinger_bands(df)
    
    desired_order = ['time', 'open', 'high', 'low', 'close', 'volume', 'EMA_5', 'EMA_9', 'SMA_20', 'EMA_20', 'SMA_50', 'EMA_50', 'SMA_200', 'EMA_200', 'RSI', 'ATR', 'Bollinger_Upper', 'Bollinger_Lower']
    df = df[desired_order]  # Reorder the DataFrame columns

    # Save the enhanced data
    df.to_csv(filepath, index=False)
    print(f"Features added and saved to {filepath}")

# Directory setup
input_dir = 'cleaned_data/'

# Iterate over each CSV in cleaned_data and apply feature engineering
for file in os.listdir(input_dir):
    if file.endswith(".csv"):
        filepath = os.path.join(input_dir, file)
        feature_engineering(filepath)

print("Feature engineering completed.")

# Contents of ./db.py


# Contents of ./risk_management.py


# Contents of ./api_handler.py
# api_handler.py

import requests
import time
from config import API_KEY, ACCOUNT_ID, DEFAULT_INSTRUMENTS, DEFAULT_INTERVAL, BASE_URL

def fetch_prices():
    url = f"{BASE_URL}/v3/accounts/{ACCOUNT_ID}/pricing"
    params = {
        'instruments': DEFAULT_INSTRUMENTS
    }
    headers = {
        'Authorization': f'Bearer {API_KEY}'
    }
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def start_price_updates():
    while True:
        prices = fetch_prices()
        if prices:
            print(prices)  # Handle the prices as needed
        time.sleep(DEFAULT_INTERVAL)  # Wait before fetching again

if __name__ == "__main__":
    print("Starting price updates...")
    start_price_updates()

# Contents of ./config.py


# Contents of ./utils.py


# Contents of ./tet.py
import requests
from config import API_KEY, ACCOUNT_ID

# Define the headers with your API key
headers = {
    'Authorization': f'Bearer {API_KEY}',
}

# Define the API endpoint for fetching account details
url = f"https://api-fxtrade.oanda.com/v3/accounts/{ACCOUNT_ID}"

response = requests.get(url, headers=headers)

if response.status_code == 200:
    print("Account details fetched successfully!")
    print(response.json())
else:
    print(f"Error: {response.status_code} - {response.text}")

# Contents of ./main.py


# Contents of ./gpt.py
import os

# Define the root directory (current directory in this case)
root_dir = '.'

# Output file to store combined code
output_file = 'combined_code.py'

# Directory to ignore (your virtual environment)
ignore_dir = 'oanda_bot_env'

# Open the output file in write mode
with open(output_file, 'w') as outfile:
    # Walk through all directories and subdirectories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip the virtual environment directory
        if ignore_dir in dirnames:
            dirnames.remove(ignore_dir)
        
        for filename in filenames:
            # Only process .py files
            if filename.endswith('.py'):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        # Write file name and contents to the combined file
                        outfile.write(f"# Contents of {file_path}\n")
                        outfile.write(infile.read())
                        outfile.write("\n\n")
                except UnicodeDecodeError:
                    print(f"Skipping {file_path} due to encoding issues.")

print(f"All Python files (excluding {ignore_dir}) have been combined into {output_file}")

# Contents of ./backtesting/backtest.py
import sys
import os
import pandas as pd
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    EMA_SHORT, EMA_LONG, ATR_MULTIPLIER, CANDLE_LIMIT, 
    USE_EMA, STOP_LOSS_ATR, INITIAL_BALANCE, PAIR_NAME, 
    START_YEAR, START_MONTH
)

# Function to calculate position size
def calculate_position_size(balance, trade_loss_limit, entry_price, stop_loss):
    risk_per_trade = balance * trade_loss_limit
    stop_loss_distance = entry_price - stop_loss
    return risk_per_trade / stop_loss_distance

# Function to enter a trade with ATR multiplier and stop loss ATR as parameters
def enter_trade(row, balance, trade_loss_limit):
    entry_price = row['close']
    stop_loss = entry_price - row['ATR'] * STOP_LOSS_ATR
    take_profit = entry_price + row['ATR'] * ATR_MULTIPLIER

    # Calculate position size
    position_size = calculate_position_size(balance, trade_loss_limit, entry_price, stop_loss)
    
    return {
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'position_size': position_size,
        'entry_time': row['time']
    }

# Function to manage open trades
def manage_trade(position, row, balance, daily_loss, total_profit, total_loss, profit_trades, loss_trades):
    if row['low'] <= position['stop_loss']:  # Stop-loss hit
        loss = (position['entry_price'] - position['stop_loss']) * position['position_size']
        balance -= loss
        daily_loss += loss / balance
        total_loss += loss
        loss_trades += 1
        print(f"Trade closed at stop-loss on {row['time']} with a loss of {loss}. New balance: {balance}")
        return None, balance, daily_loss, total_profit, total_loss, profit_trades, loss_trades, 'loss', position['stop_loss']

    elif row['high'] >= position['take_profit']:  # Take-profit hit
        gain = (position['take_profit'] - position['entry_price']) * position['position_size']
        balance += gain
        total_profit += gain
        profit_trades += 1
        print(f"Trade closed at take-profit on {row['time']} with a profit of {gain}. New balance: {balance}")
        return None, balance, daily_loss, total_profit, total_loss, profit_trades, loss_trades, 'profit', position['take_profit']

    return position, balance, daily_loss, total_profit, total_loss, profit_trades, loss_trades, None, None

# Function to calculate moving averages
def add_moving_averages(data, use_ema=False):
    if use_ema:
        data['EMA_20'] = data['close'].ewm(span=EMA_SHORT, adjust=False).mean()
        data['EMA_50'] = data['close'].ewm(span=EMA_LONG, adjust=False).mean()
    else:
        data['SMA_20'] = data['close'].rolling(window=EMA_SHORT).mean()
        data['SMA_50'] = data['close'].rolling(window=EMA_LONG).mean()
    return data

# New: Function to filter the data by year and month
def filter_data_by_year_and_month(data, start_year, start_month):
    data['year'] = pd.to_datetime(data['time']).dt.year
    data['month'] = pd.to_datetime(data['time']).dt.month
    filtered_data = data[(data['year'] >= start_year) & (data['month'] >= start_month)]
    return filtered_data

# New: Function to calculate the duration of the backtest
def calculate_duration(start_date, end_date):
    delta = end_date - start_date
    total_days = delta.days
    duration_years = total_days // 365
    remaining_days = total_days % 365
    duration_months = remaining_days // 30
    return duration_years, duration_months

# Function to run the backtest
def run_backtest(instrument, data, candle_limit=None, use_ema=False):
    trades = []
    position = None
    balance = INITIAL_BALANCE
    daily_loss = 0
    max_daily_loss = 0.06
    trade_loss_limit = 0.005
    current_date = None
    total_profit = 0
    total_loss = 0
    num_trades = 0
    profit_trades = 0
    loss_trades = 0

    # Add the moving averages based on user's choice (EMA or SMA)
    data = add_moving_averages(data, use_ema=use_ema)

    # Set the limit for the number of candles (default is all candles if not specified)
    candle_count = len(data)
    if candle_limit is not None:
        candle_count = min(candle_limit, len(data))

    # Get start and end dates from filtered data
    start_date = pd.to_datetime(data['time'].iloc[200])  # Start after MA calculations
    end_date = pd.to_datetime(data['time'].iloc[candle_count - 1])  # Last candle within the limit

    # Calculate the backtest duration based on start and end dates
    duration_years, duration_months = calculate_duration(start_date, end_date)

    # Print the start and end date for debugging
    print(f"Backtest Start Date: {start_date}")
    print(f"Backtest End Date: {end_date}")

    for i in range(200, candle_count):  # Start after 200 rows for MA calculations
        row = data.iloc[i]
        num_trades += 1

        # Extract current row date (day only)
        row_date = pd.to_datetime(row['time']).date()

        # Reset daily loss if it's a new day
        if current_date is None or row_date != current_date:
            current_date = row_date
            daily_loss = 0
            print(f"\n---- Starting new trading day: {current_date} ----")

        if daily_loss >= max_daily_loss:
            print(f"Max daily loss reached for {instrument} on {current_date}. Skipping to the next day.")
            continue

        # Entry condition: 20 EMA crosses above 50 EMA and RSI is below 70
        if use_ema:
            ma_20_above_ma_50 = row['EMA_20'] > row['EMA_50']
        else:
            ma_20_above_ma_50 = row['SMA_20'] > row['SMA_50']

        if position is None and ma_20_above_ma_50 and row['RSI'] < 70:
            position = enter_trade(row, balance, trade_loss_limit)
            print(f"Entering trade on {instrument} at {position['entry_price']} with position size {position['position_size']}")

        # Manage open position
        elif position:
            position, balance, daily_loss, total_profit, total_loss, profit_trades, loss_trades, result, exit_price = manage_trade(
                position, row, balance, daily_loss, total_profit, total_loss, profit_trades, loss_trades
            )
            if result:
                trades.append({
                    'time': row['time'],
                    'type': 'sell',
                    'result': result,
                    'entry_price': position['entry_price'] if position else None,
                    'exit_price': exit_price,
                    'profit_loss': total_profit - total_loss,
                    'position_size': position['position_size'] if position else None,
                    'balance': balance
                })

    # Calculate final statistics
    win_rate = (profit_trades / num_trades) * 100 if num_trades > 0 else 0
    loss_rate = (loss_trades / num_trades) * 100 if num_trades > 0 else 0
    total_return = balance - INITIAL_BALANCE

    # Colorize total return and final balance
    total_return_colored = f"\033[92m${total_return:.2f}\033[0m" if total_return > 0 else f"\033[91m${total_return:.2f}\033[0m"
    final_balance_colored = f"\033[92m${balance:.2f}\033[0m" if balance > INITIAL_BALANCE else f"\033[91m${balance:.2f}\033[0m"

    print("\n---- Backtest Summary ----")
    print(f"Instrument: {instrument}")
    print(f"Initial balance: ${INITIAL_BALANCE:.2f}")
    print(f"Final balance: {final_balance_colored}")
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Total Loss: ${total_loss:.2f}")
    print(f"Total Return: {total_return_colored}")
    print(f"Number of trades: {num_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Loss Rate: {loss_rate:.2f}%")
    print(f"Backtest Duration: {duration_years} year(s) and {duration_months} month(s)")
    print(f"Backtest Start Date: {start_date}")
    print(f"Backtest End Date: {end_date}")

    # Append summary stats to the trades list
    trades.append({
        'time': 'summary',
        'type': '',
        'result': '',
        'entry_price': '',
        'exit_price': '',
        'profit_loss': '',
        'position_size': '',
        'balance': f"{balance:.2f}",
        'total_profit': f"{total_profit:.2f}",
        'total_loss': f"{total_loss:.2f}",
        'num_trades': num_trades,
        'win_rate': f"{win_rate:.2f}%",
        'loss_rate': f"{loss_rate:.2f}%"
    })

    return trades

# Function to handle running backtests for all instruments or a specific pair
def backtest_instruments(input_dir, output_base_dir, pair=None, use_ema=USE_EMA):
    # If a pair is specified, backtest only that pair
    if pair:
        filepath = os.path.join(input_dir, f"{pair}.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)

            # Filter data by year and month
            df = filter_data_by_year_and_month(df, START_YEAR, START_MONTH)

            trades = run_backtest(pair, df, candle_limit=CANDLE_LIMIT, use_ema=use_ema)
            save_results(pair, trades, output_base_dir)
        else:
            print(f"Error: The pair '{pair}' does not exist in the dataset.")
        return
    
    # If no pair is specified, backtest all instruments
    for file in os.listdir(input_dir):
        if file.endswith(".csv"):
            instrument = file.split(".")[0]
            filepath = os.path.join(input_dir, file)

            # Read in the cleaned data
            df = pd.read_csv(filepath)

            # Filter data by year and month
            df = filter_data_by_year_and_month(df, START_YEAR, START_MONTH)

            # Run backtest
            trades = run_backtest(instrument, df, candle_limit=CANDLE_LIMIT, use_ema=use_ema)

            # Save results
            save_results(instrument, trades, output_base_dir)

# Function to save backtest results
def save_results(instrument, trades, output_base_dir):
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    instrument_dir = os.path.join(output_base_dir, instrument)
    if not os.path.exists(instrument_dir):
        os.makedirs(instrument_dir)

    # Save results
    results_filename = f'{instrument_dir}/{instrument}_backtest_results_{current_time}.csv'
    trades_df = pd.DataFrame(trades)  # Assuming trades is a list of dictionaries
    trades_df.to_csv(results_filename, index=False)
    print(f"Backtest results for {instrument} saved in {results_filename}.")

# Main logic
input_dir = 'cleaned_data/'
output_base_dir = 'backtesting_results/'

# Example: Run backtest on a specific pair starting from a specific year and month
backtest_instruments(input_dir, output_base_dir, pair=PAIR_NAME)

# Contents of ./data_cleaning/clean_historical_data.py
import os
import pandas as pd

# Define directories
input_dir = 'historical_data/'
output_dir = 'cleaned_data/'

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to clean and save data
def clean_data(filename):
    df = pd.read_csv(filename)
    
    # Step 1: Remove duplicates
    df_cleaned = df.drop_duplicates(subset='time')
    
    # Step 2: Handle missing values (drop rows with any missing data)
    df_cleaned = df_cleaned.dropna()
    
    # Step 3: Convert the time column to datetime format
    df_cleaned['time'] = pd.to_datetime(df_cleaned['time'])
    
    # Step 4: Check for gaps in time (assuming 5 minutes interval)
    df_cleaned = df_cleaned.sort_values(by='time')
    time_diff = df_cleaned['time'].diff().dt.total_seconds().dropna()
    gaps = time_diff[time_diff > 300]  # 300 seconds = 5 minutes
    if not gaps.empty:
        print(f"Gaps found in {filename}: \n{gaps}")
    
    # Save the cleaned data to the output directory
    output_filename = os.path.join(output_dir, os.path.basename(filename))
    df_cleaned.to_csv(output_filename, index=False)
    print(f"Data cleaned and saved to {output_filename}")

# Iterate over all files in the historical_data folder
for file in os.listdir(input_dir):
    if file.endswith(".csv"):
        filepath = os.path.join(input_dir, file)
        clean_data(filepath)

print("All files processed and cleaned.")

