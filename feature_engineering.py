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