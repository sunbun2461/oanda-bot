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