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