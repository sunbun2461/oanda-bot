Yes, let's rewrite the README to reflect that you're using **OANDA's WebSocket** from the start. Here's the revised version:

---

# **OANDA Mid/High Velocity Trading Bot**

This trading bot is built to operate on **OANDA's WebSocket API**, leveraging real-time streaming market data for efficient mid- to high-velocity forex trading. The bot is designed to run on minute-to-minute or 5-minute charts, utilizing a **Moving Average Crossover** strategy, with **RSI confirmation** and **volume filters** to optimize trade execution.

## **Features**

-   **Strategy**: Moving Average Crossover (9 EMA and 21 EMA) with RSI and volume filters for precise entries.
-   **Real-Time Data**: The bot uses OANDA's WebSocket API to stream real-time price updates, minimizing latency.
-   **Risk Management**: 0.2% risk per trade, with a daily risk cap of 6%.
-   **Stop-Loss/Take-Profit**: Dynamic stop-loss and take-profit levels based on the ATR (Average True Range).
-   **Trading Frequency**: Up to 30 trades per day, automatically executed based on the trading signals.
-   **Database Integration**: Uses SQLite to track trades and performance metrics, with the option to switch to MongoDB for scalability.

---

## **Setup Guide**

### **1. Python Environment Setup**

Make sure you have **Python 3.x** installed. Set up a virtual environment:

```bash
python3 -m venv oanda_bot_env
source oanda_bot_env/bin/activate  # Mac/Linux
.\oanda_bot_env\Scripts\activate  # Windows
```

Install the required packages:

```bash
pip install requests pandas numpy ta-lib websocket-client
```

### **2. API Key and WebSocket Configuration**

Get your **API key** from OANDA and add it to the `config.py` file:

```python
# config.py
API_KEY = 'your_oanda_api_key'
ACCOUNT_ID = 'your_oanda_account_id'
```

The WebSocket will be used to stream real-time price data and process signals.

### **3. Database Setup**

The bot uses **SQLite** to track trade history and performance. The database is automatically created when the bot is run for the first time.

---

## **File Structure**

```
/oanda_bot_project
  ├── main.py               # Main entry point for running the bot
  ├── strategies.py         # Contains trading logic (crossovers, RSI, volume filters)
  ├── risk_management.py    # Handles stop-loss and take-profit calculations
  ├── websocket_client.py   # Connects to OANDA WebSocket for real-time data
  ├── db.py                 # Database operations (SQLite for now, MongoDB later)
  ├── utils.py              # Utility functions (logging, data handling)
  ├── config.py             # API credentials and settings
  └── README.md             # Documentation
```

---

## **Trading Strategy Breakdown**

-   **EMA Crossover**: The bot monitors the 9-period and 21-period exponential moving averages (EMA). A bullish crossover signals a buy, and a bearish crossover signals a sell.
-   **RSI Confirmation**: RSI is checked before placing a trade to ensure the market is not overbought or oversold. Buy signals are only valid if RSI is below 40, and sell signals are valid if RSI is above 60.

-   **Volume Filter**: Volume is checked to ensure there’s sufficient market activity before executing trades. This helps to avoid false signals.

-   **Stop-Loss and Take-Profit**: The stop-loss is set at 1x to 1.5x ATR, and take-profit is set at 1.5x ATR, or the stop can trail with the price movement.

---

## **Running the Bot**

Start the bot by running the `main.py` file:

```bash
python main.py
```

The bot will:

1. Connect to OANDA's WebSocket API for real-time price data.
2. Analyze the data using the strategy outlined (EMA crossovers with RSI and volume filters).
3. Place trades and manage risk accordingly.
4. Log all trades and performance data in the SQLite database.

---

## **WebSocket API Details**

The bot uses OANDA's WebSocket for real-time market data. For detailed information about OANDA's WebSocket API, you can visit their official documentation:

-   **OANDA Streaming Endpoints**: [https://developer.oanda.com/rest-live-v20/stream/](https://developer.oanda.com/rest-live-v20/stream/)

---

## **Future Improvements**

-   **MongoDB Integration**: Migrate from SQLite to MongoDB for better scalability and cloud-based data storage.
-   **Advanced Strategy Filters**: Add time-of-day filters or news-based trading adjustments.
-   **Backtesting Module**: Add functionality to backtest the strategy on historical data before going live.

---

This README provides a comprehensive overview of the bot’s setup and functionality, using OANDA's WebSocket API from the start to ensure real-time responsiveness.

Let me know if you'd like to adjust or add anything else!
