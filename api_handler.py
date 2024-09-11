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