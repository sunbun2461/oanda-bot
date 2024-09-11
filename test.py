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