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