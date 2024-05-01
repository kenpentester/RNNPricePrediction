import os
import time
import numpy as np
from rich.console import Console
from binaryapi.stable_api import Binary
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import sys
import asyncio
from deriv_api import DerivAPI
from deriv_api import APIError
from datetime import datetime

app_id = 33254
api_token = os.getenv('DERIV_TOKEN', 'GatUOOB4qST3Dum')

if len(api_token) == 0:
    sys.exit("DERIV_TOKEN environment variable is not set")
# Binary Token
token = os.environ.get('BINARY_TOKEN', 'GatUOOB4qST3Dum')

console = Console(log_path=False)
# Derivitives (1HZ10V, 1HZ25V, 1HZ50V, 1HZ75V, 1HZ100V, R_10)
# Forex (frxEURUSD, frxAUDCAD, frxAUDCHF, frxAUDUSD, frxEURGBP)
symbol = "frxAUDUSD"
initial_price = 1
duration = 300
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def message_handler(message):
    msg_type = message.get('msg_type')

    if msg_type in ['candles', 'ohlc']:
        # Print candles data from message
        candles_data = message['candles']
        price_data = candles_data

        df = pd.DataFrame(price_data)

        # Extract 'close' prices
        close_prices = df['close'].astype(float).values

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        close_prices_scaled = scaler.fit_transform(close_prices.reshape(-1, 1))

        # Prepare the data for training
        X, y = [], []
        for i in range(len(close_prices_scaled) - 1):
            X.append(close_prices_scaled[i])
            y.append(close_prices_scaled[i + 1])

        X, y = np.array(X), np.array(y)

        # Reshape input to be [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

        # Define the RNN model
        model = Sequential()
        model.add(SimpleRNN(units=50, activation='relu', input_shape=(1, 1)))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X, y, epochs=50, batch_size=32)

        # Predict the closing price of the next candlestick
        last_close = close_prices_scaled[-1].reshape(1, 1, 1)
        next_close_scaled = model.predict(last_close)
        next_close = scaler.inverse_transform(next_close_scaled.reshape(-1, 1))

        # Print the predicted closing price
        console.print(f"Predicted Closing Price of Next Candlestick: {next_close[0][0]:.5f}")
        print(f"Closing Price:  {price_data[-2]['close']}")
        print(f"Opening Price:  {price_data[-2]['open']}")

        predicted_price = next_close[0][0]

        if price_data[-2]['close'] < predicted_price:
            print("Place bullish trade")
            print("Current Time:", current_time)
            
        else:
            print("Place bearish trade")
            print("Current Time:", current_time)
            
if __name__ == '__main__':
    binary = Binary(token=token, message_callback=message_handler)

    #symbol = 'frxEURUSD'
    #symbol = symbol
    style = 'candles'  # 'ticks' or 'candles'
    end = 'latest'  # 'latest' or unix epoch
    count = 1000  # default: 5000 if not provided
    granularity = 900

    # Subscribe to ticks stream
    binary.api.ticks_history(
        ticks_history=symbol,
        style=style,
        count=count,
        end=end,
        granularity=granularity,
        subscribe=False,
    )

    # Wait for 60 seconds then exit the script
    time.sleep(10)
