import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import yfinance as yf
def main(symbol):
  today = datetime.date.today().strftime('%Y-%m-%d')
  df = yf.download(symbol, start="2020-01-01", end=today)[['Close']]
  df.dropna(inplace=True)
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled_data = scaler.fit_transform(df)
  def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)
  seq_length = 60
  X, y = create_sequences(scaled_data, seq_length)
  X = X.reshape((X.shape[0], seq_length, 1))
  model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    LSTM(50),
    Dense(1)
  ])
  model.compile(optimizer='adam', loss='mean_squared_error')
  model.fit(X, y, epochs=10, batch_size=32, verbose=1)
  predicted = model.predict(X)
  predicted_prices = scaler.inverse_transform(predicted)
  actual_prices = scaler.inverse_transform(y.reshape(-1, 1))
  plt.figure(figsize=(12, 6))
  plt.plot(actual_prices, label='Actual Prices')
  plt.plot(predicted_prices, label='Predicted Prices', alpha=0.7)
  plt.title('{symbol} using LSTM (Training Fit)')
  plt.xlabel('Time Steps')
  plt.ylabel('Price (USD)')
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.show()
  future_input = scaled_data[-seq_length:].reshape(1, seq_length, 1)
  future_predictions = []
  for _ in range(30):
    next_pred = model.predict(future_input, verbose=0)[0, 0]
    future_predictions.append(next_pred)
    future_input = np.append(future_input[:, 1:, :], [[[next_pred]]], axis=1)
  future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
  future_dates = pd.date_range(start=df.index[-1], periods=31, freq='D')[1:]
  plt.figure(figsize=(12, 6))
  plt.plot(df['Close'], label='Historical Prices')
  plt.plot(future_dates, future_prices, label='30-Day Forecast', color='red')
  plt.title('{symbol} 30-Day Price Forecast (LSTM)')
  plt.xlabel('Date')
  plt.ylabel('Price (USD)')
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.show()
inp = input('Enter the symbol(format: COINNAME-USD):')
main(inp)
