#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from datetime import datetime

# Function to create sequences
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length - 1):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(x), np.array(y)

# Function to build and train the LSTM model
def build_and_train_lstm(x_train, y_train, seq_length):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    return model

# Streamlit interface
st.title('BSE SENSEX Stock Price Forecasting')
stock_code = st.text_input('Enter BSE stock code (e.g., SBIN.BO):', 'SBIN.BO')
start_date = st.date_input('Select start date:', datetime(2021, 6, 22))
end_date = st.date_input('Select end date:', datetime.now())

if st.button('Forecast'):
    data = yf.download(stock_code, start=start_date, end=end_date)
    if data.empty:
        st.error('No data found for the given stock code and date range.')
    else:
        data = data[['Close']]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        seq_length = 60
        x, y = create_sequences(scaled_data, seq_length)
        train_size = int(len(x) * 0.8)
        x_train, x_test = x[:train_size], x[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        model = build_and_train_lstm(x_train, y_train, seq_length)

        y_pred = model.predict(x_test)
        y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_inverse = scaler.inverse_transform(y_pred)

        last_sequence = scaled_data[-seq_length:]
        predicted_sequence = []
        for _ in range(15):
            last_sequence_reshaped = last_sequence.reshape((1, seq_length, 1))
            predicted_price = model.predict(last_sequence_reshaped)
            predicted_sequence.append(predicted_price[0, 0])
            last_sequence = np.append(last_sequence[1:], predicted_price)
        predicted_sequence = scaler.inverse_transform(np.array(predicted_sequence).reshape(-1, 1))

        # Plotting
        fig, ax = plt.subplots(figsize=(14, 7))
        train_dates = data.index[:train_size + seq_length]
        train_data = scaler.inverse_transform(scaled_data[:train_size + seq_length])
        test_dates = data.index[train_size + seq_length:train_size + seq_length + len(y_test)]
        test_data = scaler.inverse_transform(scaled_data[train_size + seq_length:train_size + seq_length + len(y_test)])
        forecast_dates = pd.date_range(start=data.index[-1], periods=16, closed='right')

        ax.plot(train_dates, train_data, label='Train Data')
        ax.plot(test_dates, test_data, label='Test Data')
        ax.plot(test_dates, y_pred_inverse, label='Predicted Data', linestyle='--')
        ax.plot(forecast_dates, predicted_sequence, label='Forecasted Data', linestyle='--')

        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(f'{stock_code} Stock Price - Historical, Test, Predicted, and Forecasted')
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)
        st.write(f'Next 15 days forecasted prices for {stock_code}:')
        st.write(pd.DataFrame(predicted_sequence, index=forecast_dates, columns=['Forecasted Price']))

