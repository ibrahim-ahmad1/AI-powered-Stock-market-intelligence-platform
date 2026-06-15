import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
import joblib

def create_sequences(data, window_size = 60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def train_lstm_model(input_path, model_path, scaler_path):
    df = pd.read_csv(input_path)
    close_prices = df[["Close"]].values

    #Scale data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(close_prices)

    #Create sequences
    X, y = create_sequences(scaled_data)

    # Train-test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    #Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    #Train model
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    #Save model and scaler
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    return model