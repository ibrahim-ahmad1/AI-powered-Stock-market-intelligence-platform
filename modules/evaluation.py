import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
import joblib

def calculate_rmse(data_path, model_path, scaler_path):
    df = pd.read_csv(data_path)
    close_prices = df[["Close"]].values

    # Load scaler and scale data
    scaler = joblib.load(scaler_path)
    scaled_data = scaler.transform(close_prices)

    X, y = [], []
    window_size = 60
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i])
        y.append(scaled_data[i])
    X, y = np.array(X), np.array(y)

    # Load model
    model = load_model(model_path)

    # Make predictions
    predictions = model.predict(X)

    # Inverse transform predictions and actual values
    predictions = scaler.inverse_transform(predictions)
    y_actual = scaler.inverse_transform(y)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_actual, predictions))
    
    return rmse