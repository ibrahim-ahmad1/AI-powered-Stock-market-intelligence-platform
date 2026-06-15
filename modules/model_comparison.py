import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────
# Helper: Create Sequences
# ─────────────────────────────────────────
def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# ─────────────────────────────────────────
# Compare LSTM vs Transformer
# ─────────────────────────────────────────
def compare_models(
    data_path,
    lstm_model_path, lstm_scaler_path,
    transformer_model_path, transformer_scaler_path,
    window_size=60
):
    df = pd.read_csv(data_path)
    close_prices = df[["Close"]].values

    results = {}

    for name, model_path, scaler_path in [
        ("LSTM", lstm_model_path, lstm_scaler_path),
        ("Transformer", transformer_model_path, transformer_scaler_path)
    ]:
        scaler = joblib.load(scaler_path)
        scaled_data = scaler.transform(close_prices)

        X, y = create_sequences(scaled_data, window_size)

        # Use only test set (last 20%)
        split = int(0.8 * len(X))
        X_test = X[split:]
        y_test = y[split:]

        model = load_model(model_path)
        predictions_scaled = model.predict(X_test)

        # Inverse transform
        predictions = scaler.inverse_transform(predictions_scaled)
        y_actual = scaler.inverse_transform(y_test)

        rmse = round(float(np.sqrt(mean_squared_error(y_actual, predictions))), 4)
        mae  = round(float(mean_absolute_error(y_actual, predictions)), 4)
        mape = round(float(np.mean(np.abs((y_actual - predictions) / y_actual)) * 100), 2)

        results[name] = {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "predictions": predictions.flatten().tolist(),
            "actual": y_actual.flatten().tolist()
        }

        print(f"✅ {name} — RMSE: {rmse} | MAE: {mae} | MAPE: {mape}%")

    # Winner
    winner = "LSTM" if results["LSTM"]["rmse"] < results["Transformer"]["rmse"] else "Transformer"
    print(f"\n🏆 Better Model: {winner} (lower RMSE)")
    results["winner"] = winner

    return results