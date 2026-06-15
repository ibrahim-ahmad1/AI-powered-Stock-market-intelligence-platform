import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping

# ─────────────────────────────────────────
# Create Sequences (same as LSTM)
# ─────────────────────────────────────────
def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# ─────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────
def transformer_block(inputs, head_size=64, num_heads=2, ff_dim=64, dropout=0.1):
    # Multi-Head Attention
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x + inputs)  # Residual connection

    # Feed Forward Network
    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dropout(dropout)(ff)
    ff = Dense(inputs.shape[-1])(ff)
    x = LayerNormalization(epsilon=1e-6)(x + ff)  # Residual connection

    return x

# ─────────────────────────────────────────
# Build Transformer Model
# ─────────────────────────────────────────
def build_transformer_model(input_shape, head_size=64, num_heads=2, ff_dim=64, num_blocks=2, dropout=0.1):
    inputs = Input(shape=input_shape)
    x = inputs

    # Stack multiple Transformer blocks
    for _ in range(num_blocks):
        x = transformer_block(x, head_size, num_heads, ff_dim, dropout)

    # Global average pooling + output
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(dropout)(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# ─────────────────────────────────────────
# Train Transformer Model
# ─────────────────────────────────────────
def train_transformer_model(input_path, model_path, scaler_path, window_size=60):
    df = pd.read_csv(input_path)
    close_prices = df[["Close"]].values

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Create sequences
    X, y = create_sequences(scaled_data, window_size)

    # Train-test split (80/20)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build model
    model = build_transformer_model(
        input_shape=(X_train.shape[1], X_train.shape[2])
    )

    print(model.summary())

    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    # Train
    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    # Save model and scaler
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path.replace(".h5", ".keras"))
    joblib.dump(scaler, scaler_path)
    print(f"✅ Transformer model saved to {model_path}")

    return model

# ─────────────────────────────────────────
# Predict Next Day Price
# ─────────────────────────────────────────
def predict_next_price_transformer(input_path, model_path, scaler_path, window_size=60):
    from tensorflow.keras.models import load_model
    df = pd.read_csv(input_path)
    close_prices = df[["Close"]].values

    scaler = joblib.load(scaler_path)
    scaled_data = scaler.transform(close_prices)

    last_window = scaled_data[-window_size:]
    X_input = np.array([last_window])

    model = load_model(model_path)
    predicted_scaled = model.predict(X_input)
    predicted_price = scaler.inverse_transform(predicted_scaled)

    return round(float(predicted_price[0][0]), 2)