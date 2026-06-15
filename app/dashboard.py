import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from modules.hybrid_engine import generate_trade_signal
from modules.risk_analysis import analyze_risk
from modules.model_comparison import compare_models

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(page_title="Stock Intelligence System", layout="wide")
st.title("🤖 AI-Powered Stock Market Intelligence Dashboard")

# ─────────────────────────────────────────
# FILE PATHS
# ─────────────────────────────────────────
stock_path              = "data/processed/stocks_with_indicators.csv"
sentiment_path          = "data/processed/daily_news_sentiment.csv"
lstm_model_path         = "models/lstm_model.h5"
lstm_scaler_path        = "models/scaler.pkl"
transformer_model_path  = "models/transformer_model.keras"
transformer_scaler_path = "models/transformer_scaler.pkl"

df = pd.read_csv(stock_path)

# ─────────────────────────────────────────
# STOCK PRICE CHART
# ─────────────────────────────────────────
st.subheader("📈 Stock Price Chart")
st.line_chart(df.set_index("Date")["Close"])

# ─────────────────────────────────────────
# NEXT DAY PRICE PREDICTION
# ─────────────────────────────────────────
st.subheader("🔮 Next Day Price Prediction")

lstm_model  = load_model(lstm_model_path)
lstm_scaler = joblib.load(lstm_scaler_path)
close_prices = df[["Close"]].values
scaled_data  = lstm_scaler.transform(close_prices)
X_input      = np.array([scaled_data[-60:]])
lstm_pred    = lstm_scaler.inverse_transform(lstm_model.predict(X_input))
lstm_price   = round(float(lstm_pred[0][0]), 2)

col1, col2 = st.columns(2)
col1.success(f"🧠 LSTM Prediction: ₹{lstm_price}")

if os.path.exists(transformer_model_path):
    trans_model  = load_model(transformer_model_path)
    trans_scaler = joblib.load(transformer_scaler_path)
    scaled_data2 = trans_scaler.transform(close_prices)
    X_input2     = np.array([scaled_data2[-60:]])
    trans_pred   = trans_scaler.inverse_transform(trans_model.predict(X_input2))
    trans_price  = round(float(trans_pred[0][0]), 2)
    col2.info(f"⚡ Transformer Prediction: ₹{trans_price}")
else:
    col2.warning("⚡ Transformer model not trained yet. Run main.py first.")

# ─────────────────────────────────────────
# LSTM vs TRANSFORMER COMPARISON
# ─────────────────────────────────────────
st.subheader("📊 LSTM vs Transformer — Model Comparison")

if os.path.exists(transformer_model_path):
    with st.spinner("Comparing models..."):
        results = compare_models(
            stock_path,
            lstm_model_path, lstm_scaler_path,
            transformer_model_path, transformer_scaler_path
        )

    # Metrics table
    comp_df = pd.DataFrame({
        "Metric":      ["RMSE", "MAE", "MAPE (%)"],
        "LSTM":        [results["LSTM"]["rmse"],        results["LSTM"]["mae"],        results["LSTM"]["mape"]],
        "Transformer": [results["Transformer"]["rmse"], results["Transformer"]["mae"], results["Transformer"]["mape"]],
    })
    st.table(comp_df)

    winner = results["winner"]
    st.success(f"🏆 Better Model: **{winner}** (lower RMSE)")

    # Actual vs Predicted chart
    chart_df = pd.DataFrame({
        "Actual":      results["LSTM"]["actual"],
        "LSTM":        results["LSTM"]["predictions"],
        "Transformer": results["Transformer"]["predictions"],
    })
    st.line_chart(chart_df)

else:
    st.info("📌 Train the Transformer model first by running main.py to see comparison.")

# ─────────────────────────────────────────
# HYBRID TRADING SIGNAL
# ─────────────────────────────────────────
st.subheader("🧠 Trading Decision")

decision     = generate_trade_signal(stock_path, sentiment_path)
signal_color = {"BUY": "green", "SELL": "red", "HOLD": "orange"}.get(decision['signal'], "gray")
st.markdown(f"### Signal: :{signal_color}[{decision['signal']}]")

col3, col4, col5 = st.columns(3)
col3.metric("RSI",             decision['rsi'])
col4.metric("Sentiment Score", decision['sentiment_score'])
col5.metric("MA Trend",        decision['ma_trend'].upper())

# ─────────────────────────────────────────
# RISK ANALYSIS
# ─────────────────────────────────────────
st.subheader("⚠️ Risk Analysis")

risk_info = analyze_risk(stock_path)

col6, col7, col8 = st.columns(3)
col6.metric("Volatility",    risk_info['volatility'])
col7.metric("Max Drawdown",  risk_info['max_drawdown'])
col8.metric("Risk Level",    risk_info['risk_level'])

# ─────────────────────────────────────────
# SHAP EXPLAINABILITY
# ─────────────────────────────────────────
st.subheader("🔍 Explainable AI — Why This Prediction?")

st.markdown("""
> **What is SHAP?**
> Our AI predicts tomorrow's stock price — but *why* that prediction?
> SHAP reveals **which past days influenced the model the most**,
> and how strongly they pushed the price **up** or **down**.
""")

col_a, col_b, col_c = st.columns(3)
col_a.info("🏆 **Most Influential Days**\nWhich past days had the biggest impact on today's prediction")
col_b.info("📈 **Positive Influence**\nDays that pushed predicted price **higher**")
col_c.info("📉 **Negative Influence**\nDays that pushed predicted price **lower**")

st.markdown("---")

shap_feat_path    = "data/processed/shap/shap_feature_importance.png"
shap_summary_path = "data/processed/shap/shap_summary.png"

if os.path.exists(shap_feat_path) and os.path.exists(shap_summary_path):
    col9, col10 = st.columns(2)

    with col9:
        st.image(shap_feat_path,
                 caption="📊 Most Influential Past Days",
                 use_column_width=True)
        st.markdown("""
        **How to read:**
        - 🔵 **Blue bar** = Most influential day
        - 🟢 **Green bar** = Moderately influential
        - 🟠 **Orange bar** = Less influential
        - Longer bar = that day had stronger impact on prediction
        """)

    with col10:
        st.image(shap_summary_path,
                 caption="📈 Influence Pattern Over 60 Days",
                 use_column_width=True)
        st.markdown("""
        **How to read:**
        - X-axis = how many days ago (1 = yesterday, 60 = 60 days ago)
        - Y-axis = how much that day influenced the prediction
        - 🔴 Red dotted line = single most influential day
        - Peak = the day model relied on most
        """)

    st.markdown("---")
    st.markdown("### 💡 What does this mean in simple words?")
    st.success("""
    Our LSTM model looks at the last 60 days of closing prices to predict tomorrow's price.
    The charts above show WHICH of those 60 days had the most influence on today's prediction.

    For example — if "3 days ago" has the longest bar, it means a price movement
    3 days back is strongly influencing what the model thinks will happen tomorrow.

    This is powerful because it shows the model has learned real market patterns —
    recent price movements naturally matter more than older ones! ✅
    """)

else:
    st.info("⏳ SHAP plots not found. Please run main.py first.")