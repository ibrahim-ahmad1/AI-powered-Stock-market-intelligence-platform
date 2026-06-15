from modules.stock_data import fetch_stock_price
from datetime import datetime 
import os
STOCK_PATH = "data/raw/stocks_price_data.csv"

if not os.path.exists(STOCK_PATH):
    SYMBOL = "TCS.NS"
    START_DATE = "2015-01-01"
    END_DATE = datetime.now().strftime("%Y-%m-%d")

    price_data = fetch_stock_price(SYMBOL, START_DATE, END_DATE)
    price_data.to_csv("data/raw/stocks_price_data.csv", index=False)
    print("Stock price data fetched and saved successfully.")
else:
    print("Stock price data already exists.")
from modules.news_data import fetch_stock_news

NEWS_PATH = "data/raw/stocks_news_data.csv"

if not os.path.exists(NEWS_PATH):
    COMPANY_NAME = "Tata Consultancy Services"
    news_data = fetch_stock_news(COMPANY_NAME)
    news_data.to_csv("data/raw/stocks_news_data.csv", index=False)
    print("Stock news data fetched and saved successfully.") 
else:
    print("Stock news data already exists.")
from modules.preprocessing import preprocess_stock_data, preprocess_news_data
RAW_STOCK_PATH = "data/raw/stocks_price_data.csv"
RAW_NEWS_PATH = "data/raw/stocks_news_data.csv"
PROCESSED_STOCK_PATH = "data/processed/processed_stocks_price_data.csv"
PROCESSED_NEWS_PATH = "data/processed/processed_stocks_news_data.csv"

if not os.path.exists(PROCESSED_STOCK_PATH):
    preprocess_stock_data(RAW_STOCK_PATH, PROCESSED_STOCK_PATH)
    print("Stock price data preprocessed and saved successfully.")
else:
    print("Processed stock price data already exists.")
if not os.path.exists(PROCESSED_NEWS_PATH):
    preprocess_news_data(RAW_NEWS_PATH, PROCESSED_NEWS_PATH)
    print("Stock news data preprocessed and saved successfully.")
else:
    print("Processed stock news data already exists.")

from modules.technical_indicators import add_technical_indicators

INPUT_PATH = "data/processed/processed_stocks_price_data.csv"
OUTPUT_PATH = "data/processed/stocks_with_indicators.csv"
if not os.path.exists(OUTPUT_PATH):
    add_technical_indicators(INPUT_PATH, OUTPUT_PATH)
    print("Technical indicators added successfully.")
else:
    print("Data with technical indicators already exists.")

from modules.price_prediction import train_lstm_model

LSTM_INPUT_PATH = "data/processed/stocks_with_indicators.csv"
LSTM_MODEL_PATH = "models/lstm_model.h5"
LSTM_SCALER_PATH = "models/scaler.pkl"
if not os.path.exists(LSTM_MODEL_PATH) or not os.path.exists(LSTM_SCALER_PATH):
    train_lstm_model(LSTM_INPUT_PATH, LSTM_MODEL_PATH, LSTM_SCALER_PATH)
    print("LSTM model trained and saved successfully.")
else:
    print("LSTM model and scaler already exist.")
from modules.evaluation import calculate_rmse
DATA_PATH = "data/processed/stocks_with_indicators.csv"
MODEL_PATH = "models/lstm_model.h5"
SCALER_PATH = "models/scaler.pkl"
rmse = calculate_rmse(DATA_PATH, MODEL_PATH, SCALER_PATH)
print(f"RMSE of the LSTM model: {rmse}")

from modules.sentiment_analysis import analyze_news_sentiment
NEWS_INPUT_PATH = "data/processed/processed_stocks_news_data.csv"
DAILY_SENTIMENT_PATH = "data/processed/daily_news_sentiment.csv"

if not os.path.exists(DAILY_SENTIMENT_PATH):
    analyze_news_sentiment(NEWS_INPUT_PATH, DAILY_SENTIMENT_PATH)
    print("News sentiment analysis completed and saved successfully.")
else:
    print("Daily news sentiment data already exists.")

from modules.hybrid_engine import generate_trade_signal
SENTIMENT_PATH = "data/processed/daily_news_sentiment.csv"
STOCK_WITH_INDICATORS = "data/processed/stocks_with_indicators.csv"
decision = generate_trade_signal(
    STOCK_WITH_INDICATORS,
    SENTIMENT_PATH
)
print("\n HYBRID TRADING DECISION")
print(f"Date: {decision['date']}")
print(f"Signal: {decision['signal']}")
print(f"RSI: {decision['rsi']}")
print(f"Sentiment Score: {decision['sentiment_score']}")
print(f"Trend: {decision['ma_trend']}")

from modules.risk_analysis import analyze_risk
RISK_INPUT_PATH = "data/processed/stocks_with_indicators.csv"
risk_info = analyze_risk(RISK_INPUT_PATH)
print("\n RISK ANALYSIS REPORT")
print(f"Volatility: {risk_info['volatility']}")
print(f"Max Drawdown: {risk_info['max_drawdown']}")
print(f"Risk Level: {risk_info['risk_level']}")

# STEP 5B: Train Transformer Model
from modules.transformer_model import train_transformer_model
INDICATORS_PATH = "data/processed/stocks_with_indicators.csv"
TRANSFORMER_MODEL_PATH = "models/transformer_model.keras"
TRANSFORMER_SCALER_PATH = "models/transformer_scaler.pkl"
if not os.path.exists(TRANSFORMER_MODEL_PATH):
    print("\n[5B] Training Transformer model...")
    train_transformer_model(INDICATORS_PATH, TRANSFORMER_MODEL_PATH, TRANSFORMER_SCALER_PATH)
    print("✅ Transformer model trained and saved.")
else:
    print("\n[5B] Transformer model already exists. Skipping.")

    # STEP 9: SHAP Explainability
from modules.shap_explainability import explain_model_shap

print("\n[9] Running SHAP Explainability...")
shap_results = explain_model_shap(
    INDICATORS_PATH,
    LSTM_MODEL_PATH,
    LSTM_SCALER_PATH
)
print(f"🏆 Top 5 Influential Days: {shap_results['top3_days']}")
print("✅ SHAP analysis complete!")