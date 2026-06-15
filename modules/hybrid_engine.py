import pandas as pd
def generate_trade_signal(
        stock_path,
        sentiment_path,
        rsi_buy=30,
        rsi_sell=70,
        sentiment_pos= 0.05,
        sentiment_neg= -0.05
):
    stock_df = pd.read_csv(stock_path)
    sentiment_df = pd.read_csv(sentiment_path)

    stock_df["Date"] = pd.to_datetime(stock_df["Date"]).dt.date
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.date

    latest = stock_df.iloc[-1]

    latest_date = latest["Date"]
    senti_row = sentiment_df[sentiment_df["date"] <= latest_date]
    latest_sentiment = senti_row.iloc[-1]["sentiment_score"] if len(senti_row) else 0.0

    price_trend_up = latest["MA_20"] > latest["MA_50"]
    rsi = latest["RSI_14"]
    macd_up = latest["MACD"] > latest["MACD_Signal"]

    if price_trend_up and macd_up and rsi < rsi_sell and latest_sentiment > sentiment_pos:
        signal = "BUY"
    elif (not price_trend_up) and rsi > rsi_buy and latest_sentiment < sentiment_neg:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "date": latest_date,
        "signal": signal,
        "rsi": round(rsi, 2),
        "sentiment_score": round(float(latest_sentiment), 3),
        "ma_trend": "up" if price_trend_up else "down"
    } 