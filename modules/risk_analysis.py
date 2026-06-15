import pandas as pd
import numpy as np

def analyze_risk(stock_path):
    """
    Stock ka risk analysis karta hai:
    - Daily returns
    - Volatility
    - Maximum drawdown
    """

    df = pd.read_csv(stock_path)

    # Date convert
    df["Date"] = pd.to_datetime(df["Date"])

    # 1️⃣ Daily Returns
    df["Daily_Return"] = df["Close"].pct_change()

    # 2️⃣ Volatility (Standard Deviation of Returns)
    volatility = df["Daily_Return"].std()

    # 3️⃣ Maximum Drawdown
    df["Cumulative_Max"] = df["Close"].cummax()
    df["Drawdown"] = (df["Close"] - df["Cumulative_Max"]) / df["Cumulative_Max"]

    max_drawdown = df["Drawdown"].min()

    # Risk Classification (Simple rule-based)
    if volatility < 0.01:
        risk_level = "LOW"
    elif volatility < 0.03:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    return {
        "volatility": round(float(volatility), 4),
        "max_drawdown": round(float(max_drawdown), 4),
        "risk_level": risk_level
    }
