import yfinance as yf
import pandas as pd

def fetch_stock_price(symbol, start_date, end_date):
    stock = yf.download(symbol, start=start_date, end=end_date)
    stock.reset_index(inplace=True)
    return stock