import pandas as pd
import os 

def preprocess_stock_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values("Date")
    df = df.dropna()
  
    if "Adj Close" in df.columns:
        df = df.drop(columns=["Adj Close"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    return df

import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]","", text)
    return text
def preprocess_news_data(input_path, output_path):  
    df = pd.read_csv(input_path)
    df = df.dropna(subset=["headline", "description"])
    df["text"] = df["headline"] + " " + df["description"]
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[["date", "clean_text", "source"]]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df