import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os 

def analyze_news_sentiment(input_path, output_path):
    df = pd.read_csv(input_path)
    analyzer = SentimentIntensityAnalyzer()
    def get_sentiment_score(text):
        score = analyzer.polarity_scores(text)["compound"]
        return score
        
    df["sentiment_score"] = df["clean_text"].apply(get_sentiment_score)

    def label_sentiment(score):
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"
    
    df["sentiment_label"] = df["sentiment_score"].apply(label_sentiment)

    daily_sentiment = (
        df.groupby("date")["sentiment_score"]
        .mean()
        .reset_index()
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv("data/processed/news_with_sentiment.csv", index=False)
    daily_sentiment.to_csv(output_path, index=False)

    return daily_sentiment