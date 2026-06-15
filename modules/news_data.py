import requests
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

def fetch_stock_news(comapny_name, max_results = 50):
    api_key = os.getenv("GNEWS_API_KEY")

    if not api_key:
        raise ValueError("GNEWS_API_KEY not found in .env file")
    
    url = "https://gnews.io/api/v4/search"

    paramas = {
        "q": comapny_name,
        "lang": "en",
        "max": max_results,
        "token": api_key
    }
    response = requests.get(url, params=paramas)
    data = response.json()
    articles =[]

    if "articles" in data:
        for article in data["articles"]:
            articles.append({
                "data": article["publishedAt"],
                "headline": article["title"],
                "description": article["description"],
                "source": article["source"]["name"],
            })
    return pd.DataFrame(articles)