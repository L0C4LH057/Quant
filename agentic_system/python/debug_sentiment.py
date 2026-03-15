import asyncio
from datetime import datetime, timedelta
import pandas as pd
from src.data.sentiment_fetcher import fetch_news_sentiment
from src.features.sentiment_analyzer import add_sentiment_features
from src.agents.specialized.sentiment_analysis_agent import SentimentAnalysisAgent

symbol = "AAPL"
end_date_str = datetime.now().strftime("%Y-%m-%d")
start_date_str = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

print(f"Fetching news for {symbol}...")
# 1. Fetch
try:
    news_dict = fetch_news_sentiment([symbol], start_date_str, end_date_str)
    news_df = news_dict.get(symbol, pd.DataFrame())
    print(f"[{symbol}] Fetched {len(news_df)} articles")
except Exception as e:
    print(f"FETCH FAILED: {e}")
    news_df = pd.DataFrame()

if len(news_df) > 0:
    print("Generating Features...")
    # 2. Features
    # Create a dummy df_copy
    df_copy = pd.DataFrame({"close": [150.0], "date": [pd.Timestamp.now()]})
    enriched = add_sentiment_features(df_copy, news_df)
    latest = enriched.iloc[-1].to_dict()
    print("Features:", latest)

    # 3. Agent
    agent = SentimentAnalysisAgent()
    input_data = {
        "symbol": symbol,
        "sentiment_features": latest
    }
    
    print("Running Agent Process...")
    try:
        result = asyncio.run(asyncio.wait_for(agent.process(input_data), timeout=10.0))
        print("Result:", result)
    except TimeoutError:
        print("AGENT PROCESS TIMED OUT!")
    except Exception as e:
        print(f"AGENT PROCESS FAILED: {e}")
