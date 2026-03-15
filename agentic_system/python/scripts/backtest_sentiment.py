"""
Backtest trading strategies using historical market data enriched with news sentiment.

This script demonstrates how to:
1. Load historical market data.
2. Fetch corresponding news headlines for the same timeframe.
3. Compute sentiment features (VADER or FinBERT).
4. Run a backtest using an RL agent or rule-based strategy on the enriched data.

Usage:
    python scripts/backtest_sentiment.py --symbol AAPL --start 2023-01-01 --end 2023-12-31
"""
import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from stable_baselines3 import PPO

from src.backtesting import BacktestEngine, BacktestConfig, BacktestReporter
from src.environments.trading_env import TradingEnv
from src.data.sentiment_fetcher import fetch_news_sentiment
from src.features.sentiment_analyzer import add_sentiment_features
from src.features.technical_indicators import add_all_indicators

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_dummy_market_data(start: str, end: str) -> pd.DataFrame:
    """Generate synthetic market data if we don't have a real data source wired up yet."""
    import numpy as np
    
    dates = pd.date_range(start, end, freq="B") # Business days
    n_days = len(dates)
    
    np.random.seed(42)
    returns = np.random.randn(n_days) * 0.02 + 0.0003
    price = 150 * np.cumprod(1 + returns)
    
    high = price * (1 + np.abs(np.random.randn(n_days)) * 0.01)
    low = price * (1 - np.abs(np.random.randn(n_days)) * 0.01)
    volume = np.random.randint(5000000, 20000000, n_days)
    
    df = pd.DataFrame({
        "date": dates,
        "open": price * (1 + np.random.randn(n_days) * 0.005),
        "high": high,
        "low": low,
        "close": price,
        "volume": volume,
    })
    return df


def run_sentiment_backtest(
    symbol: str, 
    start_date: str, 
    end_date: str, 
    model_type: str = "vader"
):
    logger.info("=" * 60)
    logger.info(f"Sentiment Backtest for {symbol} ({start_date} to {end_date})")
    logger.info(f"Using sentiment model: {model_type.upper()}")
    logger.info("=" * 60)

    # 1. Load Market Data
    logger.info("1. Loading historical market data...")
    # NOTE: In a fully wired system, we would use MarketDataFetcher here.
    # For demonstration, we'll use synthetic data.
    market_df = generate_dummy_market_data(start_date, end_date)
    
    # Add standard technical indicators
    market_df = add_all_indicators(market_df)
    
    # 2. Fetch News Data
    logger.info(f"2. Fetching news data for {symbol}...")
    # This uses NewsSentimentFetcher with caching
    news_dict = fetch_news_sentiment([symbol], start_date, end_date)
    news_df = news_dict.get(symbol, pd.DataFrame())
    
    if news_df.empty:
        logger.warning(f"No news data found for {symbol} in this date range.")
    else:
        logger.info(f"Found {len(news_df)} news articles.")
    
    # 3. Add Sentiment Features
    logger.info("3. Engineering sentiment features...")
    enriched_df = add_sentiment_features(
        market_df, 
        news_df, 
        lookback_windows=[1, 3, 7], 
        model=model_type
    )
    
    # Set index for the environment
    enriched_df.set_index("date", inplace=True)
    enriched_df = enriched_df.dropna() # Drop rows with NaNs from indicators
    
    logger.info(f"Data prepared: {len(enriched_df)} trading days available.")
    
    # 4. Train a quick model & Backtest
    logger.info("4. Setting up TradingEnvironment and BacktestEngine...")
    
    def env_factory(df):
        return TradingEnv(
            df=df,
            initial_balance=100_000,
            window_size=20,
        )

    # Splitting data for train/test Walk-forward
    train_size = int(len(enriched_df) * 0.7)
    train_data = enriched_df.iloc[:train_size]
    test_data = enriched_df.iloc[train_size:]
    
    if len(train_data) < 50 or len(test_data) < 20:
         logger.error("Not enough data to run a meaningful train/test split.")
         return
         
    logger.info("Training PPO model on enriched data (using sentiment observations)...")
    train_env = env_factory(train_data)
    model = PPO("MlpPolicy", train_env, verbose=0)
    model.learn(total_timesteps=5000)
    train_env.close()
    
    logger.info("Running backtest on test data fold...")
    engine = BacktestEngine(
        model=model,
        env_factory=env_factory,
        config=BacktestConfig(initial_balance=100_000),
    )
    
    result = engine.run(test_data, symbols=[symbol])
    
    # 5. Review Results
    logger.info(f"Backtest Results:")
    logger.info(f"  Total Return: {result.metrics.total_return:.2%}")
    logger.info(f"  Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
    logger.info(f"  Max Drawdown: {result.metrics.max_drawdown:.2%}")
    logger.info(f"  Win Rate: {result.metrics.win_rate:.1%}")
    logger.info(f"  Total Trades: {result.metrics.total_trades}")
    
    # Generate HTML report
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / f"sentiment_backtest_{symbol}_{model_type}.html"
    reporter = BacktestReporter(result, title=f"Sentiment Enriched Backtest - {symbol}")
    reporter.save_html(report_path)
    
    logger.info(f"  Report saved to: {report_path}")
    reporter.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sentiment-enriched backtest")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Trading symbol")
    parser.add_argument("--start", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2023-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--model", type=str, choices=["vader", "finbert"], default="vader", 
                        help="Sentiment model to use (vader or finbert)")
    
    args = parser.parse_args()
    
    run_sentiment_backtest(args.symbol, args.start, args.end, args.model)
