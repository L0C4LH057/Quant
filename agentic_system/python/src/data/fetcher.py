"""
Market data fetcher with caching.

Supports:
    - Yahoo Finance (free, no API key)
    - Alpha Vantage (requires API key)

Token Optimization:
    - Caching avoids repeated API calls
    - Simple interface reduces prompt complexity
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from ..config.base import get_config
from ..utils.validators import validate_symbol

logger = logging.getLogger(__name__)


class MarketDataFetcher:
    """
    Fetch market data from various sources.

    Supports Yahoo Finance (free) and Alpha Vantage (API key required).
    Includes caching to avoid repeated API calls.

    Example:
        >>> fetcher = MarketDataFetcher()
        >>> df = fetcher.fetch("EURUSD=X", "2023-01-01", "2024-01-01")
        >>> print(df.columns)
        ['open', 'high', 'low', 'close', 'volume']
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
    ):
        """
        Initialize fetcher.

        Args:
            cache_dir: Directory for cached data
            use_cache: Whether to use caching
        """
        config = get_config()
        self.cache_dir = cache_dir or Path(config.backtest_results_path).parent / "market_data"
        self.use_cache = use_cache
        self.alpha_vantage_key = config.alpha_vantage_key

        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Trading symbol (e.g., "EURUSD=X", "AAPL")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1d, 1h, 15m, etc.)

        Returns:
            DataFrame with columns: date, open, high, low, close, volume

        Raises:
            ValueError: If data cannot be fetched
        """
        symbol = validate_symbol(symbol)

        # Check cache first
        if self.use_cache:
            cached = self._load_from_cache(symbol, start_date, end_date, interval)
            if cached is not None:
                logger.info(f"Loaded {symbol} from cache")
                return cached

        # Fetch from Yahoo Finance
        df = self._fetch_yahoo(symbol, start_date, end_date, interval)

        # Save to cache
        if self.use_cache and df is not None and not df.empty:
            self._save_to_cache(df, symbol, start_date, end_date, interval)

        return df

    def fetch_multiple(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.

        Args:
            symbols: List of trading symbols
            start_date: Start date
            end_date: End date
            interval: Data interval

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        result = {}
        for symbol in symbols:
            try:
                df = self.fetch(symbol, start_date, end_date, interval)
                if df is not None and not df.empty:
                    result[symbol] = df
                else:
                    logger.warning(f"No data for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
        return result

    def _fetch_yahoo(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str,
    ) -> pd.DataFrame:
        """Fetch from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
            )

            if df.empty:
                raise ValueError(f"No data returned for {symbol}")

            # Standardize column names
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]

            # Rename Date/Datetime column
            if "datetime" in df.columns:
                df = df.rename(columns={"datetime": "date"})

            # Keep only needed columns
            columns = ["date", "open", "high", "low", "close", "volume"]
            df = df[[c for c in columns if c in df.columns]]

            logger.info(f"Fetched {len(df)} rows for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {e}")
            raise ValueError(f"Failed to fetch {symbol}: {e}")

    def _get_cache_path(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str,
    ) -> Path:
        """Get cache file path."""
        clean_symbol = symbol.replace("=", "_").replace("/", "_")
        filename = f"{clean_symbol}_{start_date}_{end_date}_{interval}.parquet"
        return self.cache_dir / filename

    def _load_from_cache(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str,
    ) -> Optional[pd.DataFrame]:
        """Load data from cache if exists."""
        path = self._get_cache_path(symbol, start_date, end_date, interval)
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception:
                return None
        return None

    def _save_to_cache(
        self,
        df: pd.DataFrame,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str,
    ) -> None:
        """Save data to cache."""
        path = self._get_cache_path(symbol, start_date, end_date, interval)
        try:
            df.to_parquet(path, index=False)
            logger.debug(f"Cached {symbol} to {path}")
        except Exception as e:
            logger.warning(f"Failed to cache {symbol}: {e}")


def fetch_market_data(
    symbols: List[str],
    start_date: str,
    end_date: str,
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to fetch market data.

    Args:
        symbols: List of trading symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval

    Returns:
        Dictionary mapping symbol to DataFrame

    Example:
        >>> data = fetch_market_data(["AAPL", "GOOGL"], "2023-01-01", "2024-01-01")
        >>> print(data["AAPL"].columns)
    """
    fetcher = MarketDataFetcher()
    return fetcher.fetch_multiple(symbols, start_date, end_date, interval)
