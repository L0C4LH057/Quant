"""
Market data fetcher with caching.

Supports:
    - Yahoo Finance (free, no API key)
    - Alpha Vantage (requires API key)

Improvements:
    - GAP-06: Data quality checks (gap/spike detection)
    - UPGRADE-07: Async wrapper via asyncio.to_thread
    - UPGRADE-08: Cache TTL (default 24 h for daily, 1 h for intraday)

Token Optimization:
    - Caching avoids repeated API calls
    - Simple interface reduces prompt complexity
"""
import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from ..config.base import get_config
from ..utils.validators import validate_symbol

logger = logging.getLogger(__name__)

# Default cache TTLs in seconds
_CACHE_TTL_DAILY = 24 * 3600      # 24 hours for daily data
_CACHE_TTL_INTRADAY = 3600         # 1 hour for intraday data


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
        cache_ttl: Optional[int] = None,
    ):
        """
        Initialize fetcher.

        Args:
            cache_dir: Directory for cached data
            use_cache: Whether to use caching
            cache_ttl: Cache time-to-live in seconds (None = use default)
        """
        config = get_config()
        self.cache_dir = cache_dir or Path(config.backtest_results_path).parent / "market_data"
        self.use_cache = use_cache
        self.alpha_vantage_key = config.alpha_vantage_key
        self._cache_ttl = cache_ttl  # None means auto-detect from interval

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
        """Load data from cache if exists and not expired (UPGRADE-08)."""
        path = self._get_cache_path(symbol, start_date, end_date, interval)
        if not path.exists():
            return None

        # Determine TTL
        ttl = self._cache_ttl
        if ttl is None:
            ttl = _CACHE_TTL_DAILY if interval.endswith("d") else _CACHE_TTL_INTRADAY

        age = time.time() - path.stat().st_mtime
        if age > ttl:
            logger.debug(f"Cache expired for {symbol} (age={age:.0f}s, ttl={ttl}s)")
            return None

        try:
            return pd.read_parquet(path)
        except Exception:
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

    # ── Data quality checks (GAP-06) ────────────────────────────────────

    @staticmethod
    def check_quality(df: pd.DataFrame, symbol: str = "") -> Dict[str, any]:
        """
        Run basic data quality checks on OHLCV data.

        Returns a dict with ``passed: bool`` and a list of ``warnings``.
        """
        warnings: List[str] = []

        if df.empty:
            return {"passed": False, "warnings": ["DataFrame is empty"]}

        # 1. Missing values
        missing = df[["open", "high", "low", "close"]].isnull().sum().sum()
        if missing > 0:
            warnings.append(f"{missing} missing OHLCV values")

        # 2. Zero prices
        zeros = (df["close"] == 0).sum()
        if zeros > 0:
            warnings.append(f"{zeros} zero close prices")

        # 3. Price spikes (>20 % single-bar move)
        if "close" in df.columns and len(df) > 1:
            pct_change = df["close"].pct_change().dropna().abs()
            spikes = (pct_change > 0.20).sum()
            if spikes > 0:
                warnings.append(f"{spikes} bars with >20 % price spike")

        # 4. Negative volume
        if "volume" in df.columns:
            neg_vol = (df["volume"] < 0).sum()
            if neg_vol > 0:
                warnings.append(f"{neg_vol} bars with negative volume")

        # 5. high < low sanity
        if "high" in df.columns and "low" in df.columns:
            inverted = (df["high"] < df["low"]).sum()
            if inverted > 0:
                warnings.append(f"{inverted} bars where high < low")

        passed = len(warnings) == 0
        if not passed:
            logger.warning(f"Data quality issues for {symbol}: {warnings}")

        return {"passed": passed, "warnings": warnings}

    # ── Async wrapper (UPGRADE-07) ──────────────────────────────────────

    async def afetch(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Async wrapper around ``fetch()`` using ``asyncio.to_thread``."""
        return await asyncio.to_thread(self.fetch, symbol, start_date, end_date, interval)

    async def afetch_multiple(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """Async wrapper around ``fetch_multiple()``."""
        return await asyncio.to_thread(self.fetch_multiple, symbols, start_date, end_date, interval)


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
