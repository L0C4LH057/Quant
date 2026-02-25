"""
Technical indicators using pandas-ta.

No TA-Lib compilation required - uses pure Python pandas-ta library.

Token Optimization:
    - Each function is self-contained
    - Clear function names = shorter prompts
    - Validation prevents errors = less debugging
"""
import logging
from typing import List, Optional

import pandas as pd
import pandas_ta as ta

from ..utils.validators import validate_dataframe

logger = logging.getLogger(__name__)


def add_sma(
    df: pd.DataFrame,
    period: int = 20,
    column: str = "close",
) -> pd.DataFrame:
    """
    Add Simple Moving Average.

    Args:
        df: DataFrame with OHLCV data
        period: Lookback period
        column: Column to calculate on

    Returns:
        DataFrame with new 'sma_{period}' column

    Raises:
        ValueError: If insufficient data

    Example:
        >>> df = add_sma(df, period=20)
        >>> print(df["sma_20"].iloc[-1])
    """
    df = validate_dataframe(df, required_columns=[column], min_rows=period)
    df = df.copy()

    df[f"sma_{period}"] = ta.sma(df[column], length=period)
    return df


def add_ema(
    df: pd.DataFrame,
    period: int = 12,
    column: str = "close",
) -> pd.DataFrame:
    """
    Add Exponential Moving Average.

    Args:
        df: DataFrame with OHLCV data
        period: Lookback period
        column: Column to calculate on

    Returns:
        DataFrame with new 'ema_{period}' column
    """
    df = validate_dataframe(df, required_columns=[column], min_rows=period)
    df = df.copy()

    df[f"ema_{period}"] = ta.ema(df[column], length=period)
    return df


def add_rsi(
    df: pd.DataFrame,
    period: int = 14,
    column: str = "close",
) -> pd.DataFrame:
    """
    Add Relative Strength Index.

    RSI ranges from 0 to 100:
        - >70: Overbought
        - <30: Oversold

    Args:
        df: DataFrame with OHLCV data
        period: Lookback period
        column: Column to calculate on

    Returns:
        DataFrame with new 'rsi_{period}' column
    """
    df = validate_dataframe(df, required_columns=[column], min_rows=period + 1)
    df = df.copy()

    df[f"rsi_{period}"] = ta.rsi(df[column], length=period)
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "close",
) -> pd.DataFrame:
    """
    Add MACD indicator.

    Adds three columns:
        - macd: MACD line
        - macd_signal: Signal line
        - macd_hist: Histogram (MACD - Signal)

    Args:
        df: DataFrame with OHLCV data
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
        column: Column to calculate on

    Returns:
        DataFrame with MACD columns
    """
    df = validate_dataframe(df, required_columns=[column], min_rows=slow + signal)
    df = df.copy()

    macd_result = ta.macd(df[column], fast=fast, slow=slow, signal=signal)

    if macd_result is not None:
        df["macd"] = macd_result.iloc[:, 0]
        df["macd_signal"] = macd_result.iloc[:, 2]
        df["macd_hist"] = macd_result.iloc[:, 1]

    return df


def add_bollinger(
    df: pd.DataFrame,
    period: int = 20,
    std: float = 2.0,
    column: str = "close",
) -> pd.DataFrame:
    """
    Add Bollinger Bands.

    Adds three columns:
        - bb_upper: Upper band
        - bb_middle: Middle band (SMA)
        - bb_lower: Lower band

    Args:
        df: DataFrame with OHLCV data
        period: Lookback period
        std: Number of standard deviations
        column: Column to calculate on

    Returns:
        DataFrame with Bollinger Band columns
    """
    df = validate_dataframe(df, required_columns=[column], min_rows=period)
    df = df.copy()

    bb_result = ta.bbands(df[column], length=period, std=std)

    if bb_result is not None:
        df["bb_lower"] = bb_result.iloc[:, 0]
        df["bb_middle"] = bb_result.iloc[:, 1]
        df["bb_upper"] = bb_result.iloc[:, 2]

    return df


def add_atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """
    Add Average True Range (volatility indicator).

    Args:
        df: DataFrame with OHLCV data (requires high, low, close)
        period: Lookback period

    Returns:
        DataFrame with new 'atr_{period}' column
    """
    df = validate_dataframe(
        df, required_columns=["high", "low", "close"], min_rows=period + 1
    )
    df = df.copy()

    df[f"atr_{period}"] = ta.atr(df["high"], df["low"], df["close"], length=period)
    return df


def add_all_indicators(
    df: pd.DataFrame,
    sma_periods: Optional[List[int]] = None,
    ema_periods: Optional[List[int]] = None,
    rsi_period: int = 14,
    include_atr: bool = True,
    include_macd: bool = True,
    include_bollinger: bool = True,
) -> pd.DataFrame:
    """
    Add all standard technical indicators.

    Args:
        df: DataFrame with OHLCV data
        sma_periods: List of SMA periods (default: [20, 50])
        ema_periods: List of EMA periods (default: [12])
        rsi_period: RSI period
        include_atr: Include ATR indicator
        include_macd: Include MACD indicator
        include_bollinger: Include Bollinger Bands

    Returns:
        DataFrame with all indicators added

    Example:
        >>> df = add_all_indicators(df)
        >>> print(df.columns)
        ['date', 'open', 'high', 'low', 'close', 'volume',
         'sma_20', 'sma_50', 'ema_12', 'rsi_14', 'macd', ...]
    """
    if sma_periods is None:
        sma_periods = [20, 50]
    if ema_periods is None:
        ema_periods = [12]

    df = validate_dataframe(df, required_columns=["close"])
    df = df.copy()

    # SMAs
    for period in sma_periods:
        if len(df) >= period:
            df = add_sma(df, period=period)

    # EMAs
    for period in ema_periods:
        if len(df) >= period:
            df = add_ema(df, period=period)

    # RSI
    if len(df) >= rsi_period + 1:
        df = add_rsi(df, period=rsi_period)

    # MACD
    if include_macd and len(df) >= 35:  # 26 + 9
        df = add_macd(df)

    # Bollinger Bands
    if include_bollinger and len(df) >= 20:
        df = add_bollinger(df)

    # ATR
    if include_atr and "high" in df.columns and "low" in df.columns:
        if len(df) >= 15:
            df = add_atr(df, period=14)

    # Drop rows with NaN from indicator calculation
    initial_len = len(df)
    df = df.dropna()
    dropped = initial_len - len(df)

    if dropped > 0:
        logger.info(f"Dropped {dropped} rows with NaN from indicator calculation")

    return df
