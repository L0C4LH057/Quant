"""
Pytest configuration and fixtures.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    np.random.seed(42)
    n_days = 100

    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")
    close_prices = 100 + np.cumsum(np.random.randn(n_days) * 0.5)

    df = pd.DataFrame({
        "date": dates,
        "open": close_prices + np.random.randn(n_days) * 0.2,
        "high": close_prices + np.abs(np.random.randn(n_days) * 0.3),
        "low": close_prices - np.abs(np.random.randn(n_days) * 0.3),
        "close": close_prices,
        "volume": np.random.randint(1000, 10000, n_days),
    })

    return df


@pytest.fixture
def sample_indicators(sample_market_data):
    """Sample data with technical indicators."""
    from src.features.technical_indicators import add_all_indicators
    return add_all_indicators(sample_market_data)


@pytest.fixture
def trading_env(sample_indicators):
    """Create a trading environment for testing."""
    from src.environments.trading_env import TradingEnv
    return TradingEnv(sample_indicators, initial_balance=100000)
