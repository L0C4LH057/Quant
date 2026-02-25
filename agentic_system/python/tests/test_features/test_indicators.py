"""
Tests for technical indicators.
"""
import pytest
import pandas as pd
import numpy as np


class TestTechnicalIndicators:
    """Test technical indicators."""

    def test_add_sma(self, sample_market_data):
        """Test SMA calculation."""
        from src.features.technical_indicators import add_sma

        df = add_sma(sample_market_data, period=20)
        assert "sma_20" in df.columns
        assert not df["sma_20"].iloc[-1:].isna().any()

    def test_add_rsi(self, sample_market_data):
        """Test RSI calculation."""
        from src.features.technical_indicators import add_rsi

        df = add_rsi(sample_market_data, period=14)
        assert "rsi_14" in df.columns

        # RSI should be between 0 and 100
        valid_rsi = df["rsi_14"].dropna()
        assert valid_rsi.min() >= 0
        assert valid_rsi.max() <= 100

    def test_add_macd(self, sample_market_data):
        """Test MACD calculation."""
        from src.features.technical_indicators import add_macd

        df = add_macd(sample_market_data)
        assert "macd" in df.columns
        assert "macd_signal" in df.columns
        assert "macd_hist" in df.columns

    def test_add_bollinger(self, sample_market_data):
        """Test Bollinger Bands."""
        from src.features.technical_indicators import add_bollinger

        df = add_bollinger(sample_market_data)
        assert "bb_upper" in df.columns
        assert "bb_middle" in df.columns
        assert "bb_lower" in df.columns

        # Upper should be above lower
        valid = df[["bb_upper", "bb_lower"]].dropna()
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()

    def test_add_all_indicators(self, sample_market_data):
        """Test adding all indicators."""
        from src.features.technical_indicators import add_all_indicators

        df = add_all_indicators(sample_market_data)

        expected = ["sma_20", "sma_50", "ema_12", "rsi_14", "macd"]
        for col in expected:
            assert col in df.columns

    def test_insufficient_data_raises(self):
        """Test that insufficient data raises error."""
        from src.features.technical_indicators import add_sma

        small_df = pd.DataFrame({"close": [1, 2, 3]})

        with pytest.raises(ValueError, match="at least"):
            add_sma(small_df, period=10)
