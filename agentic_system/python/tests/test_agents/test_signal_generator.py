"""
Tests for RLSignalGenerator.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, action_value: float):
        self.action_value = action_value

    def predict(self, observation, deterministic=True):
        return np.array([self.action_value])


@pytest.fixture
def sample_data():
    """Create sample market data with enough rows for indicators."""
    np.random.seed(42)
    n_days = 200

    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")
    close_prices = 100 + np.cumsum(np.random.randn(n_days) * 0.5)

    return pd.DataFrame({
        "date": dates,
        "open": close_prices + np.random.randn(n_days) * 0.2,
        "high": close_prices + np.abs(np.random.randn(n_days) * 0.3),
        "low": close_prices - np.abs(np.random.randn(n_days) * 0.3),
        "close": close_prices,
        "volume": np.random.randint(1000, 10000, n_days),
    })


@pytest.fixture
def buy_ensemble():
    """Create ensemble that always outputs buy signals."""
    from src.agents.rl_agents.ensemble_agent import EnsembleAgent

    ensemble = EnsembleAgent()
    ensemble.add_agent("A", MockAgent(0.8))
    ensemble.add_agent("B", MockAgent(0.7))
    ensemble.add_agent("C", MockAgent(0.9))
    return ensemble


@pytest.fixture
def hold_ensemble():
    """Create ensemble that always outputs hold signals."""
    from src.agents.rl_agents.ensemble_agent import EnsembleAgent

    ensemble = EnsembleAgent()
    ensemble.add_agent("A", MockAgent(0.05))
    ensemble.add_agent("B", MockAgent(-0.01))
    return ensemble


class TestRLSignalGenerator:
    """Test signal generation pipeline."""

    def test_initialization(self, buy_ensemble):
        """Test signal generator initializes correctly."""
        from src.agents.rl_agents.signal_generator import RLSignalGenerator

        gen = RLSignalGenerator(buy_ensemble)
        assert gen.ensemble.agent_count == 3

    def test_requires_agents(self):
        """Test initialization fails with empty ensemble."""
        from src.agents.rl_agents.ensemble_agent import EnsembleAgent
        from src.agents.rl_agents.signal_generator import RLSignalGenerator

        empty_ensemble = EnsembleAgent()
        with pytest.raises(ValueError, match="no agents"):
            RLSignalGenerator(empty_ensemble)

    def test_generate_buy_signal(self, buy_ensemble, sample_data):
        """Test buy signal generation with stop-loss and take-profit."""
        from src.agents.rl_agents.signal_generator import RLSignalGenerator

        gen = RLSignalGenerator(buy_ensemble, min_confidence=0.5)
        result = gen.generate(sample_data, symbol="EURUSD")

        assert result["symbol"] == "EURUSD"
        assert result["signal"] == "buy"
        assert result["confidence"] > 0
        assert result["current_price"] > 0

        # Stop-loss should be below current price for buy
        if result["stop_loss"] is not None:
            assert result["stop_loss"] < result["current_price"]
        # Take-profit should be above current price for buy
        if result["take_profit"] is not None:
            assert result["take_profit"] > result["current_price"]

    def test_generate_hold_signal(self, hold_ensemble, sample_data):
        """Test hold signal with no SL/TP."""
        from src.agents.rl_agents.signal_generator import RLSignalGenerator

        gen = RLSignalGenerator(hold_ensemble)
        result = gen.generate(sample_data, symbol="USDJPY")

        assert result["signal"] == "hold"
        assert result["position_size_pct"] == 0.0

    def test_result_structure(self, buy_ensemble, sample_data):
        """Test all expected fields are present."""
        from src.agents.rl_agents.signal_generator import RLSignalGenerator

        gen = RLSignalGenerator(buy_ensemble, min_confidence=0.5)
        result = gen.generate(sample_data)

        required_fields = [
            "symbol", "signal", "confidence", "current_price",
            "stop_loss", "take_profit", "position_size_pct",
            "risk_reward_ratio", "agent_votes", "indicators",
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_position_sizing(self, buy_ensemble, sample_data):
        """Test position size scales with confidence."""
        from src.agents.rl_agents.signal_generator import RLSignalGenerator

        gen = RLSignalGenerator(
            buy_ensemble,
            max_position_pct=0.10,
            min_confidence=0.5,
        )
        result = gen.generate(sample_data)

        if result["signal"] != "hold":
            assert result["position_size_pct"] > 0
            assert result["position_size_pct"] <= 0.10

    def test_insufficient_data_returns_hold(self, buy_ensemble):
        """Test insufficient data returns safe hold signal."""
        from src.agents.rl_agents.signal_generator import RLSignalGenerator

        gen = RLSignalGenerator(buy_ensemble)

        # Very small dataset
        small_df = pd.DataFrame({
            "close": [100, 101, 102],
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "volume": [1000, 1000, 1000],
        })

        result = gen.generate(small_df)
        assert result["signal"] == "hold"
