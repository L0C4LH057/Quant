"""
Tests for EnsembleAgent — voting, confidence, and signal generation.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock


class MockAgent:
    """Mock RL agent for testing ensemble voting."""

    def __init__(self, action_value: float):
        self.action_value = action_value

    def predict(self, observation, deterministic=True):
        return np.array([self.action_value])


@pytest.fixture
def sample_data():
    """Create sample market data."""
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


class TestEnsembleAgent:
    """Test ensemble voting logic."""

    def test_initialization(self):
        """Test ensemble initializes empty."""
        from src.agents.rl_agents.ensemble_agent import EnsembleAgent

        ensemble = EnsembleAgent()
        assert ensemble.agent_count == 0

    def test_add_agent(self):
        """Test adding agents to ensemble."""
        from src.agents.rl_agents.ensemble_agent import EnsembleAgent

        ensemble = EnsembleAgent()
        ensemble.add_agent("test", MockAgent(0.5), weight=1.0)

        assert ensemble.agent_count == 1
        assert "test" in ensemble.agent_names

    def test_remove_agent(self):
        """Test removing agents from ensemble."""
        from src.agents.rl_agents.ensemble_agent import EnsembleAgent

        ensemble = EnsembleAgent()
        ensemble.add_agent("test", MockAgent(0.5))
        ensemble.remove_agent("test")

        assert ensemble.agent_count == 0

    def test_predict_requires_agents(self):
        """Test predict raises error when no agents."""
        from src.agents.rl_agents.ensemble_agent import EnsembleAgent

        ensemble = EnsembleAgent()
        obs = np.zeros(10)

        with pytest.raises(RuntimeError, match="No agents"):
            ensemble.predict(obs)

    def test_unanimous_buy(self):
        """Test unanimous buy signal → high confidence."""
        from src.agents.rl_agents.ensemble_agent import EnsembleAgent

        ensemble = EnsembleAgent()
        ensemble.add_agent("A", MockAgent(0.8))
        ensemble.add_agent("B", MockAgent(0.6))
        ensemble.add_agent("C", MockAgent(0.9))

        result = ensemble.predict(np.zeros(10))

        assert result["signal"] == "buy"
        assert result["confidence"] == 1.0
        assert result["agreement_pct"] == 1.0

    def test_unanimous_sell(self):
        """Test unanimous sell signal."""
        from src.agents.rl_agents.ensemble_agent import EnsembleAgent

        ensemble = EnsembleAgent()
        ensemble.add_agent("A", MockAgent(-0.8))
        ensemble.add_agent("B", MockAgent(-0.6))
        ensemble.add_agent("C", MockAgent(-0.9))

        result = ensemble.predict(np.zeros(10))

        assert result["signal"] == "sell"
        assert result["confidence"] == 1.0

    def test_majority_vote(self):
        """Test majority voting: 3 buy, 2 sell → buy."""
        from src.agents.rl_agents.ensemble_agent import EnsembleAgent

        ensemble = EnsembleAgent()
        ensemble.add_agent("A", MockAgent(0.5))   # buy
        ensemble.add_agent("B", MockAgent(0.7))   # buy
        ensemble.add_agent("C", MockAgent(0.3))   # buy
        ensemble.add_agent("D", MockAgent(-0.5))  # sell
        ensemble.add_agent("E", MockAgent(-0.7))  # sell

        result = ensemble.predict(np.zeros(10))

        assert result["signal"] == "buy"
        assert result["agreement_pct"] == pytest.approx(0.6, abs=0.01)

    def test_hold_when_uncertain(self):
        """Test hold signal when near-zero actions."""
        from src.agents.rl_agents.ensemble_agent import EnsembleAgent

        ensemble = EnsembleAgent()
        ensemble.add_agent("A", MockAgent(0.05))   # hold
        ensemble.add_agent("B", MockAgent(-0.01))  # hold
        ensemble.add_agent("C", MockAgent(0.02))   # hold

        result = ensemble.predict(np.zeros(10))

        assert result["signal"] == "hold"

    def test_weighted_voting(self):
        """Test that higher weights influence outcome."""
        from src.agents.rl_agents.ensemble_agent import EnsembleAgent

        ensemble = EnsembleAgent()
        # 1 agent says buy with weight 10
        ensemble.add_agent("heavy", MockAgent(0.8), weight=10.0)
        # 2 agents say sell with weight 1 each
        ensemble.add_agent("light1", MockAgent(-0.8), weight=1.0)
        ensemble.add_agent("light2", MockAgent(-0.8), weight=1.0)

        result = ensemble.predict(np.zeros(10))

        assert result["signal"] == "buy"  # heavy agent dominates

    def test_result_structure(self):
        """Test that result contains all expected fields."""
        from src.agents.rl_agents.ensemble_agent import EnsembleAgent

        ensemble = EnsembleAgent()
        ensemble.add_agent("A", MockAgent(0.5))

        result = ensemble.predict(np.zeros(10))

        assert "signal" in result
        assert "confidence" in result
        assert "raw_action" in result
        assert "agent_votes" in result
        assert "agreement_pct" in result

    def test_agent_votes_detail(self):
        """Test that per-agent votes are recorded."""
        from src.agents.rl_agents.ensemble_agent import EnsembleAgent

        ensemble = EnsembleAgent()
        ensemble.add_agent("PPO", MockAgent(0.5))
        ensemble.add_agent("SAC", MockAgent(-0.3))

        result = ensemble.predict(np.zeros(10))

        assert "PPO" in result["agent_votes"]
        assert "SAC" in result["agent_votes"]
        assert result["agent_votes"]["PPO"]["vote"] == "buy"
        assert result["agent_votes"]["SAC"]["vote"] == "sell"


class TestEnsembleWithRealAgents:
    """Integration test with real SB3 agents (short training)."""

    def test_ensemble_with_ppo_sac(self, sample_data):
        """Test ensemble with real PPO and SAC agents."""
        from src.environments.trading_env import TradingEnv
        from src.agents.rl_agents.ppo_agent import PPOAgent
        from src.agents.rl_agents.sac_agent import SACAgent
        from src.agents.rl_agents.ensemble_agent import EnsembleAgent

        env = TradingEnv(sample_data, initial_balance=100000)

        # Train minimal agents
        ppo = PPOAgent(env)
        ppo.train(total_timesteps=500)

        sac = SACAgent(env)
        sac.train(total_timesteps=500)

        # Create ensemble
        ensemble = EnsembleAgent()
        ensemble.add_agent("PPO", ppo)
        ensemble.add_agent("SAC", sac)

        # Test prediction
        obs, _ = env.reset()
        result = ensemble.predict(obs)

        assert result["signal"] in ("buy", "sell", "hold")
        assert 0 <= result["confidence"] <= 1
