"""
Tests for RL agents: A2C, TD3, DQN.

Tests initialization, prediction, and short training runs for each agent.
"""
import pytest
import numpy as np
import pandas as pd
import gymnasium as gym


@pytest.fixture
def sample_data():
    """Create sample market data for testing."""
    np.random.seed(42)
    n_days = 200

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
def continuous_env(sample_data):
    """Create continuous trading environment."""
    from src.environments.trading_env import TradingEnv
    return TradingEnv(sample_data, initial_balance=100000)


@pytest.fixture
def discrete_env(sample_data):
    """Create discrete trading environment."""
    from src.environments.discrete_trading_env import DiscreteTradingEnv
    return DiscreteTradingEnv(sample_data, initial_balance=100000)


class TestA2CAgent:
    """Test A2C agent."""

    def test_initialization(self, continuous_env):
        """Test A2C agent initializes correctly."""
        from src.agents.rl_agents.a2c_agent import A2CAgent

        agent = A2CAgent(continuous_env)
        assert agent.model is not None

    def test_predict(self, continuous_env):
        """Test A2C prediction."""
        from src.agents.rl_agents.a2c_agent import A2CAgent

        agent = A2CAgent(continuous_env)
        obs, _ = continuous_env.reset()
        action = agent.predict(obs)

        assert isinstance(action, np.ndarray)
        assert action.shape == (1,)

    def test_training_smoke(self, continuous_env):
        """Test short training run completes."""
        from src.agents.rl_agents.a2c_agent import A2CAgent

        agent = A2CAgent(continuous_env)
        agent.train(total_timesteps=500)

        # Verify model can predict after training
        obs, _ = continuous_env.reset()
        action = agent.predict(obs)
        assert isinstance(action, np.ndarray)


class TestTD3Agent:
    """Test TD3 agent."""

    def test_initialization(self, continuous_env):
        """Test TD3 agent initializes correctly."""
        from src.agents.rl_agents.td3_agent import TD3Agent

        agent = TD3Agent(continuous_env)
        assert agent.model is not None

    def test_predict(self, continuous_env):
        """Test TD3 prediction."""
        from src.agents.rl_agents.td3_agent import TD3Agent

        agent = TD3Agent(continuous_env)
        obs, _ = continuous_env.reset()
        action = agent.predict(obs)

        assert isinstance(action, np.ndarray)
        assert action.shape == (1,)

    def test_training_smoke(self, continuous_env):
        """Test short training run completes."""
        from src.agents.rl_agents.td3_agent import TD3Agent

        agent = TD3Agent(continuous_env)
        agent.train(total_timesteps=500)

        obs, _ = continuous_env.reset()
        action = agent.predict(obs)
        assert isinstance(action, np.ndarray)


class TestDQNAgent:
    """Test DQN agent."""

    def test_initialization(self, discrete_env):
        """Test DQN agent initializes correctly."""
        from src.agents.rl_agents.dqn_agent import DQNAgent

        agent = DQNAgent(discrete_env)
        assert agent.model is not None

    def test_requires_discrete_env(self, continuous_env):
        """Test DQN rejects continuous action space."""
        from src.agents.rl_agents.dqn_agent import DQNAgent

        with pytest.raises(ValueError, match="Discrete"):
            DQNAgent(continuous_env)

    def test_predict(self, discrete_env):
        """Test DQN prediction."""
        from src.agents.rl_agents.dqn_agent import DQNAgent

        agent = DQNAgent(discrete_env)
        obs, _ = discrete_env.reset()
        action = agent.predict(obs)

        assert isinstance(action, np.ndarray)

    def test_training_smoke(self, discrete_env):
        """Test short training run completes."""
        from src.agents.rl_agents.dqn_agent import DQNAgent

        agent = DQNAgent(discrete_env, learning_starts=100)
        agent.train(total_timesteps=500)

        obs, _ = discrete_env.reset()
        action = agent.predict(obs)
        assert isinstance(action, np.ndarray)
