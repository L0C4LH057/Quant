"""
Tests for DiscreteTradingEnv.
"""
import pytest
import numpy as np
import pandas as pd
import gymnasium as gym


@pytest.fixture
def sample_data():
    """Create sample market data."""
    np.random.seed(42)
    n_days = 100

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


class TestDiscreteTradingEnv:
    """Test discrete action trading environment."""

    def test_action_space_is_discrete(self, sample_data):
        """Test that action space is Discrete(3)."""
        from src.environments.discrete_trading_env import DiscreteTradingEnv

        env = DiscreteTradingEnv(sample_data)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 3

    def test_reset(self, sample_data):
        """Test reset returns valid observation."""
        from src.environments.discrete_trading_env import DiscreteTradingEnv

        env = DiscreteTradingEnv(sample_data)
        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape

    def test_buy_action(self, sample_data):
        """Test buy action (action=2)."""
        from src.environments.discrete_trading_env import DiscreteTradingEnv

        env = DiscreteTradingEnv(sample_data)
        env.reset()

        obs, reward, _, _, info = env.step(2)  # Buy

        assert env.holdings > 0
        assert env.balance < env.initial_balance

    def test_hold_action(self, sample_data):
        """Test hold action (action=1)."""
        from src.environments.discrete_trading_env import DiscreteTradingEnv

        env = DiscreteTradingEnv(sample_data)
        env.reset()

        initial_balance = env.balance
        obs, reward, _, _, info = env.step(1)  # Hold

        assert env.holdings == 0
        assert env.balance == initial_balance

    def test_sell_after_buy(self, sample_data):
        """Test sell action (action=0) after buying."""
        from src.environments.discrete_trading_env import DiscreteTradingEnv

        env = DiscreteTradingEnv(sample_data)
        env.reset()

        env.step(2)  # Buy
        assert env.holdings > 0

        env.step(0)  # Sell
        assert env.holdings == pytest.approx(0, abs=1e-10)

    def test_invalid_action_raises(self, sample_data):
        """Test invalid action raises ValueError."""
        from src.environments.discrete_trading_env import DiscreteTradingEnv

        env = DiscreteTradingEnv(sample_data)
        env.reset()

        with pytest.raises(ValueError, match="Invalid action"):
            env.step(5)

    def test_episode_completes(self, sample_data):
        """Test episode runs to completion with random discrete actions."""
        from src.environments.discrete_trading_env import DiscreteTradingEnv

        env = DiscreteTradingEnv(sample_data)
        obs, _ = env.reset()

        done = False
        steps = 0

        while not done and steps < 200:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        assert done

    def test_dqn_compatibility(self, sample_data):
        """Test DQN can train on discrete env."""
        from src.environments.discrete_trading_env import DiscreteTradingEnv
        from src.agents.rl_agents.dqn_agent import DQNAgent

        env = DiscreteTradingEnv(sample_data)
        agent = DQNAgent(env, learning_starts=50)
        agent.train(total_timesteps=200)

        obs, _ = env.reset()
        action = agent.predict(obs)
        assert isinstance(action, np.ndarray)
