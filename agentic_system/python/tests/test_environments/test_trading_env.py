"""
Tests for trading environment.
"""
import pytest
import numpy as np


class TestTradingEnv:
    """Test trading environment."""

    def test_env_initialization(self, trading_env):
        """Test environment initializes correctly."""
        assert trading_env.initial_balance == 100000
        assert trading_env.current_step == 0

    def test_reset(self, trading_env):
        """Test reset returns valid observation."""
        obs, info = trading_env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == trading_env.observation_space.shape
        assert info["balance"] == trading_env.initial_balance

    def test_step_buy_action(self, trading_env):
        """Test buy action execution."""
        trading_env.reset()

        # Buy action
        obs, reward, terminated, truncated, info = trading_env.step(np.array([0.5]))

        assert trading_env.holdings > 0
        assert trading_env.balance < trading_env.initial_balance
        assert len(trading_env.trades_log) == 1

    def test_step_sell_action(self, trading_env):
        """Test sell action after buying."""
        trading_env.reset()

        # Buy first
        trading_env.step(np.array([1.0]))
        initial_holdings = trading_env.holdings

        # Then sell
        trading_env.step(np.array([-0.5]))

        assert trading_env.holdings < initial_holdings
        assert len(trading_env.trades_log) == 2

    def test_action_clipping(self, trading_env):
        """Test actions are clipped to valid range."""
        trading_env.reset()

        # Action outside range should be clipped
        obs, reward, _, _, _ = trading_env.step(np.array([2.0]))

        # Should not crash
        assert isinstance(obs, np.ndarray)

    def test_episode_completes(self, trading_env):
        """Test episode runs to completion."""
        obs, info = trading_env.reset()

        done = False
        steps = 0
        max_steps = 1000

        while not done and steps < max_steps:
            action = trading_env.action_space.sample()
            obs, reward, terminated, truncated, info = trading_env.step(action)
            done = terminated or truncated
            steps += 1

        assert done
        assert steps < max_steps

    def test_trades_log(self, trading_env):
        """Test trade logging works."""
        trading_env.reset()

        # Make some trades
        trading_env.step(np.array([0.5]))  # Buy
        trading_env.step(np.array([-0.3]))  # Sell

        df = trading_env.get_trades_df()
        assert len(df) == 2
        assert "side" in df.columns
        assert "price" in df.columns
