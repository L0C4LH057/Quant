"""
Tests for configuration module.
"""
import os
import pytest


class TestFinRLConfig:
    """Test FinRL configuration."""

    def test_default_config_valid(self):
        """Test default configuration is valid."""
        from src.config.finrl_config import FinRLConfig

        config = FinRLConfig()
        assert config.initial_amount == 1_000_000
        assert config.transaction_cost_pct == 0.001
        assert config.window_size == 30

    def test_invalid_initial_amount_raises(self):
        """Test that negative amount raises error."""
        from src.config.finrl_config import FinRLConfig

        with pytest.raises(ValueError, match="must be positive"):
            FinRLConfig(initial_amount=-1000)

    def test_invalid_transaction_cost_raises(self):
        """Test that invalid transaction cost raises."""
        from src.config.finrl_config import FinRLConfig

        with pytest.raises(ValueError):
            FinRLConfig(transaction_cost_pct=1.5)

    def test_to_dict(self):
        """Test to_dict method."""
        from src.config.finrl_config import FinRLConfig

        config = FinRLConfig()
        d = config.to_dict()

        assert "initial_amount" in d
        assert "data_source" in d
        assert d["initial_amount"] == 1_000_000


class TestModelConfigs:
    """Test RL model configurations."""

    def test_ppo_config(self):
        """Test PPO config."""
        from src.config.model_configs import PPOConfig

        config = PPOConfig()
        assert config.learning_rate == 3e-4
        assert config.n_steps == 2048

    def test_sac_config(self):
        """Test SAC config."""
        from src.config.model_configs import SACConfig

        config = SACConfig()
        assert config.buffer_size == 1_000_000

    def test_get_algorithm_config(self):
        """Test algorithm config factory."""
        from src.config.model_configs import get_algorithm_config

        ppo = get_algorithm_config("PPO")
        assert ppo.learning_rate == 3e-4

        with pytest.raises(ValueError):
            get_algorithm_config("INVALID")
