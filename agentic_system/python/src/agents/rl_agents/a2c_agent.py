"""
A2C agent for trading.

A2C (Advantage Actor-Critic):
    - On-policy, synchronous variant of A3C
    - Faster training than PPO but less stable
    - Good for rapid prototyping and short experiments

Recommended for quick iteration on strategy ideas.
"""
import logging
from typing import Any, Dict, Optional

import gymnasium as gym
from stable_baselines3 import A2C

from ..rl_agents.base_agent import BaseRLAgent
from ...config.model_configs import A2CConfig

logger = logging.getLogger(__name__)


class A2CAgent(BaseRLAgent):
    """
    A2C (Advantage Actor-Critic) agent.

    Faster than PPO, useful for quick experiments.

    Example:
        >>> env = TradingEnv(df)
        >>> agent = A2CAgent(env, learning_rate=7e-4)
        >>> agent.train(total_timesteps=50000)
        >>> agent.save("models/a2c_forex")
    """

    def __init__(
        self,
        env: gym.Env,
        config: Optional[A2CConfig] = None,
        tensorboard_log: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize A2C agent.

        Args:
            env: Trading environment
            config: A2C configuration (uses defaults if None)
            tensorboard_log: TensorBoard log directory
            **kwargs: Override config parameters
        """
        if config is None:
            config = A2CConfig()

        config_dict = config.to_dict()
        config_dict.update(kwargs)

        super().__init__(
            env=env,
            algorithm_class=A2C,
            config=config_dict,
            tensorboard_log=tensorboard_log,
        )

    @classmethod
    def load_trained(
        cls,
        path: str,
        env: gym.Env,
    ) -> "A2CAgent":
        """Load a trained A2C agent."""
        agent = cls(env)
        agent.model = A2C.load(path, env=env)
        logger.info(f"Loaded trained A2C agent from {path}")
        return agent
