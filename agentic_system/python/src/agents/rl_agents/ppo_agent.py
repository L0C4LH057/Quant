"""
PPO agent for trading.

PPO is a good general-purpose algorithm:
    - Works with continuous actions
    - Stable training
    - Good sample efficiency

Recommended for initial experiments.
"""
import logging
from typing import Any, Dict, Optional

import gymnasium as gym
from stable_baselines3 import PPO

from ..rl_agents.base_agent import BaseRLAgent
from ...config.model_configs import PPOConfig

logger = logging.getLogger(__name__)


class PPOAgent(BaseRLAgent):
    """
    PPO (Proximal Policy Optimization) agent.

    Good default choice for trading environments.

    Example:
        >>> env = TradingEnv(df)
        >>> agent = PPOAgent(env, learning_rate=3e-4)
        >>> agent.train(total_timesteps=50000)
        >>> agent.save("models/ppo_forex")
    """

    def __init__(
        self,
        env: gym.Env,
        config: Optional[PPOConfig] = None,
        tensorboard_log: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize PPO agent.

        Args:
            env: Trading environment
            config: PPO configuration (uses defaults if None)
            tensorboard_log: TensorBoard log directory
            **kwargs: Override config parameters
        """
        # Use default config if not provided
        if config is None:
            config = PPOConfig()

        # Convert to dict and apply overrides
        config_dict = config.to_dict()
        config_dict.update(kwargs)

        super().__init__(
            env=env,
            algorithm_class=PPO,
            config=config_dict,
            tensorboard_log=tensorboard_log,
        )

    @classmethod
    def load_trained(
        cls,
        path: str,
        env: gym.Env,
    ) -> "PPOAgent":
        """
        Load a trained PPO agent.

        Args:
            path: Path to saved model
            env: Environment for the agent

        Returns:
            Loaded PPOAgent instance
        """
        agent = cls(env)
        agent.model = PPO.load(path, env=env)
        logger.info(f"Loaded trained PPO agent from {path}")
        return agent
