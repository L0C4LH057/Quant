"""
TD3 agent for trading.

TD3 (Twin Delayed DDPG):
    - Deterministic policy (lower variance than SAC)
    - Twin critics to reduce overestimation bias
    - Delayed policy updates for stability
    - Target policy smoothing for robustness

Recommended for strategies where consistent, low-variance actions matter.
"""
import logging
from typing import Any, Dict, Optional

import gymnasium as gym
from stable_baselines3 import TD3

from ..rl_agents.base_agent import BaseRLAgent
from ...config.model_configs import TD3Config

logger = logging.getLogger(__name__)


class TD3Agent(BaseRLAgent):
    """
    TD3 (Twin Delayed DDPG) agent.

    Deterministic policy — good for lower-variance trading strategies.

    Example:
        >>> env = TradingEnv(df)
        >>> agent = TD3Agent(env)
        >>> agent.train(total_timesteps=100000)
        >>> agent.save("models/td3_forex")
    """

    def __init__(
        self,
        env: gym.Env,
        config: Optional[TD3Config] = None,
        tensorboard_log: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize TD3 agent.

        Args:
            env: Trading environment
            config: TD3 configuration (uses defaults if None)
            tensorboard_log: TensorBoard log directory
            **kwargs: Override config parameters
        """
        if config is None:
            config = TD3Config()

        config_dict = config.to_dict()
        config_dict.update(kwargs)

        super().__init__(
            env=env,
            algorithm_class=TD3,
            config=config_dict,
            tensorboard_log=tensorboard_log,
        )

    @classmethod
    def load_trained(
        cls,
        path: str,
        env: gym.Env,
    ) -> "TD3Agent":
        """Load a trained TD3 agent."""
        agent = cls(env)
        agent.model = TD3.load(path, env=env)
        logger.info(f"Loaded trained TD3 agent from {path}")
        return agent
