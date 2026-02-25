"""
SAC agent for trading.

SAC (Soft Actor-Critic) is optimal for continuous actions:
    - Auto-tuned entropy for exploration
    - Off-policy (sample efficient)
    - Stable with replay buffer

Recommended for production trading.
"""
import logging
from typing import Any, Dict, Optional

import gymnasium as gym
from stable_baselines3 import SAC

from ..rl_agents.base_agent import BaseRLAgent
from ...config.model_configs import SACConfig

logger = logging.getLogger(__name__)


class SACAgent(BaseRLAgent):
    """
    SAC (Soft Actor-Critic) agent.

    Best choice for continuous action trading.

    Example:
        >>> env = TradingEnv(df)
        >>> agent = SACAgent(env, buffer_size=100000)
        >>> agent.train(total_timesteps=100000)
    """

    def __init__(
        self,
        env: gym.Env,
        config: Optional[SACConfig] = None,
        tensorboard_log: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize SAC agent.

        Args:
            env: Trading environment
            config: SAC configuration (uses defaults if None)
            tensorboard_log: TensorBoard log directory
            **kwargs: Override config parameters
        """
        if config is None:
            config = SACConfig()

        config_dict = config.to_dict()
        config_dict.update(kwargs)

        super().__init__(
            env=env,
            algorithm_class=SAC,
            config=config_dict,
            tensorboard_log=tensorboard_log,
        )

    @classmethod
    def load_trained(
        cls,
        path: str,
        env: gym.Env,
    ) -> "SACAgent":
        """Load a trained SAC agent."""
        agent = cls(env)
        agent.model = SAC.load(path, env=env)
        logger.info(f"Loaded trained SAC agent from {path}")
        return agent
