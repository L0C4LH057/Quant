"""
DQN agent for trading.

DQN (Deep Q-Network):
    - Discrete action space only (sell/hold/buy)
    - Experience replay for sample efficiency
    - Target network for training stability
    - Epsilon-greedy exploration

Best suited for simple buy/hold/sell decision-making.
Requires DiscreteTradingEnv wrapper.
"""
import logging
from typing import Any, Dict, Optional

import gymnasium as gym
from stable_baselines3 import DQN

from ..rl_agents.base_agent import BaseRLAgent
from ...config.model_configs import DQNConfig

logger = logging.getLogger(__name__)


class DQNAgent(BaseRLAgent):
    """
    DQN (Deep Q-Network) agent.

    Only works with discrete action spaces (sell/hold/buy).
    Use with DiscreteTradingEnv.

    Example:
        >>> from src.environments.discrete_trading_env import DiscreteTradingEnv
        >>> env = DiscreteTradingEnv(df)
        >>> agent = DQNAgent(env)
        >>> agent.train(total_timesteps=50000)
        >>> agent.save("models/dqn_forex")
    """

    def __init__(
        self,
        env: gym.Env,
        config: Optional[DQNConfig] = None,
        tensorboard_log: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize DQN agent.

        Args:
            env: Trading environment (must have discrete action space)
            config: DQN configuration (uses defaults if None)
            tensorboard_log: TensorBoard log directory
            **kwargs: Override config parameters

        Raises:
            ValueError: If environment does not have discrete action space
        """
        # Validate discrete action space
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError(
                f"DQN requires Discrete action space, got {type(env.action_space).__name__}. "
                "Use DiscreteTradingEnv wrapper."
            )

        if config is None:
            config = DQNConfig()

        config_dict = config.to_dict()
        config_dict.update(kwargs)

        super().__init__(
            env=env,
            algorithm_class=DQN,
            config=config_dict,
            tensorboard_log=tensorboard_log,
        )

    @classmethod
    def load_trained(
        cls,
        path: str,
        env: gym.Env,
    ) -> "DQNAgent":
        """Load a trained DQN agent."""
        agent = cls(env)
        agent.model = DQN.load(path, env=env)
        logger.info(f"Loaded trained DQN agent from {path}")
        return agent
