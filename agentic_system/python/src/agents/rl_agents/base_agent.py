"""
Base wrapper for Stable-Baselines3 RL agents.

Provides unified interface for training, saving, loading, and prediction.

Token Optimization:
    - Simple interface reduces prompt complexity
    - Consistent method signatures across algorithms
"""
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

import gymnasium as gym
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class BaseRLAgent(ABC):
    """
    Base class for RL agent wrappers.

    Provides unified interface for all SB3 algorithms.

    Example:
        >>> agent = PPOAgent(env, learning_rate=3e-4)
        >>> agent.train(total_timesteps=10000)
        >>> agent.save("models/ppo_trading")
        >>> action = agent.predict(observation)
    """

    def __init__(
        self,
        env: gym.Env,
        algorithm_class: Type[BaseAlgorithm],
        config: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
    ):
        """
        Initialize agent.

        Args:
            env: Gymnasium environment
            algorithm_class: SB3 algorithm class (PPO, SAC, etc.)
            config: Algorithm hyperparameters
            tensorboard_log: Path for TensorBoard logs
        """
        self.env = env
        self.algorithm_class = algorithm_class
        self.config = config or {}
        self.tensorboard_log = tensorboard_log

        # Create the model
        self.model = self._create_model()

        logger.info(f"Initialized {self.__class__.__name__} agent")

    def _create_model(self) -> BaseAlgorithm:
        """Create the SB3 model instance."""
        return self.algorithm_class(
            policy="MlpPolicy",
            env=self.env,
            tensorboard_log=self.tensorboard_log,
            **self.config,
        )

    def train(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        progress_bar: bool = True,
    ) -> "BaseRLAgent":
        """
        Train the agent.

        Args:
            total_timesteps: Total training timesteps
            callback: Optional callback for monitoring
            progress_bar: Show progress bar

        Returns:
            Self for method chaining
        """
        logger.info(f"Training for {total_timesteps} timesteps")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=progress_bar,
        )

        logger.info("Training complete")
        return self

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """
        Get action for observation.

        Args:
            observation: Current observation
            deterministic: Use deterministic policy

        Returns:
            Action array
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to file.

        Args:
            path: File path (without extension)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        logger.info(f"Saved model to {path}")

    def load(self, path: Union[str, Path]) -> "BaseRLAgent":
        """
        Load model from file.

        Args:
            path: File path

        Returns:
            Self with loaded model
        """
        path = Path(path)
        self.model = self.algorithm_class.load(str(path), env=self.env)
        logger.info(f"Loaded model from {path}")
        return self

    def evaluate(
        self,
        n_episodes: int = 5,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate agent performance.

        Args:
            n_episodes: Number of evaluation episodes
            deterministic: Use deterministic policy

        Returns:
            Dictionary with mean/std reward and episode lengths
        """
        episode_rewards = []
        episode_lengths = []

        for _ in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                action = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

        return {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
        }

    @property
    def algorithm_name(self) -> str:
        """Get algorithm name."""
        return self.algorithm_class.__name__
