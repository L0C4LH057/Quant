"""
TrainingManager - Autonomous RL training orchestration.

Designed for AGI-level robustness:
    - Self-monitoring and adaptive training
    - Automatic failure recovery
    - Comprehensive audit logging
    - Multi-environment support
    - Distributed training ready

Token Optimization:
    - Single entry point for all training operations
    - State machine pattern for training phases
"""
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

import gymnasium as gym
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize

from .callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    MetricsCallback,
    TrainingStateCallback,
)

logger = logging.getLogger(__name__)


class TrainingPhase(Enum):
    """Training lifecycle phases."""
    INITIALIZED = "initialized"
    PREPARING = "preparing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class TrainingConfig:
    """
    Training configuration with sensible defaults.
    
    Attributes:
        total_timesteps: Total training timesteps
        eval_freq: Evaluation frequency in timesteps
        eval_episodes: Episodes per evaluation
        checkpoint_freq: Checkpoint frequency in timesteps
        early_stopping_patience: Early stopping patience (checks)
        early_stopping_threshold: Minimum improvement threshold
        n_envs: Number of parallel environments
        use_multiprocessing: Use SubprocVecEnv vs DummyVecEnv
        tensorboard_log: TensorBoard log directory
        checkpoint_path: Model checkpoint directory
        seed: Random seed for reproducibility
    """
    total_timesteps: int = 500_000  # UPGRADE-04: raised from 100K
    eval_freq: int = 10_000
    eval_episodes: int = 5
    checkpoint_freq: int = 10_000
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.01
    n_envs: int = 1
    use_multiprocessing: bool = False
    tensorboard_log: Optional[str] = None
    checkpoint_path: str = "checkpoints"
    seed: Optional[int] = None
    use_vec_normalize: bool = True  # UPGRADE-03: Wrap envs in VecNormalize
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
        }


@dataclass
class TrainingResult:
    """
    Complete training result with metrics and artifacts.
    
    Attributes:
        job_id: Unique training job identifier
        status: Final training phase
        timesteps_completed: Actual timesteps trained
        episodes_completed: Total episodes run
        best_reward: Best mean episode reward achieved
        final_reward: Final mean episode reward
        training_time: Total training time in seconds
        model_path: Path to best saved model
        metrics_history: Training metrics over time
        config: Training configuration used
    """
    job_id: str
    status: TrainingPhase
    timesteps_completed: int
    episodes_completed: int
    best_reward: float
    final_reward: float
    training_time: float
    model_path: Optional[str]
    metrics_history: List[Dict[str, float]] = field(default_factory=list)
    config: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "timesteps_completed": self.timesteps_completed,
            "episodes_completed": self.episodes_completed,
            "best_reward": self.best_reward,
            "final_reward": self.final_reward,
            "training_time": self.training_time,
            "model_path": self.model_path,
            "error_message": self.error_message,
        }


class TrainingManager:
    """
    Autonomous training orchestrator for RL agents.
    
    Features:
        - Complete training lifecycle management
        - Automatic checkpointing and recovery
        - Early stopping with patience
        - Multi-environment parallelization
        - Comprehensive logging and metrics
        - Training resumption from checkpoints
    
    Example:
        >>> from stable_baselines3 import PPO
        >>> manager = TrainingManager(
        ...     env_factory=lambda: TradingEnv(df),
        ...     algorithm_class=PPO,
        ...     config=TrainingConfig(total_timesteps=100_000)
        ... )
        >>> result = manager.train()
        >>> print(f"Best reward: {result.best_reward}")
    """
    
    def __init__(
        self,
        env_factory: Callable[[], gym.Env],
        algorithm_class: Type[BaseAlgorithm],
        algorithm_kwargs: Optional[Dict[str, Any]] = None,
        config: Optional[TrainingConfig] = None,
        experiment_name: Optional[str] = None,
    ):
        """
        Initialize training manager.
        
        Args:
            env_factory: Callable that creates a new environment instance
            algorithm_class: SB3 algorithm class (PPO, SAC, etc.)
            algorithm_kwargs: Algorithm-specific hyperparameters
            config: Training configuration
            experiment_name: Name for this experiment (used in logging/paths)
        """
        self.env_factory = env_factory
        self.algorithm_class = algorithm_class
        self.algorithm_kwargs = algorithm_kwargs or {}
        self.config = config or TrainingConfig()
        self.experiment_name = experiment_name or f"exp_{datetime.now():%Y%m%d_%H%M%S}"
        
        # Training state
        self.job_id = str(uuid.uuid4())[:8]
        self.phase = TrainingPhase.INITIALIZED
        self.model: Optional[BaseAlgorithm] = None
        self.envs: Optional[VecEnv] = None
        
        # Metrics tracking
        self.episode_rewards: List[float] = []
        self.best_mean_reward = -np.inf
        self.start_time: Optional[float] = None
        
        # Setup paths
        self.base_path = Path(self.config.checkpoint_path) / self.experiment_name
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"TrainingManager initialized | Job: {self.job_id} | "
            f"Algorithm: {algorithm_class.__name__} | "
            f"Experiment: {self.experiment_name}"
        )
    
    def _create_envs(self) -> VecEnv:
        """Create vectorized environments with optional VecNormalize."""
        if self.config.n_envs == 1:
            venv = DummyVecEnv([self.env_factory])
        elif self.config.use_multiprocessing:
            venv = SubprocVecEnv([self.env_factory] * self.config.n_envs)
        else:
            venv = DummyVecEnv([self.env_factory] * self.config.n_envs)

        # UPGRADE-03: normalise observations & rewards
        if self.config.use_vec_normalize:
            venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

        return venv
    
    def _create_model(self) -> BaseAlgorithm:
        """Create the RL model."""
        kwargs = {
            "policy": "MlpPolicy",
            "env": self.envs,
            "verbose": 0,
            **self.algorithm_kwargs,
        }
        
        if self.config.tensorboard_log:
            kwargs["tensorboard_log"] = self.config.tensorboard_log
            
        if self.config.seed is not None:
            kwargs["seed"] = self.config.seed
            
        return self.algorithm_class(**kwargs)
    
    def _create_callbacks(self) -> CallbackList:
        """Create training callbacks."""
        callbacks = [
            MetricsCallback(log_freq=100, verbose=1),
            CheckpointCallback(
                save_path=self.base_path / "checkpoints",
                save_freq=self.config.checkpoint_freq,
                save_best=True,
                keep_last_n=3,
                name_prefix=f"{self.algorithm_class.__name__.lower()}",
                verbose=1,
            ),
            TrainingStateCallback(
                state_path=self.base_path / "training_state.json",
                save_freq=1000,
                verbose=0,
            ),
        ]
        
        if self.config.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    patience=self.config.early_stopping_patience,
                    min_improvement=self.config.early_stopping_threshold,
                    check_freq=self.config.eval_freq,
                    min_timesteps=self.config.eval_freq * 2,
                    verbose=1,
                )
            )
        
        return CallbackList(callbacks)
    
    def _set_phase(self, phase: TrainingPhase) -> None:
        """Update training phase with logging."""
        old_phase = self.phase
        self.phase = phase
        logger.info(f"Training phase: {old_phase.value} -> {phase.value}")
    
    def train(self, resume_from: Optional[str] = None) -> TrainingResult:
        """
        Execute complete training run.
        
        Args:
            resume_from: Path to checkpoint to resume from
            
        Returns:
            TrainingResult with all metrics and paths
        """
        self.start_time = time.time()
        
        try:
            # Preparation
            self._set_phase(TrainingPhase.PREPARING)
            self.envs = self._create_envs()
            
            if resume_from:
                logger.info(f"Resuming from checkpoint: {resume_from}")
                self.model = self.algorithm_class.load(resume_from, env=self.envs)
            else:
                self.model = self._create_model()
            
            callbacks = self._create_callbacks()
            
            # Save initial config
            config_path = self.base_path / "config.json"
            with open(config_path, "w") as f:
                json.dump({
                    "job_id": self.job_id,
                    "experiment_name": self.experiment_name,
                    "algorithm": self.algorithm_class.__name__,
                    "algorithm_kwargs": {
                        k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v
                        for k, v in self.algorithm_kwargs.items()
                    },
                    "config": self.config.to_dict(),
                    "started_at": datetime.now().isoformat(),
                }, f, indent=2)
            
            # Training
            self._set_phase(TrainingPhase.TRAINING)
            logger.info(f"Starting training for {self.config.total_timesteps:,} timesteps")
            
            self.model.learn(
                total_timesteps=self.config.total_timesteps,
                callback=callbacks,
                progress_bar=True,
            )
            
            # Evaluation
            self._set_phase(TrainingPhase.EVALUATING)
            eval_result = self._evaluate(self.config.eval_episodes)
            
            # Completion
            self._set_phase(TrainingPhase.COMPLETED)
            
            # Find best model path
            best_model_path = self.base_path / "checkpoints" / f"{self.algorithm_class.__name__.lower()}_best.zip"
            if not best_model_path.exists():
                # Fall back to saving current model
                best_model_path = self.base_path / "final_model.zip"
                self.model.save(str(best_model_path))
            
            # Load training state for metrics
            state_path = self.base_path / "training_state.json"
            if state_path.exists():
                with open(state_path) as f:
                    state = json.load(f)
                    episodes = state.get("statistics", {}).get("total_episodes", 0)
                    best_reward = state.get("statistics", {}).get("max_reward", eval_result["mean_reward"])
            else:
                episodes = 0
                best_reward = eval_result["mean_reward"]
            
            return TrainingResult(
                job_id=self.job_id,
                status=TrainingPhase.COMPLETED,
                timesteps_completed=self.model.num_timesteps,
                episodes_completed=episodes,
                best_reward=best_reward,
                final_reward=eval_result["mean_reward"],
                training_time=time.time() - self.start_time,
                model_path=str(best_model_path),
                config=self.config.to_dict(),
            )
            
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            self._set_phase(TrainingPhase.STOPPED)
            return self._create_interrupted_result()
            
        except Exception as e:
            logger.exception(f"Training failed: {e}")
            self._set_phase(TrainingPhase.FAILED)
            return self._create_failed_result(str(e))
            
        finally:
            self._cleanup()
    
    def _evaluate(self, n_episodes: int) -> Dict[str, float]:
        """Run evaluation episodes."""
        if not self.model:
            return {"mean_reward": 0, "std_reward": 0}
        
        eval_env = self.env_factory()
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(n_episodes):
            obs, info = eval_env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
        
        result = {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
        }
        
        logger.info(
            f"Evaluation: {n_episodes} episodes | "
            f"Mean reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}"
        )
        
        return result
    
    def _create_interrupted_result(self) -> TrainingResult:
        """Create result for interrupted training."""
        return TrainingResult(
            job_id=self.job_id,
            status=TrainingPhase.STOPPED,
            timesteps_completed=self.model.num_timesteps if self.model else 0,
            episodes_completed=0,
            best_reward=self.best_mean_reward,
            final_reward=0,
            training_time=time.time() - (self.start_time or time.time()),
            model_path=None,
            config=self.config.to_dict(),
        )
    
    def _create_failed_result(self, error: str) -> TrainingResult:
        """Create result for failed training."""
        return TrainingResult(
            job_id=self.job_id,
            status=TrainingPhase.FAILED,
            timesteps_completed=0,
            episodes_completed=0,
            best_reward=0,
            final_reward=0,
            training_time=time.time() - (self.start_time or time.time()),
            model_path=None,
            config=self.config.to_dict(),
            error_message=error,
        )
    
    def _cleanup(self) -> None:
        """Cleanup resources."""
        if self.envs:
            try:
                self.envs.close()
            except Exception as e:
                logger.warning(f"Error closing environments: {e}")
        
        logger.info(f"Training manager cleaned up | Job: {self.job_id}")
    
    def stop(self) -> None:
        """Request training stop (for async use)."""
        logger.info("Stop requested")
        # This would be used with a custom callback that checks a stop flag
        # For now, training can be interrupted with Ctrl+C


def create_training_manager(
    env_factory: Callable[[], gym.Env],
    algorithm: str = "PPO",
    total_timesteps: int = 100_000,
    learning_rate: float = 3e-4,
    **kwargs,
) -> TrainingManager:
    """
    Factory function to create a TrainingManager.
    
    Args:
        env_factory: Callable that creates environment
        algorithm: Algorithm name (PPO, SAC, A2C, TD3, DQN)
        total_timesteps: Training timesteps
        learning_rate: Learning rate
        **kwargs: Additional TrainingConfig parameters
        
    Returns:
        Configured TrainingManager
        
    Example:
        >>> manager = create_training_manager(
        ...     env_factory=lambda: TradingEnv(df),
        ...     algorithm="PPO",
        ...     total_timesteps=50_000
        ... )
    """
    from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
    
    algorithms = {
        "PPO": PPO,
        "SAC": SAC,
        "A2C": A2C,
        "TD3": TD3,
        "DQN": DQN,
    }
    
    algorithm_class = algorithms.get(algorithm.upper())
    if not algorithm_class:
        raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(algorithms.keys())}")
    
    config = TrainingConfig(
        total_timesteps=total_timesteps,
        **{k: v for k, v in kwargs.items() if hasattr(TrainingConfig, k)}
    )
    
    algorithm_kwargs = {"learning_rate": learning_rate}
    
    return TrainingManager(
        env_factory=env_factory,
        algorithm_class=algorithm_class,
        algorithm_kwargs=algorithm_kwargs,
        config=config,
    )
