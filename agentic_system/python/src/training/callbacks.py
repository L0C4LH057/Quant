"""
Training callbacks for RL agents.

Custom Stable-Baselines3 callbacks for:
- Model checkpointing with versioning
- Early stopping on reward plateau
- Metrics logging (TensorBoard compatible)
- Training state persistence for resumption

Token Optimization:
    - Each callback is self-contained
    - Minimal dependencies between callbacks
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EventCallback

logger = logging.getLogger(__name__)


class MetricsCallback(BaseCallback):
    """
    Logs training metrics at each step.
    
    Tracks:
        - Episode rewards and lengths
        - Learning rate
        - Policy/value losses
        - Custom metrics via callable
    
    Example:
        >>> callback = MetricsCallback(log_freq=100)
        >>> model.learn(total_timesteps=10000, callback=callback)
    """
    
    def __init__(
        self,
        log_freq: int = 100,
        custom_metrics_fn: Optional[Callable[[], Dict[str, float]]] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.custom_metrics_fn = custom_metrics_fn
        
        # Tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.start_time: float = 0.0
        
    def _on_training_start(self) -> None:
        self.start_time = time.time()
        logger.info("Training started - MetricsCallback active")
        
    def _on_step(self) -> bool:
        # Check for episode end
        if self.locals.get("dones") is not None:
            for idx, done in enumerate(self.locals["dones"]):
                if done:
                    info = self.locals.get("infos", [{}])[idx]
                    ep_reward = info.get("episode", {}).get("r", 0)
                    ep_length = info.get("episode", {}).get("l", 0)
                    
                    if ep_reward:
                        self.episode_rewards.append(ep_reward)
                        self.episode_lengths.append(ep_length)
        
        # Log at frequency
        if self.n_calls % self.log_freq == 0:
            self._log_metrics()
            
        return True
    
    def _log_metrics(self) -> None:
        elapsed = time.time() - self.start_time
        fps = self.num_timesteps / elapsed if elapsed > 0 else 0
        
        metrics = {
            "timesteps": self.num_timesteps,
            "fps": fps,
            "time_elapsed": elapsed,
        }
        
        # Recent episode stats
        if self.episode_rewards:
            recent = self.episode_rewards[-100:]
            metrics.update({
                "ep_reward_mean": np.mean(recent),
                "ep_reward_std": np.std(recent),
                "ep_length_mean": np.mean(self.episode_lengths[-100:]),
                "episodes_total": len(self.episode_rewards),
            })
        
        # Custom metrics
        if self.custom_metrics_fn:
            try:
                custom = self.custom_metrics_fn()
                metrics.update(custom)
            except Exception as e:
                logger.warning(f"Custom metrics failed: {e}")
        
        # Log to TensorBoard if available
        if self.logger:
            for key, value in metrics.items():
                self.logger.record(f"train/{key}", value)
        
        if self.verbose >= 1:
            logger.info(
                f"Step {self.num_timesteps:,} | "
                f"FPS: {fps:.0f} | "
                f"Reward: {metrics.get('ep_reward_mean', 0):.2f}"
            )


class CheckpointCallback(BaseCallback):
    """
    Saves model checkpoints with versioning.
    
    Features:
        - Save every N steps
        - Save best model by reward
        - Keep only last K checkpoints
        - Save training state for resumption
    
    Example:
        >>> callback = CheckpointCallback(
        ...     save_path="checkpoints/",
        ...     save_freq=5000,
        ...     keep_last_n=5
        ... )
    """
    
    def __init__(
        self,
        save_path: Union[str, Path],
        save_freq: int = 5000,
        save_best: bool = True,
        keep_last_n: int = 5,
        name_prefix: str = "model",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.save_freq = save_freq
        self.save_best = save_best
        self.keep_last_n = keep_last_n
        self.name_prefix = name_prefix
        
        # Tracking
        self.best_mean_reward = -np.inf
        self.episode_rewards: List[float] = []
        self.saved_models: List[Path] = []
        
    def _init_callback(self) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoints will be saved to: {self.save_path}")
        
    def _on_step(self) -> bool:
        # Track episode rewards
        if self.locals.get("dones") is not None:
            for idx, done in enumerate(self.locals["dones"]):
                if done:
                    info = self.locals.get("infos", [{}])[idx]
                    ep_reward = info.get("episode", {}).get("r")
                    if ep_reward is not None:
                        self.episode_rewards.append(ep_reward)
        
        # Periodic save
        if self.n_calls % self.save_freq == 0:
            self._save_checkpoint(f"{self.name_prefix}_{self.num_timesteps}")
            
        # Best model save
        if self.save_best and len(self.episode_rewards) >= 10:
            mean_reward = np.mean(self.episode_rewards[-100:])
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self._save_checkpoint(f"{self.name_prefix}_best")
                logger.info(f"New best model! Mean reward: {mean_reward:.2f}")
                
        return True
    
    def _save_checkpoint(self, name: str) -> None:
        path = self.save_path / name
        self.model.save(str(path))
        
        # Save training state
        state = {
            "timesteps": self.num_timesteps,
            "best_reward": float(self.best_mean_reward),
            "episodes": len(self.episode_rewards),
            "timestamp": time.time(),
        }
        state_path = self.save_path / f"{name}_state.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)
        
        # Track and cleanup old checkpoints
        if "best" not in name:
            self.saved_models.append(path)
            if len(self.saved_models) > self.keep_last_n:
                old_path = self.saved_models.pop(0)
                try:
                    (old_path.with_suffix(".zip")).unlink(missing_ok=True)
                    (self.save_path / f"{old_path.name}_state.json").unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to cleanup {old_path}: {e}")
        
        if self.verbose >= 1:
            logger.info(f"Saved checkpoint: {path}")


class EarlyStoppingCallback(EventCallback):
    """
    Stops training when reward plateaus.
    
    Uses relative improvement threshold over a patience window.
    
    Example:
        >>> callback = EarlyStoppingCallback(
        ...     patience=10,
        ...     min_improvement=0.01
        ... )
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_improvement: float = 0.01,
        check_freq: int = 1000,
        min_timesteps: int = 10000,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.patience = patience
        self.min_improvement = min_improvement
        self.check_freq = check_freq
        self.min_timesteps = min_timesteps
        
        # Tracking
        self.episode_rewards: List[float] = []
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0
        self.last_check_timestep = 0
        
    def _on_step(self) -> bool:
        # Track rewards
        if self.locals.get("dones") is not None:
            for idx, done in enumerate(self.locals["dones"]):
                if done:
                    info = self.locals.get("infos", [{}])[idx]
                    ep_reward = info.get("episode", {}).get("r")
                    if ep_reward is not None:
                        self.episode_rewards.append(ep_reward)
        
        # Check for early stopping
        if (self.num_timesteps - self.last_check_timestep) >= self.check_freq:
            self.last_check_timestep = self.num_timesteps
            
            if self.num_timesteps < self.min_timesteps:
                return True
                
            if len(self.episode_rewards) < 10:
                return True
            
            mean_reward = np.mean(self.episode_rewards[-100:])
            
            # Check improvement
            improvement = (mean_reward - self.best_mean_reward) / (abs(self.best_mean_reward) + 1e-8)
            
            if improvement > self.min_improvement:
                self.best_mean_reward = mean_reward
                self.no_improvement_count = 0
                if self.verbose >= 1:
                    logger.info(f"Improvement: {improvement:.2%}, new best: {mean_reward:.2f}")
            else:
                self.no_improvement_count += 1
                if self.verbose >= 1:
                    logger.info(
                        f"No improvement ({self.no_improvement_count}/{self.patience}), "
                        f"current: {mean_reward:.2f}, best: {self.best_mean_reward:.2f}"
                    )
            
            # Stop if patience exceeded
            if self.no_improvement_count >= self.patience:
                logger.warning(
                    f"Early stopping triggered after {self.num_timesteps:,} timesteps. "
                    f"Best reward: {self.best_mean_reward:.2f}"
                )
                return False
                
        return True


class TrainingStateCallback(BaseCallback):
    """
    Persists complete training state for resumption.
    
    Enables:
        - Resume from exact point after interruption
        - Transfer learning state
        - Training analysis post-hoc
    
    Example:
        >>> callback = TrainingStateCallback("training_state.json")
        >>> # Later, to resume:
        >>> state = TrainingStateCallback.load_state("training_state.json")
    """
    
    def __init__(
        self,
        state_path: Union[str, Path],
        save_freq: int = 1000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.state_path = Path(state_path)
        self.save_freq = save_freq
        
        # Full state tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.losses: List[Dict[str, float]] = []
        self.start_time: float = 0.0
        
    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        
    def _on_step(self) -> bool:
        # Track episodes
        if self.locals.get("dones") is not None:
            for idx, done in enumerate(self.locals["dones"]):
                if done:
                    info = self.locals.get("infos", [{}])[idx]
                    ep_info = info.get("episode", {})
                    if ep_info:
                        self.episode_rewards.append(ep_info.get("r", 0))
                        self.episode_lengths.append(ep_info.get("l", 0))
        
        # Save state periodically
        if self.n_calls % self.save_freq == 0:
            self._save_state()
            
        return True
    
    def _on_training_end(self) -> None:
        self._save_state()
        logger.info(f"Final training state saved to {self.state_path}")
    
    def _save_state(self) -> None:
        state = {
            "timesteps": self.num_timesteps,
            "n_calls": self.n_calls,
            "elapsed_time": time.time() - self.start_time,
            "episode_rewards": self.episode_rewards[-1000:],  # Last 1000
            "episode_lengths": self.episode_lengths[-1000:],
            "statistics": {
                "total_episodes": len(self.episode_rewards),
                "mean_reward": float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0,
                "std_reward": float(np.std(self.episode_rewards[-100:])) if self.episode_rewards else 0,
                "max_reward": float(max(self.episode_rewards)) if self.episode_rewards else 0,
            },
            "timestamp": time.time(),
        }
        
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)
    
    @staticmethod
    def load_state(path: Union[str, Path]) -> Dict[str, Any]:
        """Load training state from file."""
        with open(path) as f:
            return json.load(f)
