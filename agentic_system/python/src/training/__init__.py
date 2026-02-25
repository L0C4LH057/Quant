"""
Training module for PipFlow AI.

Provides autonomous training orchestration:
    - TrainingManager: Complete training lifecycle
    - Callbacks: Metrics, checkpoints, early stopping
    - HyperparameterTuner: Optuna-based optimization

Example:
    >>> from src.training import TrainingManager, TrainingConfig
    >>> manager = TrainingManager(
    ...     env_factory=lambda: TradingEnv(df),
    ...     algorithm_class=PPO,
    ...     config=TrainingConfig(total_timesteps=100_000)
    ... )
    >>> result = manager.train()
"""
from .callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    MetricsCallback,
    TrainingStateCallback,
)
from .trainer import (
    TrainingConfig,
    TrainingManager,
    TrainingPhase,
    TrainingResult,
    create_training_manager,
)
from .hyperparameter_tuning import (
    HyperparameterTuner,
    TuningResult,
)

__all__ = [
    # Manager
    "TrainingManager",
    "TrainingConfig",
    "TrainingResult",
    "TrainingPhase",
    "create_training_manager",
    # Callbacks
    "MetricsCallback",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "TrainingStateCallback",
    # Tuning
    "HyperparameterTuner",
    "TuningResult",
]
