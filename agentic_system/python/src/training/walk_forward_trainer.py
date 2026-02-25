"""
Walk-forward training for RL agents.

Design Decision:
    Standard train/test splits cause look-ahead bias in time series.
    Walk-forward validation simulates real-world deployment:

    Fold 1: Train [0..A], Validate [A..B]
    Fold 2: Train [0..B], Validate [B..C]    (expanding window)
    Fold 3: Train [0..C], Validate [C..D]
    ...

    Each fold trains only on past data and validates on unseen future data.
    This is the gold standard for financial time series validation.

    The trainer reports per-fold metrics and aggregated out-of-sample
    performance, giving a realistic estimate of live trading performance.
"""
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3.common.base_class import BaseAlgorithm

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Result from a single walk-forward fold."""

    fold: int
    train_size: int
    test_size: int
    train_reward: float
    test_reward: float
    test_return_pct: float
    test_trades: int
    training_time: float
    model_path: Optional[str] = None


@dataclass
class WalkForwardReport:
    """
    Complete walk-forward validation report.

    Aggregates results across all folds to provide out-of-sample metrics.
    """

    fold_results: List[WalkForwardResult] = field(default_factory=list)
    total_training_time: float = 0.0
    best_fold: int = 0

    @property
    def n_folds(self) -> int:
        """Number of completed folds."""
        return len(self.fold_results)

    @property
    def mean_test_reward(self) -> float:
        """Mean out-of-sample reward across folds."""
        if not self.fold_results:
            return 0.0
        return float(np.mean([r.test_reward for r in self.fold_results]))

    @property
    def mean_test_return(self) -> float:
        """Mean out-of-sample return percentage."""
        if not self.fold_results:
            return 0.0
        return float(np.mean([r.test_return_pct for r in self.fold_results]))

    @property
    def std_test_return(self) -> float:
        """Standard deviation of out-of-sample returns."""
        if not self.fold_results:
            return 0.0
        return float(np.std([r.test_return_pct for r in self.fold_results]))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "n_folds": self.n_folds,
            "mean_test_reward": round(self.mean_test_reward, 4),
            "mean_test_return_pct": round(self.mean_test_return, 4),
            "std_test_return_pct": round(self.std_test_return, 4),
            "best_fold": self.best_fold,
            "total_training_time": round(self.total_training_time, 2),
            "folds": [
                {
                    "fold": r.fold,
                    "train_size": r.train_size,
                    "test_size": r.test_size,
                    "train_reward": round(r.train_reward, 4),
                    "test_reward": round(r.test_reward, 4),
                    "test_return_pct": round(r.test_return_pct, 4),
                    "test_trades": r.test_trades,
                    "training_time": round(r.training_time, 2),
                }
                for r in self.fold_results
            ],
        }


class WalkForwardTrainer:
    """
    Walk-forward trainer for RL agents.

    Trains models using expanding window approach to prevent
    look-ahead bias in financial time series.

    Example:
        >>> trainer = WalkForwardTrainer(
        ...     env_factory=lambda df: TradingEnv(df, reward_type="sharpe"),
        ...     algorithm_class=PPO,
        ...     n_folds=5,
        ...     timesteps_per_fold=50000,
        ... )
        >>> report = trainer.train(market_data_df)
        >>> print(report.mean_test_return)
    """

    def __init__(
        self,
        env_factory: Callable[[pd.DataFrame], gym.Env],
        algorithm_class: Type[BaseAlgorithm],
        algorithm_kwargs: Optional[Dict[str, Any]] = None,
        n_folds: int = 5,
        min_train_ratio: float = 0.4,
        timesteps_per_fold: int = 50_000,
        eval_episodes: int = 3,
        save_path: Optional[str] = None,
    ):
        """
        Initialize walk-forward trainer.

        Args:
            env_factory: Callable that takes a DataFrame and returns a gym.Env.
                This allows the trainer to create fresh environments for each fold.
            algorithm_class: SB3 algorithm class (PPO, SAC, A2C, TD3, DQN)
            algorithm_kwargs: Algorithm hyperparameters
            n_folds: Number of walk-forward folds (default: 5)
            min_train_ratio: Minimum fraction of data for first training set (default: 0.4)
            timesteps_per_fold: Training timesteps per fold
            eval_episodes: Episodes per evaluation
            save_path: Directory to save per-fold models (optional)
        """
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}")
        if not 0 < min_train_ratio < 1:
            raise ValueError(f"min_train_ratio must be in (0, 1), got {min_train_ratio}")

        self.env_factory = env_factory
        self.algorithm_class = algorithm_class
        self.algorithm_kwargs = algorithm_kwargs or {}
        self.n_folds = n_folds
        self.min_train_ratio = min_train_ratio
        self.timesteps_per_fold = timesteps_per_fold
        self.eval_episodes = eval_episodes
        self.save_path = Path(save_path) if save_path else None

        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"WalkForwardTrainer: {algorithm_class.__name__}, "
            f"{n_folds} folds, {timesteps_per_fold} timesteps/fold"
        )

    def _split_folds(
        self,
        df: pd.DataFrame,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Split data into walk-forward folds (expanding window).

        Returns:
            List of (train_df, test_df) tuples
        """
        n = len(df)
        min_train = int(n * self.min_train_ratio)
        remaining = n - min_train
        test_size = remaining // self.n_folds

        if test_size < 10:
            raise ValueError(
                f"Insufficient data for {self.n_folds} folds. "
                f"Need at least {min_train + self.n_folds * 10} rows, got {n}."
            )

        folds = []
        for i in range(self.n_folds):
            train_end = min_train + i * test_size
            test_end = min(train_end + test_size, n)

            train_df = df.iloc[:train_end].copy()
            test_df = df.iloc[train_end:test_end].copy()

            if len(test_df) > 0:
                folds.append((train_df, test_df))

        return folds

    def train(self, df: pd.DataFrame) -> WalkForwardReport:
        """
        Execute walk-forward training across all folds.

        Args:
            df: Full market data DataFrame with indicators already added.
                Should have sufficient rows for n_folds splitting.

        Returns:
            WalkForwardReport with per-fold and aggregated metrics
        """
        report = WalkForwardReport()
        total_start = time.time()

        # Split into folds
        folds = self._split_folds(df)
        logger.info(f"Walk-forward: {len(folds)} folds created")

        best_reward = -np.inf

        for fold_idx, (train_df, test_df) in enumerate(folds):
            fold_start = time.time()
            logger.info(
                f"Fold {fold_idx + 1}/{len(folds)}: "
                f"train={len(train_df)} rows, test={len(test_df)} rows"
            )

            try:
                # Create environments
                train_env = self.env_factory(train_df)
                test_env = self.env_factory(test_df)

                # Create and train model
                model = self.algorithm_class(
                    policy="MlpPolicy",
                    env=train_env,
                    verbose=0,
                    **self.algorithm_kwargs,
                )
                model.learn(total_timesteps=self.timesteps_per_fold)

                # Evaluate on train set
                train_reward = self._evaluate(model, train_env)

                # Evaluate on test set (out-of-sample)
                test_reward, test_return, test_trades = self._evaluate_detailed(
                    model, test_env
                )

                # Save model if path provided
                model_path = None
                if self.save_path:
                    model_path = str(
                        self.save_path / f"fold_{fold_idx + 1}"
                    )
                    model.save(model_path)

                fold_time = time.time() - fold_start

                result = WalkForwardResult(
                    fold=fold_idx + 1,
                    train_size=len(train_df),
                    test_size=len(test_df),
                    train_reward=train_reward,
                    test_reward=test_reward,
                    test_return_pct=test_return,
                    test_trades=test_trades,
                    training_time=fold_time,
                    model_path=model_path,
                )
                report.fold_results.append(result)

                # Track best fold
                if test_reward > best_reward:
                    best_reward = test_reward
                    report.best_fold = fold_idx + 1

                logger.info(
                    f"Fold {fold_idx + 1}: train_reward={train_reward:.4f}, "
                    f"test_reward={test_reward:.4f}, "
                    f"test_return={test_return:.2f}%, "
                    f"trades={test_trades}, time={fold_time:.1f}s"
                )

                # Cleanup
                train_env.close()
                test_env.close()

            except Exception as e:
                logger.error(f"Fold {fold_idx + 1} failed: {e}")
                report.fold_results.append(
                    WalkForwardResult(
                        fold=fold_idx + 1,
                        train_size=len(train_df),
                        test_size=len(test_df),
                        train_reward=0.0,
                        test_reward=0.0,
                        test_return_pct=0.0,
                        test_trades=0,
                        training_time=time.time() - fold_start,
                    )
                )

        report.total_training_time = time.time() - total_start

        logger.info(
            f"Walk-forward complete: {report.n_folds} folds, "
            f"mean_test_return={report.mean_test_return:.2f}% ± "
            f"{report.std_test_return:.2f}%, "
            f"best_fold={report.best_fold}, "
            f"total_time={report.total_training_time:.1f}s"
        )

        return report

    def _evaluate(self, model: BaseAlgorithm, env: gym.Env) -> float:
        """Evaluate model and return mean reward."""
        rewards = []
        for _ in range(self.eval_episodes):
            obs, _ = env.reset()
            done = False
            total = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += reward
                done = terminated or truncated
            rewards.append(total)
        return float(np.mean(rewards))

    def _evaluate_detailed(
        self,
        model: BaseAlgorithm,
        env: gym.Env,
    ) -> Tuple[float, float, int]:
        """
        Detailed evaluation returning reward, return%, and trade count.

        Returns:
            (mean_reward, mean_return_pct, mean_trades)
        """
        rewards = []
        returns = []
        trades = []

        for _ in range(self.eval_episodes):
            obs, info = env.reset()
            done = False
            total = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                done = terminated or truncated
            rewards.append(total)
            returns.append(info.get("return_pct", 0.0))
            trades.append(info.get("num_trades", 0))

        return (
            float(np.mean(rewards)),
            float(np.mean(returns)),
            int(np.mean(trades)),
        )
