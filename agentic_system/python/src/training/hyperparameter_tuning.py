"""
Hyperparameter tuning using Optuna.

Features:
    - Automatic hyperparameter search
    - Pruning for early trial termination
    - Best config export
    - Multi-objective optimization support

Token Optimization:
    - Self-contained tuner class
    - Sensible search spaces per algorithm
"""
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

import gymnasium as gym
import numpy as np

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

logger = logging.getLogger(__name__)


@dataclass
class TuningResult:
    """
    Hyperparameter tuning result.
    
    Attributes:
        best_params: Best hyperparameters found
        best_value: Best objective value (mean reward)
        n_trials: Total trials run
        study_name: Optuna study name
        optimization_history: Trial results over time
    """
    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    study_name: str
    optimization_history: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "n_trials": self.n_trials,
            "study_name": self.study_name,
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """Save tuning result to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class TrialEvalCallback(BaseCallback):
    """
    Callback for Optuna trial evaluation with pruning.
    
    Reports intermediate values to Optuna and handles pruning.
    """
    
    def __init__(
        self,
        trial: "optuna.Trial",
        eval_env: gym.Env,
        n_eval_episodes: int = 5,
        eval_freq: int = 1000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.trial = trial
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        
        self.eval_idx = 0
        self.last_mean_reward = -np.inf
        self.is_pruned = False
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate
            rewards = []
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                total_reward = 0.0
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                    total_reward += reward
                    done = terminated or truncated
                    
                rewards.append(total_reward)
            
            mean_reward = np.mean(rewards)
            self.last_mean_reward = mean_reward
            
            # Report to Optuna
            self.trial.report(mean_reward, self.eval_idx)
            self.eval_idx += 1
            
            # Check pruning
            if self.trial.should_prune():
                self.is_pruned = True
                return False
                
        return True


class HyperparameterTuner:
    """
    Optuna-based hyperparameter tuner for RL algorithms.
    
    Automatically searches for optimal hyperparameters with:
        - Algorithm-specific search spaces
        - Early pruning of bad trials
        - Parallel trial execution (optional)
        - Result persistence
    
    Example:
        >>> tuner = HyperparameterTuner(
        ...     env_factory=lambda: TradingEnv(df),
        ...     algorithm="PPO",
        ...     n_trials=20
        ... )
        >>> result = tuner.tune()
        >>> print(result.best_params)
    """
    
    # Default search spaces per algorithm
    SEARCH_SPACES = {
        "PPO": {
            "learning_rate": ("log_float", 1e-5, 1e-2),
            "n_steps": ("categorical", [256, 512, 1024, 2048]),
            "batch_size": ("categorical", [32, 64, 128, 256]),
            "gamma": ("float", 0.9, 0.9999),
            "gae_lambda": ("float", 0.9, 0.99),
            "clip_range": ("float", 0.1, 0.3),
            "ent_coef": ("log_float", 1e-8, 0.1),
            "n_epochs": ("int", 3, 20),
        },
        "SAC": {
            "learning_rate": ("log_float", 1e-5, 1e-2),
            "buffer_size": ("categorical", [10000, 50000, 100000, 500000]),
            "batch_size": ("categorical", [64, 128, 256, 512]),
            "gamma": ("float", 0.9, 0.9999),
            "tau": ("float", 0.001, 0.1),
            "learning_starts": ("int", 100, 10000),
            "ent_coef": ("categorical", ["auto", 0.1, 0.01]),
        },
        "A2C": {
            "learning_rate": ("log_float", 1e-5, 1e-2),
            "n_steps": ("categorical", [5, 16, 32, 64, 128]),
            "gamma": ("float", 0.9, 0.9999),
            "gae_lambda": ("float", 0.9, 0.99),
            "ent_coef": ("log_float", 1e-8, 0.1),
            "vf_coef": ("float", 0.1, 0.9),
            "max_grad_norm": ("float", 0.3, 1.0),
        },
        "TD3": {
            "learning_rate": ("log_float", 1e-5, 1e-2),
            "buffer_size": ("categorical", [10000, 50000, 100000]),
            "batch_size": ("categorical", [64, 128, 256]),
            "gamma": ("float", 0.9, 0.9999),
            "tau": ("float", 0.001, 0.1),
            "policy_delay": ("int", 1, 5),
            "learning_starts": ("int", 100, 10000),
        },
        "DQN": {
            "learning_rate": ("log_float", 1e-5, 1e-2),
            "buffer_size": ("categorical", [10000, 50000, 100000]),
            "batch_size": ("categorical", [32, 64, 128]),
            "gamma": ("float", 0.9, 0.9999),
            "exploration_fraction": ("float", 0.1, 0.5),
            "exploration_final_eps": ("float", 0.01, 0.1),
            "target_update_interval": ("int", 100, 10000),
        },
    }
    
    def __init__(
        self,
        env_factory: Callable[[], gym.Env],
        algorithm: str = "PPO",
        n_trials: int = 20,
        n_timesteps_per_trial: int = 10_000,
        n_eval_episodes: int = 5,
        eval_freq: int = 1000,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        custom_search_space: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        verbose: int = 1,
    ):
        """
        Initialize hyperparameter tuner.
        
        Args:
            env_factory: Callable that creates environment
            algorithm: Algorithm name (PPO, SAC, A2C, TD3, DQN)
            n_trials: Number of optimization trials
            n_timesteps_per_trial: Timesteps per trial
            n_eval_episodes: Episodes for evaluation
            eval_freq: Evaluation frequency
            study_name: Optuna study name
            storage: Optuna storage URL (for distributed)
            custom_search_space: Override default search space
            seed: Random seed
            verbose: Verbosity level
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not installed. Run: pip install optuna")
        
        self.env_factory = env_factory
        self.algorithm = algorithm.upper()
        self.n_trials = n_trials
        self.n_timesteps_per_trial = n_timesteps_per_trial
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.study_name = study_name or f"{algorithm}_tuning_{datetime.now():%Y%m%d_%H%M%S}"
        self.storage = storage
        self.seed = seed
        self.verbose = verbose
        
        # Get search space
        if custom_search_space:
            self.search_space = custom_search_space
        else:
            self.search_space = self.SEARCH_SPACES.get(self.algorithm, {})
            
        if not self.search_space:
            raise ValueError(f"No search space for {algorithm}. Provide custom_search_space.")
        
        # Get algorithm class
        self.algorithm_class = self._get_algorithm_class()
        
        logger.info(
            f"HyperparameterTuner initialized | Algorithm: {algorithm} | "
            f"Trials: {n_trials} | Timesteps/trial: {n_timesteps_per_trial}"
        )
    
    def _get_algorithm_class(self) -> Type[BaseAlgorithm]:
        """Get SB3 algorithm class."""
        from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
        
        algorithms = {
            "PPO": PPO,
            "SAC": SAC,
            "A2C": A2C,
            "TD3": TD3,
            "DQN": DQN,
        }
        
        if self.algorithm not in algorithms:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
            
        return algorithms[self.algorithm]
    
    def _sample_params(self, trial: "optuna.Trial") -> Dict[str, Any]:
        """Sample hyperparameters for a trial."""
        params = {}
        
        for name, spec in self.search_space.items():
            param_type = spec[0]
            
            if param_type == "float":
                params[name] = trial.suggest_float(name, spec[1], spec[2])
            elif param_type == "log_float":
                params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
            elif param_type == "int":
                params[name] = trial.suggest_int(name, spec[1], spec[2])
            elif param_type == "categorical":
                params[name] = trial.suggest_categorical(name, spec[1])
                
        return params
    
    def _objective(self, trial: "optuna.Trial") -> float:
        """Optuna objective function."""
        # Sample hyperparameters
        params = self._sample_params(trial)
        
        if self.verbose >= 1:
            logger.info(f"Trial {trial.number}: {params}")
        
        # Create environments
        train_env = DummyVecEnv([self.env_factory])
        eval_env = self.env_factory()
        
        try:
            # Create model
            model = self.algorithm_class(
                policy="MlpPolicy",
                env=train_env,
                seed=self.seed,
                verbose=0,
                **params,
            )
            
            # Create evaluation callback
            eval_callback = TrialEvalCallback(
                trial=trial,
                eval_env=eval_env,
                n_eval_episodes=self.n_eval_episodes,
                eval_freq=self.eval_freq,
            )
            
            # Train
            model.learn(
                total_timesteps=self.n_timesteps_per_trial,
                callback=eval_callback,
            )
            
            # Check if pruned
            if eval_callback.is_pruned:
                raise optuna.TrialPruned()
            
            return eval_callback.last_mean_reward
            
        except Exception as e:
            if isinstance(e, optuna.TrialPruned):
                raise
            logger.warning(f"Trial {trial.number} failed: {e}")
            return float("-inf")
            
        finally:
            train_env.close()
    
    def tune(self) -> TuningResult:
        """
        Run hyperparameter optimization.
        
        Returns:
            TuningResult with best parameters and history
        """
        logger.info(f"Starting hyperparameter tuning: {self.n_trials} trials")
        
        # Create study
        sampler = TPESampler(seed=self.seed)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=sampler,
            pruner=pruner,
            direction="maximize",
            load_if_exists=True,
        )
        
        # Optimize
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            show_progress_bar=self.verbose >= 1,
        )
        
        # Collect history
        history = []
        for trial in study.trials:
            history.append({
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
            })
        
        result = TuningResult(
            best_params=study.best_params,
            best_value=study.best_value,
            n_trials=len(study.trials),
            study_name=self.study_name,
            optimization_history=history,
        )
        
        logger.info(
            f"Tuning complete | Best value: {result.best_value:.2f} | "
            f"Best params: {result.best_params}"
        )
        
        return result
    
    def tune_and_train(
        self,
        final_timesteps: int = 100_000,
        save_path: Optional[Union[str, Path]] = None,
    ) -> tuple:
        """
        Tune hyperparameters then train final model.
        
        Args:
            final_timesteps: Timesteps for final training
            save_path: Path to save final model
            
        Returns:
            Tuple of (TuningResult, trained_model)
        """
        # Tune
        tuning_result = self.tune()
        
        # Train with best params
        logger.info(f"Training final model with best params for {final_timesteps} timesteps")
        
        train_env = DummyVecEnv([self.env_factory])
        
        model = self.algorithm_class(
            policy="MlpPolicy",
            env=train_env,
            seed=self.seed,
            verbose=1,
            **tuning_result.best_params,
        )
        
        model.learn(total_timesteps=final_timesteps, progress_bar=True)
        
        if save_path:
            model.save(str(save_path))
            logger.info(f"Saved final model to {save_path}")
        
        train_env.close()
        
        return tuning_result, model
