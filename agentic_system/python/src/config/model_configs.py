"""
RL Model hyperparameter configurations.

Pre-optimized defaults for trading tasks.
All algorithms from Stable-Baselines3.

Token Optimization:
    - Dataclasses minimize boilerplate
    - Default values are trading-optimized
    - Simple to_dict() for serialization
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class PPOConfig:
    """
    Proximal Policy Optimization (PPO) configuration.

    Good general-purpose algorithm for trading.
    On-policy, works well with continuous actions.
    """

    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    verbose: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to SB3 compatible kwargs."""
        return {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "verbose": self.verbose,
        }


@dataclass
class A2CConfig:
    """
    Advantage Actor-Critic (A2C) configuration.

    Faster than PPO but less stable.
    Good for quick experiments.
    """

    learning_rate: float = 7e-4
    n_steps: int = 5
    gamma: float = 0.99
    gae_lambda: float = 1.0
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    rms_prop_eps: float = 1e-5
    verbose: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to SB3 compatible kwargs."""
        return {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "rms_prop_eps": self.rms_prop_eps,
            "verbose": self.verbose,
        }


@dataclass
class SACConfig:
    """
    Soft Actor-Critic (SAC) configuration.

    Best for continuous action spaces.
    Off-policy, sample efficient.
    Recommended for production trading.
    """

    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    learning_starts: int = 1000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str = "auto"  # Auto-tune entropy
    target_update_interval: int = 1
    target_entropy: Optional[str] = "auto"
    verbose: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to SB3 compatible kwargs."""
        return {
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "gamma": self.gamma,
            "train_freq": self.train_freq,
            "gradient_steps": self.gradient_steps,
            "ent_coef": self.ent_coef,
            "target_update_interval": self.target_update_interval,
            "verbose": self.verbose,
        }


@dataclass
class TD3Config:
    """
    Twin Delayed DDPG (TD3) configuration.

    Similar to SAC but deterministic.
    Good for lower-variance trading strategies.
    """

    learning_rate: float = 1e-3
    buffer_size: int = 1_000_000
    learning_starts: int = 1000
    batch_size: int = 100
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    policy_delay: int = 2
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    verbose: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to SB3 compatible kwargs."""
        return {
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "gamma": self.gamma,
            "train_freq": self.train_freq,
            "gradient_steps": self.gradient_steps,
            "policy_delay": self.policy_delay,
            "target_policy_noise": self.target_policy_noise,
            "target_noise_clip": self.target_noise_clip,
            "verbose": self.verbose,
        }


@dataclass
class DQNConfig:
    """
    Deep Q-Network (DQN) configuration.

    Only for discrete action spaces.
    Use for simple buy/hold/sell decisions.
    """

    learning_rate: float = 1e-4
    buffer_size: int = 1_000_000
    learning_starts: int = 1000
    batch_size: int = 32
    tau: float = 1.0
    gamma: float = 0.99
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 10000
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    verbose: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to SB3 compatible kwargs."""
        return {
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "gamma": self.gamma,
            "train_freq": self.train_freq,
            "gradient_steps": self.gradient_steps,
            "target_update_interval": self.target_update_interval,
            "exploration_fraction": self.exploration_fraction,
            "exploration_initial_eps": self.exploration_initial_eps,
            "exploration_final_eps": self.exploration_final_eps,
            "verbose": self.verbose,
        }


# Algorithm name to config mapping
ALGORITHM_CONFIGS = {
    "PPO": PPOConfig,
    "A2C": A2CConfig,
    "SAC": SACConfig,
    "TD3": TD3Config,
    "DQN": DQNConfig,
}


def get_algorithm_config(algorithm: str) -> Any:
    """
    Get default config for an algorithm.

    Args:
        algorithm: Algorithm name (PPO, A2C, SAC, TD3, DQN)

    Returns:
        Default configuration instance

    Raises:
        ValueError: If algorithm is not supported
    """
    if algorithm not in ALGORITHM_CONFIGS:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. "
            f"Supported: {list(ALGORITHM_CONFIGS.keys())}"
        )
    return ALGORITHM_CONFIGS[algorithm]()
