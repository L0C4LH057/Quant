"""
FinRL-specific configuration for trading environments and training.

Token Optimization:
    - Uses dataclass for minimal boilerplate
    - Validation in __post_init__ catches errors early
    - Clear naming reduces explanation needs in prompts
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class FinRLConfig:
    """
    FinRL Framework Configuration.

    All parameters for trading environments and RL training.

    Attributes:
        data_source: Market data provider
        start_date: Backtest start date (YYYY-MM-DD)
        end_date: Backtest end date (YYYY-MM-DD)
        initial_amount: Starting capital
        transaction_cost_pct: Cost per trade (0.001 = 0.1%)
        reward_scaling: Scale factor for training stability
        tech_indicator_list: List of technical indicators to use

    Example:
        >>> config = FinRLConfig(initial_amount=100_000)
        >>> print(config.transaction_cost_pct)
        0.001
    """

    # Data Source
    data_source: str = "yahoofinance"
    start_date: str = "2020-01-01"
    end_date: str = "2024-01-01"

    # Symbols (default forex pairs)
    symbols: List[str] = field(default_factory=lambda: ["EURUSD=X", "GBPUSD=X", "USDJPY=X"])

    # Technical Indicators
    use_technical_indicator: bool = True
    tech_indicator_list: List[str] = field(
        default_factory=lambda: [
            "sma_20",
            "sma_50",
            "ema_12",
            "rsi_14",
            "macd",
            "macd_signal",
            "bb_upper",
            "bb_lower",
            "atr_14",
        ]
    )

    # Environment Parameters
    initial_amount: float = 1_000_000
    transaction_cost_pct: float = 0.001  # 0.1% per trade
    reward_scaling: float = 1e-4

    # Training Parameters
    timesteps_per_iteration: int = 10_000
    total_timesteps: int = 100_000
    n_eval_episodes: int = 5

    # Lookback window for observations
    window_size: int = 30

    def __post_init__(self) -> None:
        """
        Validate configuration after initialization.

        Raises:
            ValueError: If any parameter is invalid
        """
        if self.initial_amount <= 0:
            raise ValueError(f"Initial amount must be positive, got {self.initial_amount}")

        if not 0 <= self.transaction_cost_pct < 1:
            raise ValueError(
                f"Transaction cost must be in [0, 1), got {self.transaction_cost_pct}"
            )

        if self.reward_scaling <= 0:
            raise ValueError(f"Reward scaling must be positive, got {self.reward_scaling}")

        if self.window_size < 1:
            raise ValueError(f"Window size must be >= 1, got {self.window_size}")

        valid_sources = ["yahoofinance", "alphaadvantage", "binance"]
        if self.data_source not in valid_sources:
            raise ValueError(f"Data source must be one of {valid_sources}, got {self.data_source}")

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "data_source": self.data_source,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "symbols": self.symbols,
            "initial_amount": self.initial_amount,
            "transaction_cost_pct": self.transaction_cost_pct,
            "window_size": self.window_size,
            "tech_indicator_list": self.tech_indicator_list,
        }
