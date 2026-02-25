"""
Backtesting module for PipFlow AI.

Provides historical simulation and performance analysis:
    - BacktestEngine: Run strategies on historical data
    - Metrics: Comprehensive performance calculations
    - Reporter: HTML reports with visualizations

Example:
    >>> from src.backtesting import BacktestEngine, BacktestConfig
    >>> engine = BacktestEngine(
    ...     model=trained_model,
    ...     env_factory=lambda df: TradingEnv(df),
    ... )
    >>> result = engine.run(market_data)
    >>> print(result.metrics.summary())
"""
from .engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
)
from .metrics import (
    BacktestMetrics,
    MetricsCalculator,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    total_return,
    cagr,
)
from .reporter import BacktestReporter

__all__ = [
    # Engine
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    # Metrics
    "BacktestMetrics",
    "MetricsCalculator",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "total_return",
    "cagr",
    # Reporter
    "BacktestReporter",
]
