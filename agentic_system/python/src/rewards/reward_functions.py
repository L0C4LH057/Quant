"""
Reward functions for RL trading agents.

Token Optimization:
    - Simple formulas, easy to explain
    - Each function is self-contained
    - Composable for custom rewards

References:
    - Sharpe: W.F. Sharpe (1966) "Mutual Fund Performance"
    - Sortino: F.A. Sortino (1994) "Downside Risk"
"""
from typing import List, Optional

import numpy as np

from ..utils.validators import validate_array


def profit_reward(
    portfolio_value: float,
    initial_value: float,
) -> float:
    """
    Simple profit-based reward.

    Reward = (current - initial) / initial

    Args:
        portfolio_value: Current portfolio value
        initial_value: Initial portfolio value

    Returns:
        Percentage return as decimal

    Example:
        >>> profit_reward(110_000, 100_000)
        0.1  # 10% profit
    """
    if initial_value <= 0:
        return 0.0
    return (portfolio_value - initial_value) / initial_value


def sharpe_reward(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """
    Sharpe ratio reward (risk-adjusted return).

    Formula:
        Sharpe = sqrt(periods) * (mean(returns) - risk_free) / std(returns)

    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate (default: 2%)
        periods_per_year: Trading periods per year (252 for daily)

    Returns:
        Annualized Sharpe ratio

    Raises:
        ValueError: If insufficient returns

    Example:
        >>> returns = np.array([0.01, 0.02, -0.01, 0.015])
        >>> sharpe = sharpe_reward(returns)
    """
    if len(returns) < 2:
        return 0.0

    # Convert annual risk-free to period rate
    rf_period = risk_free_rate / periods_per_year

    excess_returns = returns - rf_period
    mean_excess = np.mean(excess_returns)
    std = np.std(excess_returns)

    if std == 0 or np.isnan(std):
        return 0.0

    # Annualize
    sharpe = np.sqrt(periods_per_year) * mean_excess / std

    return float(sharpe)


def sortino_reward(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """
    Sortino ratio reward (downside risk adjusted).

    Like Sharpe but only penalizes downside volatility.

    Formula:
        Sortino = sqrt(periods) * (mean(returns) - risk_free) / downside_std

    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        Annualized Sortino ratio

    Example:
        >>> returns = np.array([0.01, 0.02, -0.01, 0.015])
        >>> sortino = sortino_reward(returns)
    """
    if len(returns) < 2:
        return 0.0

    rf_period = risk_free_rate / periods_per_year
    excess_returns = returns - rf_period
    mean_excess = np.mean(excess_returns)

    # Calculate downside deviation (only negative returns)
    negative_returns = excess_returns[excess_returns < 0]
    if len(negative_returns) == 0:
        # No downside, perfect score
        return 10.0 if mean_excess > 0 else 0.0

    downside_std = np.std(negative_returns)

    if downside_std == 0 or np.isnan(downside_std):
        return 0.0

    # Annualize
    sortino = np.sqrt(periods_per_year) * mean_excess / downside_std

    return float(sortino)


def max_drawdown(returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown from returns.

    Args:
        returns: Array of period returns

    Returns:
        Maximum drawdown as positive decimal (0.2 = 20% drawdown)
    """
    if len(returns) < 1:
        return 0.0

    # Calculate cumulative wealth
    wealth = np.cumprod(1 + returns)

    # Running maximum
    running_max = np.maximum.accumulate(wealth)

    # Drawdown at each point
    drawdowns = (running_max - wealth) / running_max

    return float(np.max(drawdowns))


def risk_adjusted_reward(
    returns: np.ndarray,
    sharpe_weight: float = 0.5,
    sortino_weight: float = 0.3,
    drawdown_weight: float = 0.2,
    risk_free_rate: float = 0.02,
) -> float:
    """
    Combined risk-adjusted reward.

    Combines Sharpe, Sortino, and drawdown penalty.

    Formula:
        reward = sharpe_weight * sharpe
               + sortino_weight * sortino
               - drawdown_weight * max_drawdown

    Args:
        returns: Array of period returns
        sharpe_weight: Weight for Sharpe ratio
        sortino_weight: Weight for Sortino ratio
        drawdown_weight: Weight for drawdown penalty
        risk_free_rate: Annual risk-free rate

    Returns:
        Combined risk-adjusted reward

    Example:
        >>> returns = np.array([0.01, -0.005, 0.02, 0.01])
        >>> reward = risk_adjusted_reward(returns)
    """
    if len(returns) < 2:
        return 0.0

    # Calculate components
    sharpe = sharpe_reward(returns, risk_free_rate)
    sortino = sortino_reward(returns, risk_free_rate)
    mdd = max_drawdown(returns)

    # Combine with weights
    reward = (
        sharpe_weight * sharpe
        + sortino_weight * sortino
        - drawdown_weight * mdd * 10  # Scale drawdown
    )

    return float(reward)


class RewardCalculator:
    """
    Stateful reward calculator that tracks returns history.

    Useful for calculating running Sharpe/Sortino during episodes.

    Example:
        >>> calc = RewardCalculator(window=50)
        >>> for step in episode:
        ...     reward = calc.calculate(portfolio_value)
    """

    def __init__(
        self,
        initial_value: float = 1_000_000,
        window: int = 50,
        reward_type: str = "sharpe",
    ):
        """
        Initialize calculator.

        Args:
            initial_value: Initial portfolio value
            window: Rolling window for calculations
            reward_type: Type of reward (profit, sharpe, sortino, risk_adjusted)
        """
        self.initial_value = initial_value
        self.window = window
        self.reward_type = reward_type

        self.previous_value = initial_value
        self.returns_history: List[float] = []

    def calculate(self, current_value: float) -> float:
        """
        Calculate reward for current step.

        Args:
            current_value: Current portfolio value

        Returns:
            Reward value
        """
        # Calculate period return
        if self.previous_value > 0:
            period_return = (current_value - self.previous_value) / self.previous_value
        else:
            period_return = 0.0

        # Update history
        self.returns_history.append(period_return)

        # Keep only window
        if len(self.returns_history) > self.window:
            self.returns_history = self.returns_history[-self.window :]

        # Update previous value
        self.previous_value = current_value

        # Calculate reward based on type
        returns = np.array(self.returns_history)

        if self.reward_type == "profit":
            return profit_reward(current_value, self.initial_value)

        elif self.reward_type == "sharpe":
            return sharpe_reward(returns)

        elif self.reward_type == "sortino":
            return sortino_reward(returns)

        elif self.reward_type == "risk_adjusted":
            return risk_adjusted_reward(returns)

        else:
            # Default to simple return
            return period_return

    def reset(self, initial_value: Optional[float] = None) -> None:
        """Reset calculator state."""
        if initial_value is not None:
            self.initial_value = initial_value
        self.previous_value = self.initial_value
        self.returns_history = []
