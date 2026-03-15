"""
Performance metrics for backtesting.

Calculates comprehensive trading performance metrics:
    - Returns (total, annualized, CAGR)
    - Risk metrics (Sharpe, Sortino, Calmar)
    - Drawdown analysis
    - Trade statistics

GAP-13 fix: Added ``infer_periods_per_year`` helper that auto-detects the
correct annualisation factor from the data's DatetimeIndex instead of
always assuming 252 (daily).

Token Optimization:
    - Each metric is a standalone function
    - MetricsCalculator provides batch computation
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── GAP-13: Auto-detect annualisation factor ─────────────────────────────


def infer_periods_per_year(
    index: Optional[pd.DatetimeIndex] = None,
    n_periods: int = 0,
    total_days: float = 0.0,
) -> int:
    """
    Infer the correct ``periods_per_year`` from a DatetimeIndex **or** from
    the number of periods and calendar span.

    Falls back to 252 (daily) when detection is ambiguous.
    """
    if index is not None and len(index) >= 2:
        median_delta = pd.Series(index).diff().dropna().median()
        seconds = median_delta.total_seconds()
    elif n_periods > 1 and total_days > 0:
        seconds = (total_days * 86400) / n_periods
    else:
        return 252  # default to daily

    if seconds < 120:         # ~1 min
        return 252 * 390      # ~1-minute bars (390 per trading day)
    elif seconds < 600:       # ~5 min
        return 252 * 78
    elif seconds < 1800:      # ~15 min
        return 252 * 26
    elif seconds < 7200:      # ~1 h
        return 252 * 6
    elif seconds < 28800:     # ~4 h
        return 252 * 2
    elif seconds < 172800:    # ~1 d
        return 252
    elif seconds < 864000:    # ~1 w
        return 52
    else:
        return 12             # monthly


@dataclass
class BacktestMetrics:
    """
    Complete backtest performance metrics.
    
    All metrics are calculated from equity curve and trade log.
    """
    # Returns
    total_return: float = 0.0
    annualized_return: float = 0.0
    cagr: float = 0.0
    
    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Risk
    volatility: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # days
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    avg_trade_duration: float = 0.0  # hours
    
    # Additional
    best_day: float = 0.0
    worst_day: float = 0.0
    exposure_time: float = 0.0  # % of time in market
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in self.__dict__.items()}
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        return f"""
=== Backtest Performance Summary ===
Total Return: {self.total_return:.2%}
CAGR: {self.cagr:.2%}
Sharpe Ratio: {self.sharpe_ratio:.2f}
Sortino Ratio: {self.sortino_ratio:.2f}
Max Drawdown: {self.max_drawdown:.2%}

Win Rate: {self.win_rate:.1%} ({self.winning_trades}/{self.total_trades})
Profit Factor: {self.profit_factor:.2f}
Avg Profit: ${self.avg_profit:,.2f}
Avg Loss: ${self.avg_loss:,.2f}
"""


def calculate_returns(equity: np.ndarray) -> np.ndarray:
    """Calculate period returns from equity curve."""
    if len(equity) < 2:
        return np.array([0.0])
    returns = np.diff(equity) / equity[:-1]
    return np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)


def total_return(equity: np.ndarray) -> float:
    """Calculate total return from equity curve."""
    if len(equity) < 2 or equity[0] == 0:
        return 0.0
    return (equity[-1] - equity[0]) / equity[0]


def annualized_return(equity: np.ndarray, periods_per_year: int = 252) -> float:
    """Calculate annualized return."""
    if len(equity) < 2:
        return 0.0
    total = total_return(equity)
    n_periods = len(equity) - 1
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0
    return (1 + total) ** (1 / years) - 1


def cagr(equity: np.ndarray, periods_per_year: int = 252) -> float:
    """Calculate Compound Annual Growth Rate."""
    return annualized_return(equity, periods_per_year)


def volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Calculate annualized volatility."""
    if len(returns) < 2:
        return 0.0
    return np.std(returns) * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """Calculate Sharpe ratio."""
    if len(returns) < 2:
        return 0.0
    
    rf_period = risk_free_rate / periods_per_year
    excess_returns = returns - rf_period
    
    std = np.std(excess_returns)
    if std == 0:
        return 0.0
    
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / std


def sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """Calculate Sortino ratio (downside risk adjusted)."""
    if len(returns) < 2:
        return 0.0
    
    rf_period = risk_free_rate / periods_per_year
    excess_returns = returns - rf_period
    
    downside = excess_returns[excess_returns < 0]
    if len(downside) == 0:
        return 10.0 if np.mean(excess_returns) > 0 else 0.0
    
    downside_std = np.std(downside)
    if downside_std == 0:
        return 0.0
    
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / downside_std


def max_drawdown(equity: np.ndarray) -> float:
    """Calculate maximum drawdown."""
    if len(equity) < 2:
        return 0.0
    
    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity) / running_max
    return float(np.max(drawdowns))


def max_drawdown_duration(equity: np.ndarray) -> int:
    """Calculate maximum drawdown duration in periods."""
    if len(equity) < 2:
        return 0
    
    running_max = np.maximum.accumulate(equity)
    in_drawdown = equity < running_max
    
    max_duration = 0
    current_duration = 0
    
    for is_dd in in_drawdown:
        if is_dd:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    
    return max_duration


def calmar_ratio(
    equity: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """Calculate Calmar ratio (return / max drawdown)."""
    ann_ret = annualized_return(equity, periods_per_year)
    mdd = max_drawdown(equity)
    
    if mdd == 0:
        return 0.0 if ann_ret <= 0 else 10.0
    
    return ann_ret / mdd


def trade_statistics(trades: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate trade statistics.
    
    Args:
        trades: DataFrame with columns [side, price, amount, pnl, timestamp]
        
    Returns:
        Dictionary of trade statistics
    """
    if trades.empty or "pnl" not in trades.columns:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
        }
    
    # Round-trip trades (paired entries)
    closed_trades = trades[trades["pnl"].notna()].copy()
    
    if closed_trades.empty:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
        }
    
    winners = closed_trades[closed_trades["pnl"] > 0]
    losers = closed_trades[closed_trades["pnl"] < 0]
    
    total_profit = winners["pnl"].sum() if not winners.empty else 0
    total_loss = abs(losers["pnl"].sum()) if not losers.empty else 0
    
    return {
        "total_trades": len(closed_trades),
        "winning_trades": len(winners),
        "losing_trades": len(losers),
        "win_rate": len(winners) / len(closed_trades) if len(closed_trades) > 0 else 0,
        "avg_profit": winners["pnl"].mean() if not winners.empty else 0,
        "avg_loss": losers["pnl"].mean() if not losers.empty else 0,
        "profit_factor": total_profit / total_loss if total_loss > 0 else 0,
    }


class MetricsCalculator:
    """
    Batch metrics calculator for backtest results.
    
    Example:
        >>> calculator = MetricsCalculator(
        ...     equity=equity_curve,
        ...     trades=trades_df
        ... )
        >>> metrics = calculator.calculate_all()
    """
    
    def __init__(
        self,
        equity: np.ndarray,
        trades: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ):
        """
        Initialize calculator.
        
        Args:
            equity: Equity curve array
            trades: Trade log DataFrame
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
        """
        self.equity = np.array(equity)
        self.trades = trades if trades is not None else pd.DataFrame()
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        
        # Pre-calculate returns
        self.returns = calculate_returns(self.equity)
    
    def calculate_all(self) -> BacktestMetrics:
        """Calculate all metrics."""
        metrics = BacktestMetrics()
        
        # Returns
        metrics.total_return = total_return(self.equity)
        metrics.annualized_return = annualized_return(self.equity, self.periods_per_year)
        metrics.cagr = cagr(self.equity, self.periods_per_year)
        
        # Risk-adjusted
        metrics.sharpe_ratio = sharpe_ratio(self.returns, self.risk_free_rate, self.periods_per_year)
        metrics.sortino_ratio = sortino_ratio(self.returns, self.risk_free_rate, self.periods_per_year)
        metrics.calmar_ratio = calmar_ratio(self.equity, self.periods_per_year)
        
        # Risk
        metrics.volatility = volatility(self.returns, self.periods_per_year)
        metrics.max_drawdown = max_drawdown(self.equity)
        metrics.max_drawdown_duration = max_drawdown_duration(self.equity)
        
        # Daily extremes
        if len(self.returns) > 0:
            metrics.best_day = float(np.max(self.returns))
            metrics.worst_day = float(np.min(self.returns))
        
        # Trade statistics
        trade_stats = trade_statistics(self.trades)
        metrics.total_trades = trade_stats["total_trades"]
        metrics.winning_trades = trade_stats["winning_trades"]
        metrics.losing_trades = trade_stats["losing_trades"]
        metrics.win_rate = trade_stats["win_rate"]
        metrics.avg_profit = trade_stats["avg_profit"]
        metrics.avg_loss = trade_stats["avg_loss"]
        metrics.profit_factor = trade_stats["profit_factor"]
        
        return metrics
