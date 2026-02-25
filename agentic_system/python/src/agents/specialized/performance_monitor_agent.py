"""
Performance Monitoring Agent.

Tracks performance and generates alerts.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .base_specialized_agent import BaseSpecializedAgent
from ...rewards.reward_functions import sharpe_reward, sortino_reward, max_drawdown

logger = logging.getLogger(__name__)


class PerformanceMonitorAgent(BaseSpecializedAgent):
    """
    Performance Monitoring Agent.

    Responsibilities:
        - Track P&L and metrics
        - Calculate Sharpe, Sortino, drawdown
        - Generate alerts on limits
    """

    def __init__(
        self,
        daily_loss_limit: float = 0.05,  # 5% daily loss limit
        drawdown_limit: float = 0.15,  # 15% max drawdown
        llm_provider: Optional[Any] = None,
    ):
        super().__init__(
            name="performance_monitor",
            role="Performance Monitoring",
            llm_provider=llm_provider,
        )
        self.daily_loss_limit = daily_loss_limit
        self.drawdown_limit = drawdown_limit
        self.returns_history: List[float] = []
        self.initial_balance = 0
        self.peak_balance = 0

    @property
    def system_prompt(self) -> str:
        return """You are a performance monitor. Track metrics and alert on issues.
Alert thresholds: 5% daily loss, 15% max drawdown.
Output JSON: {"metrics": {...}, "alerts": [...]}"""

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update metrics and check for alerts.

        Args:
            input_data: {
                "current_balance": float,
                "period_return": float (optional),
                "trades_today": int
            }

        Returns:
            {
                "metrics": {...},
                "alerts": [...],
                "trading_allowed": bool
            }
        """
        current_balance = input_data.get("current_balance", 0)
        period_return = input_data.get("period_return")

        # Initialize if needed
        if self.initial_balance == 0:
            self.initial_balance = current_balance
            self.peak_balance = current_balance

        # Update peak
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance

        # Record return
        if period_return is not None:
            self.returns_history.append(period_return)

        # Calculate metrics
        metrics = self._calculate_metrics(current_balance)

        # Check alerts
        alerts = self._check_alerts(metrics)

        # Trading allowed?
        trading_allowed = len([a for a in alerts if a["severity"] == "critical"]) == 0

        result = {
            "metrics": metrics,
            "alerts": alerts,
            "trading_allowed": trading_allowed,
        }

        self.update_state("last_report", result)
        return result

    def _calculate_metrics(self, current_balance: float) -> Dict[str, Any]:
        """Calculate performance metrics."""
        returns = np.array(self.returns_history) if self.returns_history else np.array([0])

        # Total return
        total_return = (current_balance - self.initial_balance) / self.initial_balance * 100

        # Current drawdown
        current_drawdown = (self.peak_balance - current_balance) / self.peak_balance

        # Maximum drawdown
        mdd = max_drawdown(returns) if len(returns) > 1 else 0

        # Risk metrics
        sharpe = sharpe_reward(returns) if len(returns) > 1 else 0
        sortino = sortino_reward(returns) if len(returns) > 1 else 0

        return {
            "total_return_pct": round(total_return, 2),
            "current_drawdown": round(current_drawdown * 100, 2),
            "max_drawdown": round(mdd * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "trade_count": len(self.returns_history),
            "win_rate": self._calculate_win_rate(),
        }

    def _calculate_win_rate(self) -> float:
        """Calculate win rate."""
        if not self.returns_history:
            return 0.0

        wins = sum(1 for r in self.returns_history if r > 0)
        return round(wins / len(self.returns_history) * 100, 1)

    def _check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, str]]:
        """Check for alert conditions."""
        alerts = []

        # Daily loss check
        if len(self.returns_history) > 0:
            daily_return = self.returns_history[-1]
            if daily_return < -self.daily_loss_limit:
                alerts.append({
                    "type": "daily_loss",
                    "severity": "critical",
                    "message": f"Daily loss limit exceeded: {daily_return*100:.1f}%",
                })

        # Drawdown check
        if metrics["current_drawdown"] > self.drawdown_limit * 100:
            alerts.append({
                "type": "drawdown",
                "severity": "critical",
                "message": f"Drawdown limit exceeded: {metrics['current_drawdown']:.1f}%",
            })
        elif metrics["current_drawdown"] > self.drawdown_limit * 50:
            alerts.append({
                "type": "drawdown",
                "severity": "warning",
                "message": f"Drawdown approaching limit: {metrics['current_drawdown']:.1f}%",
            })

        return alerts

    def reset(self, initial_balance: float) -> None:
        """Reset metrics tracking."""
        self.initial_balance = initial_balance
        self.peak_balance = initial_balance
        self.returns_history = []
