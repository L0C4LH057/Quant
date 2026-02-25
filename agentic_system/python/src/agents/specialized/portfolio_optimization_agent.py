"""
Portfolio Optimization Agent.

Optimizes asset allocation using Modern Portfolio Theory.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .base_specialized_agent import BaseSpecializedAgent

logger = logging.getLogger(__name__)


class PortfolioOptimizationAgent(BaseSpecializedAgent):
    """
    Portfolio Optimization Agent.

    Responsibilities:
        - Calculate optimal asset allocation
        - Analyze correlations
        - Generate rebalancing recommendations
    """

    def __init__(
        self,
        target_volatility: float = 0.15,  # 15% annual vol target
        llm_provider: Optional[Any] = None,
    ):
        super().__init__(
            name="portfolio_optimizer",
            role="Portfolio Optimization",
            llm_provider=llm_provider,
        )
        self.target_volatility = target_volatility

    @property
    def system_prompt(self) -> str:
        return """You are a portfolio optimizer. Recommend asset allocations.
Target volatility: 15%. Diversify across assets.
Output JSON: {"allocations": {"ASSET": pct}, "reason": "brief"}"""

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate optimal allocations.

        Args:
            input_data: {
                "assets": ["SYM1", "SYM2", ...],
                "returns": {symbol: [returns]},
                "current_allocations": {symbol: pct}
            }

        Returns:
            {
                "target_allocations": {...},
                "rebalance_needed": bool,
                "trades": [...]
            }
        """
        assets = input_data.get("assets", [])
        returns_data = input_data.get("returns", {})
        current = input_data.get("current_allocations", {})

        if not assets:
            return {"error": "No assets provided", "target_allocations": {}}

        # Calculate optimal weights
        target_allocations = self._calculate_allocations(assets, returns_data)

        # Check if rebalancing needed
        rebalance_needed, trades = self._check_rebalance(current, target_allocations)

        result = {
            "target_allocations": target_allocations,
            "rebalance_needed": rebalance_needed,
            "trades": trades,
            "diversification_score": self._diversification_score(target_allocations),
        }

        self.update_state("last_optimization", result)
        return result

    def _calculate_allocations(
        self,
        assets: List[str],
        returns_data: Dict[str, List[float]],
    ) -> Dict[str, float]:
        """
        Calculate equal-weighted allocations (simplified).

        For production, implement mean-variance optimization.
        """
        n = len(assets)
        if n == 0:
            return {}

        # Simple equal weight for now
        weight = round(1.0 / n, 4)
        allocations = {asset: weight for asset in assets}

        return allocations

    def _check_rebalance(
        self,
        current: Dict[str, float],
        target: Dict[str, float],
        threshold: float = 0.05,
    ) -> tuple:
        """Check if rebalancing is needed."""
        trades = []
        rebalance_needed = False

        for asset, target_weight in target.items():
            current_weight = current.get(asset, 0)
            diff = target_weight - current_weight

            if abs(diff) > threshold:
                rebalance_needed = True
                trades.append({
                    "asset": asset,
                    "action": "buy" if diff > 0 else "sell",
                    "amount_pct": round(abs(diff) * 100, 2),
                })

        return rebalance_needed, trades

    def _diversification_score(self, allocations: Dict[str, float]) -> float:
        """Calculate diversification score (0-1)."""
        if not allocations:
            return 0.0

        weights = list(allocations.values())
        n = len(weights)

        # Herfindahl-Hirschman Index inverse
        hhi = sum(w ** 2 for w in weights)
        max_diversified = 1.0 / n

        score = 1 - (hhi - max_diversified) / (1 - max_diversified)
        return round(max(0, min(1, score)), 2)
