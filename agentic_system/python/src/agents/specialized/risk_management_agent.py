"""
Risk Management Agent.

Calculates position sizes and manages risk.

Token Optimization:
    - Simple formulas, no LLM needed
    - Clear output format
"""
import logging
from typing import Any, Dict, Optional

from .base_specialized_agent import BaseSpecializedAgent

logger = logging.getLogger(__name__)


class RiskManagementAgent(BaseSpecializedAgent):
    """
    Risk Management Agent.

    Responsibilities:
        - Calculate position sizes (Kelly Criterion)
        - Set stop-loss levels
        - Calculate VaR and risk metrics
        - Enforce risk limits

    Output Format:
        {
            "position_size": float,
            "stop_loss": float,
            "take_profit": float,
            "risk_amount": float,
            "risk_reward_ratio": float
        }
    """

    def __init__(
        self,
        max_risk_per_trade: float = 0.02,  # 2% max risk
        max_position_pct: float = 0.25,  # 25% max position
        llm_provider: Optional[Any] = None,
    ):
        super().__init__(
            name="risk_manager",
            role="Risk Management",
            llm_provider=llm_provider,
        )
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_pct = max_position_pct

    @property
    def system_prompt(self) -> str:
        return """You are a risk manager. Calculate safe position sizes.
Max risk per trade: 2%. Always recommend stop-losses.
Output JSON: {"approved": true/false, "reason": "brief"}"""

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate risk parameters.

        Args:
            input_data: {
                "signal": str,
                "confidence": float,
                "current_price": float,
                "account_balance": float,
                "atr": float (optional)
            }

        Returns:
            {
                "approved": bool,
                "position_size": float,
                "stop_loss": float,
                "take_profit": float,
                "risk_amount": float,
                "reason": str
            }
        """
        signal = input_data.get("signal", "hold")
        confidence = input_data.get("confidence", 0.5)
        current_price = input_data.get("current_price", 100)
        account_balance = input_data.get("account_balance", 100000)
        atr = input_data.get("atr", current_price * 0.01)  # Default 1% ATR

        # No trade for hold signal
        if signal == "hold":
            return {
                "approved": False,
                "position_size": 0,
                "stop_loss": 0,
                "take_profit": 0,
                "risk_amount": 0,
                "reason": "Hold signal - no trade",
            }

        # Calculate risk-adjusted position size
        risk_pct = self._calculate_risk_pct(confidence)
        risk_amount = account_balance * risk_pct

        # Stop loss distance (based on ATR)
        stop_distance = atr * 2  # 2x ATR stop

        # Position size based on risk
        position_size = risk_amount / stop_distance

        # Apply maximum position limit
        max_position = account_balance * self.max_position_pct / current_price
        position_size = min(position_size, max_position)

        # Calculate stop loss and take profit levels
        if signal == "buy":
            stop_loss = current_price - stop_distance
            take_profit = current_price + (stop_distance * 2)  # 2:1 R:R
        else:  # sell
            stop_loss = current_price + stop_distance
            take_profit = current_price - (stop_distance * 2)

        result = {
            "approved": True,
            "position_size": round(position_size, 4),
            "stop_loss": round(stop_loss, 5),
            "take_profit": round(take_profit, 5),
            "risk_amount": round(risk_amount, 2),
            "risk_pct": round(risk_pct * 100, 2),
            "risk_reward_ratio": 2.0,
            "reason": f"Risk: {risk_pct*100:.1f}%, Size: {position_size:.4f}",
        }

        self.update_state("last_risk_calc", result)
        logger.info(f"RiskManagement: Position={position_size:.4f}, Risk=${risk_amount:.2f}")

        return result

    def _calculate_risk_pct(self, confidence: float) -> float:
        """
        Calculate risk percentage based on confidence.

        Higher confidence = more risk (up to max).
        Uses modified Kelly Criterion.
        """
        # Scale risk by confidence (0.5 to 1.0 confidence)
        scaled_confidence = max(0, (confidence - 0.5) * 2)

        # Kelly fraction (simplified)
        kelly = scaled_confidence * self.max_risk_per_trade

        # Apply fractional Kelly (half Kelly for safety)
        risk_pct = kelly * 0.5

        return min(risk_pct, self.max_risk_per_trade)

    def calculate_var(
        self,
        returns: list,
        confidence: float = 0.95,
    ) -> float:
        """
        Calculate Value at Risk.

        Args:
            returns: Historical returns
            confidence: Confidence level (default 95%)

        Returns:
            VaR as positive decimal
        """
        import numpy as np

        if len(returns) < 2:
            return 0.0

        returns = np.array(returns)
        var = np.percentile(returns, (1 - confidence) * 100)

        return abs(var)
