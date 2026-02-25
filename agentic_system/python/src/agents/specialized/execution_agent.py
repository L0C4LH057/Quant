"""
Execution Agent.

Executes trades via broker APIs.
"""
import logging
from typing import Any, Dict, Optional

from .base_specialized_agent import BaseSpecializedAgent

logger = logging.getLogger(__name__)


class ExecutionAgent(BaseSpecializedAgent):
    """
    Execution Agent.

    Responsibilities:
        - Execute trades via broker API
        - Track order status
        - Report execution quality
    """

    def __init__(
        self,
        broker_client: Optional[Any] = None,
        llm_provider: Optional[Any] = None,
    ):
        super().__init__(
            name="executor",
            role="Trade Execution",
            llm_provider=llm_provider,
        )
        self.broker_client = broker_client
        self.pending_orders: Dict[str, Dict] = {}

    @property
    def system_prompt(self) -> str:
        return """You are a trade executor. Execute orders efficiently.
Report slippage and execution quality.
Output JSON: {"order_id": str, "status": str, "slippage": float}"""

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute trade order.

        Args:
            input_data: {
                "action": "buy" | "sell",
                "symbol": str,
                "size": float,
                "price": float (limit) or None (market),
                "stop_loss": float,
                "take_profit": float
            }

        Returns:
            {
                "order_id": str,
                "status": "executed" | "pending" | "failed",
                "filled_price": float,
                "slippage": float
            }
        """
        action = input_data.get("action")
        symbol = input_data.get("symbol")
        size = input_data.get("size", 0)
        target_price = input_data.get("price")
        stop_loss = input_data.get("stop_loss")
        take_profit = input_data.get("take_profit")

        # Validate
        if not action or not symbol or size <= 0:
            return {
                "status": "failed",
                "error": "Invalid order parameters",
            }

        # Simulate execution (replace with real broker call)
        result = await self._execute_order(
            action=action,
            symbol=symbol,
            size=size,
            target_price=target_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        self.update_state("last_execution", result)
        logger.info(f"Execution: {action} {size} {symbol} -> {result['status']}")

        return result

    async def _execute_order(
        self,
        action: str,
        symbol: str,
        size: float,
        target_price: Optional[float],
        stop_loss: Optional[float],
        take_profit: Optional[float],
    ) -> Dict[str, Any]:
        """
        Execute order (simulation or real broker).
        """
        import uuid
        import random

        # For now, simulate execution
        order_id = str(uuid.uuid4())[:8]

        # Simulate slippage (0-0.5%)
        slippage_pct = random.uniform(0, 0.005)
        filled_price = target_price or 0

        if action == "buy":
            filled_price = filled_price * (1 + slippage_pct)
        else:
            filled_price = filled_price * (1 - slippage_pct)

        return {
            "order_id": order_id,
            "status": "executed",
            "action": action,
            "symbol": symbol,
            "size": size,
            "target_price": target_price,
            "filled_price": round(filled_price, 5),
            "slippage_pct": round(slippage_pct * 100, 3),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get status of pending order."""
        if order_id in self.pending_orders:
            return self.pending_orders[order_id]
        return {"order_id": order_id, "status": "unknown"}
