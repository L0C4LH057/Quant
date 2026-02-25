"""
LangGraph state definitions.

Token Optimization:
    - Minimal state for context passing
    - Typed fields for validation
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MarketState:
    """Current market state."""

    symbol: str = ""
    current_price: float = 0.0
    indicators: Dict[str, float] = field(default_factory=dict)
    signal: str = "hold"
    confidence: float = 0.5


@dataclass
class PortfolioState:
    """Current portfolio state."""

    balance: float = 0.0
    holdings: Dict[str, float] = field(default_factory=dict)
    total_value: float = 0.0
    return_pct: float = 0.0


@dataclass
class AgentState:
    """
    Combined state for agent workflow.

    This state flows through the LangGraph workflow.

    Token Optimization:
        - Flat structure for easy serialization
        - Only essential fields
    """

    # Market data
    market: MarketState = field(default_factory=MarketState)

    # Portfolio data
    portfolio: PortfolioState = field(default_factory=PortfolioState)

    # Agent outputs
    analysis_result: Dict[str, Any] = field(default_factory=dict)
    risk_result: Dict[str, Any] = field(default_factory=dict)
    allocation_result: Dict[str, Any] = field(default_factory=dict)
    execution_result: Dict[str, Any] = field(default_factory=dict)
    monitoring_result: Dict[str, Any] = field(default_factory=dict)
    rl_signal_result: Dict[str, Any] = field(default_factory=dict)

    # Workflow control
    should_trade: bool = False
    error: Optional[str] = None
    step: int = 0

    # Messages between agents
    messages: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "market": {
                "symbol": self.market.symbol,
                "current_price": self.market.current_price,
                "signal": self.market.signal,
                "confidence": self.market.confidence,
            },
            "portfolio": {
                "balance": self.portfolio.balance,
                "total_value": self.portfolio.total_value,
                "return_pct": self.portfolio.return_pct,
            },
            "should_trade": self.should_trade,
            "step": self.step,
            "error": self.error,
        }

    def add_message(self, sender: str, content: Dict[str, Any]) -> None:
        """Add message to history."""
        self.messages.append({"sender": sender, "content": content})
