"""
LangGraph state definitions.

Uses TypedDict with Annotated reducers as required by LangGraph >=0.2.

Token Optimization:
    - Minimal state for context passing
    - Typed fields for validation
"""
import operator
from typing import Annotated, Any, Dict, List, Optional

from typing_extensions import TypedDict

import pandas as pd


class MarketState(TypedDict, total=False):
    """Current market state."""

    symbol: str
    current_price: float
    indicators: Dict[str, float]
    signal: str
    confidence: float
    ohlcv_df: Optional[pd.DataFrame]  # Real OHLCV data for RL signal node


class PortfolioState(TypedDict, total=False):
    """Current portfolio state."""

    balance: float
    holdings: Dict[str, float]
    total_value: float
    return_pct: float


class AgentState(TypedDict, total=False):
    """
    Combined state for agent workflow.

    This state flows through the LangGraph workflow.
    Uses TypedDict with Annotated reducers for LangGraph >=0.2 compatibility.

    Token Optimization:
        - Flat structure for easy serialization
        - Only essential fields
    """

    # Market data
    market: MarketState

    # Portfolio data
    portfolio: PortfolioState

    # Agent outputs
    analysis_result: Dict[str, Any]
    risk_result: Dict[str, Any]
    allocation_result: Dict[str, Any]
    execution_result: Dict[str, Any]
    monitoring_result: Dict[str, Any]
    rl_signal_result: Dict[str, Any]

    # Workflow control
    should_trade: bool
    error: Optional[str]
    step: int

    # Messages between agents — use reducer to accumulate across nodes
    messages: Annotated[List[Dict[str, Any]], operator.add]


# ── Helper functions ──────────────────────────────────────────────────────


def create_initial_state(
    symbol: str = "",
    price: float = 0.0,
    indicators: Optional[Dict[str, float]] = None,
    balance: float = 100_000,
    ohlcv_df: Optional[pd.DataFrame] = None,
) -> AgentState:
    """Create a properly initialised AgentState."""
    return AgentState(
        market=MarketState(
            symbol=symbol,
            current_price=price,
            indicators=indicators or {},
            signal="hold",
            confidence=0.5,
            ohlcv_df=ohlcv_df,
        ),
        portfolio=PortfolioState(
            balance=balance,
            holdings={},
            total_value=balance,
            return_pct=0.0,
        ),
        analysis_result={},
        risk_result={},
        allocation_result={},
        execution_result={},
        monitoring_result={},
        rl_signal_result={},
        should_trade=False,
        error=None,
        step=0,
        messages=[],
    )


def state_to_dict(state: AgentState) -> Dict[str, Any]:
    """Convert AgentState to a serialisable dictionary."""
    market = state.get("market", {})
    portfolio = state.get("portfolio", {})
    return {
        "market": {
            "symbol": market.get("symbol", ""),
            "current_price": market.get("current_price", 0.0),
            "signal": market.get("signal", "hold"),
            "confidence": market.get("confidence", 0.5),
        },
        "portfolio": {
            "balance": portfolio.get("balance", 0.0),
            "total_value": portfolio.get("total_value", 0.0),
            "return_pct": portfolio.get("return_pct", 0.0),
        },
        "should_trade": state.get("should_trade", False),
        "step": state.get("step", 0),
        "error": state.get("error"),
    }
