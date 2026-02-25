"""
Token-optimized prompt templates for trading agents.

Token Optimization:
    - Concise system prompts (< 200 tokens each)
    - Structured output formats
    - Reusable templates
"""
from typing import Dict, Any


class TradingPrompts:
    """
    Token-optimized prompts for trading agents.

    All prompts designed to minimize tokens while maximizing clarity.
    """

    # System prompts for each agent (kept short!)
    MARKET_ANALYSIS = """You are a forex market analyst.
Analyze indicators and generate trading signals.

Output JSON:
{"signal": "buy|sell|hold", "confidence": 0.0-1.0, "reason": "brief explanation"}

Be concise. Focus on actionable insights."""

    RISK_MANAGEMENT = """You are a risk manager for forex trading.
Calculate safe position sizes based on account balance and risk tolerance.

Output JSON:
{"position_size": float, "stop_loss": float, "take_profit": float, "risk_pct": float}

Max risk per trade: 2%. Always set stop-loss."""

    PORTFOLIO_OPTIMIZATION = """You are a portfolio optimizer.
Recommend asset allocations for diversification.

Output JSON:
{"allocations": {"SYMBOL": percentage}, "rebalance": true|false, "reason": "brief"}

Target volatility: 15% annually."""

    EXECUTION = """You are a trade executor.
Analyze execution quality and slippage.

Output JSON:
{"execute": true|false, "reason": "brief", "expected_slippage": float}

Minimize market impact."""

    PERFORMANCE_MONITOR = """You are a performance analyst.
Evaluate trading performance and generate alerts.

Output JSON:
{"status": "ok|warning|critical", "metrics": {...}, "alert": "message if any"}

Alert on: >5% daily loss, >15% drawdown."""


def format_market_analysis_prompt(
    symbol: str,
    price: float,
    indicators: Dict[str, float],
) -> str:
    """
    Format market analysis prompt.

    Token Optimization: Minimal data, structured format.
    """
    # Only include key indicators
    ind_str = ", ".join(f"{k}={v:.2f}" for k, v in list(indicators.items())[:6])

    return f"""Symbol: {symbol}
Price: {price:.5f}
Indicators: {ind_str}

Analyze and provide trading signal."""


def format_risk_prompt(
    signal: str,
    confidence: float,
    price: float,
    balance: float,
    atr: float,
) -> str:
    """Format risk management prompt."""
    return f"""Signal: {signal} (confidence: {confidence:.2f})
Entry price: {price:.5f}
Account balance: ${balance:,.0f}
ATR (volatility): {atr:.5f}

Calculate position size and stop-loss."""


def format_portfolio_prompt(
    assets: list,
    current_allocations: Dict[str, float],
) -> str:
    """Format portfolio optimization prompt."""
    alloc_str = ", ".join(f"{k}={v:.1%}" for k, v in current_allocations.items())

    return f"""Assets: {', '.join(assets)}
Current allocations: {alloc_str if alloc_str else 'None'}

Recommend optimal allocations."""


def format_monitor_prompt(
    total_return: float,
    drawdown: float,
    sharpe: float,
    recent_trades: int,
) -> str:
    """Format performance monitoring prompt."""
    return f"""Performance metrics:
- Total return: {total_return:+.2%}
- Current drawdown: {drawdown:.2%}
- Sharpe ratio: {sharpe:.2f}
- Recent trades: {recent_trades}

Evaluate performance and generate alerts if needed."""
