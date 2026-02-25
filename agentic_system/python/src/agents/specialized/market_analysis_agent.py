"""
Market Analysis Agent.

Analyzes market conditions and generates trading signals.

Token Optimization:
    - Concise signal format
    - Structured output
"""
import logging
from typing import Any, Dict, List, Optional

from .base_specialized_agent import BaseSpecializedAgent

logger = logging.getLogger(__name__)


class MarketAnalysisAgent(BaseSpecializedAgent):
    """
    Market Analysis Agent.

    Responsibilities:
        - Analyze price action and indicators
        - Detect patterns and trends
        - Generate trading signals with confidence

    Output Format:
        {
            "signal": "buy" | "sell" | "hold",
            "confidence": 0.0-1.0,
            "reason": "Brief explanation",
            "indicators": {...}
        }
    """

    def __init__(self, llm_provider: Optional[Any] = None):
        super().__init__(
            name="market_analyst",
            role="Market Analysis",
            llm_provider=llm_provider,
        )

    @property
    def system_prompt(self) -> str:
        """Concise system prompt for market analysis."""
        return """You are a market analyst for forex trading.
Analyze indicators and price action.
Output JSON: {"signal": "buy/sell/hold", "confidence": 0.0-1.0, "reason": "brief"}
Be concise. Focus on actionable signals."""

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market and generate signal.

        Uses LLM for enhanced analysis if provider is configured,
        otherwise falls back to rule-based analysis.

        Args:
            input_data: {
                "symbol": str,
                "current_price": float,
                "indicators": {...},
                "use_llm": bool (optional, default False)
            }

        Returns:
            {
                "signal": str,
                "confidence": float,
                "reason": str,
                "indicators_summary": {...}
            }
        """
        symbol = input_data.get("symbol", "UNKNOWN")
        indicators = input_data.get("indicators", {})
        current_price = input_data.get("current_price", 0)
        use_llm = input_data.get("use_llm", False)

        # Try LLM-enhanced analysis if requested and provider available
        if use_llm and self.llm_provider is not None:
            try:
                result = await self._analyze_with_llm(symbol, current_price, indicators)
                if result:
                    self.update_state("last_analysis", result)
                    self.update_state("last_signal", result.get("signal", "hold"))
                    logger.info(f"MarketAnalysis (LLM): {symbol} -> {result['signal']}")
                    return result
            except Exception as e:
                logger.warning(f"LLM analysis failed, using rules: {e}")

        # Fallback to rule-based analysis
        signal, confidence, reason = self._analyze_indicators(indicators)

        result = {
            "symbol": symbol,
            "signal": signal,
            "confidence": confidence,
            "reason": reason,
            "current_price": current_price,
            "indicators_summary": self._summarize_indicators(indicators),
            "analysis_type": "rule_based",
        }

        self.update_state("last_analysis", result)
        self.update_state("last_signal", signal)

        logger.info(f"MarketAnalysis: {symbol} -> {signal} ({confidence:.2f})")
        return result

    async def _analyze_with_llm(
        self,
        symbol: str,
        price: float,
        indicators: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM for enhanced market analysis.

        Token Optimization:
            - Minimal prompt with key indicators only
            - Structured JSON output
        """
        from ...llm.prompts import format_market_analysis_prompt

        user_prompt = format_market_analysis_prompt(symbol, price, indicators)

        response = await self.llm_provider.generate_structured(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            max_tokens=200,
        )

        if "error" in response:
            return None

        return {
            "symbol": symbol,
            "signal": response.get("signal", "hold"),
            "confidence": float(response.get("confidence", 0.5)),
            "reason": response.get("reason", "LLM analysis"),
            "current_price": price,
            "indicators_summary": self._summarize_indicators(indicators),
            "analysis_type": "llm_enhanced",
        }



    def _analyze_indicators(
        self,
        indicators: Dict[str, float],
    ) -> tuple:
        """
        Rule-based analysis of indicators.

        Returns:
            (signal, confidence, reason)
        """
        signals = []
        reasons = []

        # RSI analysis
        rsi = indicators.get("rsi_14", 50)
        if rsi < 30:
            signals.append(1)  # Buy signal
            reasons.append("RSI oversold")
        elif rsi > 70:
            signals.append(-1)  # Sell signal
            reasons.append("RSI overbought")
        else:
            signals.append(0)

        # MACD analysis
        macd = indicators.get("macd", 0)
        macd_signal = indicators.get("macd_signal", 0)
        if macd > macd_signal:
            signals.append(1)
            reasons.append("MACD bullish")
        elif macd < macd_signal:
            signals.append(-1)
            reasons.append("MACD bearish")
        else:
            signals.append(0)

        # SMA crossover
        sma_20 = indicators.get("sma_20", 0)
        sma_50 = indicators.get("sma_50", 0)
        current = indicators.get("close", 0)

        if current > sma_20 > sma_50:
            signals.append(1)
            reasons.append("Above SMAs")
        elif current < sma_20 < sma_50:
            signals.append(-1)
            reasons.append("Below SMAs")
        else:
            signals.append(0)

        # Aggregate signals
        avg_signal = sum(signals) / len(signals) if signals else 0

        if avg_signal > 0.3:
            signal = "buy"
            confidence = min(0.5 + abs(avg_signal) * 0.4, 0.95)
        elif avg_signal < -0.3:
            signal = "sell"
            confidence = min(0.5 + abs(avg_signal) * 0.4, 0.95)
        else:
            signal = "hold"
            confidence = 0.6

        reason = ", ".join(reasons) if reasons else "Mixed signals"

        return signal, confidence, reason

    def _summarize_indicators(
        self,
        indicators: Dict[str, float],
    ) -> Dict[str, str]:
        """Summarize indicators in human-readable format."""
        summary = {}

        if "rsi_14" in indicators:
            rsi = indicators["rsi_14"]
            if rsi < 30:
                summary["rsi"] = f"{rsi:.1f} (oversold)"
            elif rsi > 70:
                summary["rsi"] = f"{rsi:.1f} (overbought)"
            else:
                summary["rsi"] = f"{rsi:.1f} (neutral)"

        if "macd" in indicators:
            macd = indicators["macd"]
            signal = indicators.get("macd_signal", 0)
            if macd > signal:
                summary["macd"] = "bullish crossover"
            else:
                summary["macd"] = "bearish crossover"

        return summary
