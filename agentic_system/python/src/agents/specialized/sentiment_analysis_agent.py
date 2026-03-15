"""
Sentiment Analysis Agent.

Analyzes news sentiment and generates trading signals based on
sentiment features, extending the specialist agent pattern.

Token Optimization:
    - Concise signal format
    - Structured output matching other specialized agents
"""
import logging
from typing import Any, Dict, Optional

from .base_specialized_agent import BaseSpecializedAgent

logger = logging.getLogger(__name__)


class SentimentAnalysisAgent(BaseSpecializedAgent):
    """
    Sentiment Analysis Agent.

    Responsibilities:
        - Analyze news sentiment for a given symbol
        - Detect sentiment regime shifts (bearish → bullish)
        - Generate trading signals with confidence based on sentiment

    Output Format:
        {
            "signal": "buy" | "sell" | "hold",
            "confidence": 0.0-1.0,
            "reason": "Brief explanation",
            "sentiment_summary": {...}
        }
    """

    # Thresholds for rule-based signal generation
    BULLISH_THRESHOLD = 0.15
    BEARISH_THRESHOLD = -0.15
    HIGH_CONFIDENCE_THRESHOLD = 0.30
    MIN_ARTICLES = 3  # Minimum articles for a confident signal

    def __init__(self, llm_provider: Optional[Any] = None):
        super().__init__(
            name="sentiment_analyst",
            role="News Sentiment Analysis",
            llm_provider=llm_provider,
        )

    @property
    def system_prompt(self) -> str:
        """Concise system prompt for sentiment analysis."""
        return """You are a financial news sentiment analyst.
Analyze the sentiment data and news headlines provided.
Consider: headline tone, volume of coverage, sentiment momentum, and divergence with price.
Output JSON: {"signal": "buy/sell/hold", "confidence": 0.0-1.0, "reason": "brief"}
Be concise. Focus on actionable signals from news sentiment."""

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment and generate signal.

        Uses LLM for enhanced analysis if provider is configured,
        otherwise falls back to rule-based analysis.

        Args:
            input_data: {
                "symbol": str,
                "sentiment_features": {
                    "sentiment_score": float,       # -1.0 to 1.0
                    "sentiment_magnitude": float,   # 0.0 to 1.0
                    "sentiment_volume": int,         # article count
                    "sentiment_momentum": float,     # rate of change
                    "sentiment_divergence": float,   # 0.0 or 1.0
                },
                "headlines": list (optional),       # top headlines for LLM
                "sentiment_summary": dict (optional),
                "use_llm": bool (optional, default False),
            }

        Returns:
            {
                "signal": str,
                "confidence": float,
                "reason": str,
                "sentiment_summary": dict,
            }
        """
        symbol = input_data.get("symbol", "UNKNOWN")
        features = input_data.get("sentiment_features", {})
        headlines = input_data.get("headlines", [])
        summary = input_data.get("sentiment_summary", {})
        use_llm = input_data.get("use_llm", False)

        # Try LLM-enhanced analysis if requested
        if use_llm and self.llm_provider is not None and headlines:
            try:
                result = await self._analyze_with_llm(
                    symbol, features, headlines, summary
                )
                if result:
                    self.update_state("last_analysis", result)
                    self.update_state("last_signal", result.get("signal", "hold"))
                    logger.info(
                        f"SentimentAnalysis (LLM): {symbol} -> {result['signal']}"
                    )
                    return result
            except Exception as e:
                logger.warning(f"LLM sentiment analysis failed, using rules: {e}")

        # Fallback to rule-based analysis
        signal, confidence, reason = self._analyze_sentiment(features)

        result = {
            "symbol": symbol,
            "signal": signal,
            "confidence": confidence,
            "reason": reason,
            "sentiment_summary": summary or self._make_summary(features),
            "analysis_type": "rule_based",
        }

        self.update_state("last_analysis", result)
        self.update_state("last_signal", signal)

        logger.info(f"SentimentAnalysis: {symbol} -> {signal} ({confidence:.2f})")
        return result

    def _analyze_sentiment(
        self,
        features: Dict[str, float],
    ) -> tuple:
        """
        Rule-based sentiment signal generation.

        Returns:
            (signal, confidence, reason)
        """
        score = features.get("sentiment_score", 0.0)
        magnitude = features.get("sentiment_magnitude", 0.0)
        volume = features.get("sentiment_volume", 0)
        momentum = features.get("sentiment_momentum", 0.0)
        divergence = features.get("sentiment_divergence", 0.0)

        reasons = []
        signal_value = 0.0  # -1 to +1 scale

        # Insufficient data → hold with low confidence
        if volume < self.MIN_ARTICLES:
            return "hold", 0.3, "Insufficient news volume for confident signal"

        # Core sentiment direction
        if score > self.HIGH_CONFIDENCE_THRESHOLD:
            signal_value += 0.6
            reasons.append(f"Strong bullish sentiment ({score:+.2f})")
        elif score > self.BULLISH_THRESHOLD:
            signal_value += 0.3
            reasons.append(f"Mild bullish sentiment ({score:+.2f})")
        elif score < -self.HIGH_CONFIDENCE_THRESHOLD:
            signal_value -= 0.6
            reasons.append(f"Strong bearish sentiment ({score:+.2f})")
        elif score < self.BEARISH_THRESHOLD:
            signal_value -= 0.3
            reasons.append(f"Mild bearish sentiment ({score:+.2f})")
        else:
            reasons.append(f"Neutral sentiment ({score:+.2f})")

        # Momentum boost: sentiment is accelerating in one direction
        if abs(momentum) > 0.1:
            direction = "improving" if momentum > 0 else "deteriorating"
            signal_value += 0.2 * (1 if momentum > 0 else -1)
            reasons.append(f"Sentiment {direction}")

        # Divergence: sentiment disagrees with price (contrarian signal)
        if divergence > 0.5:
            # Divergence weakens the sentiment signal — market may be right
            signal_value *= 0.6
            reasons.append("⚠ Sentiment-price divergence detected")

        # High magnitude with many articles = higher confidence
        conf_boost = min(0.2, magnitude * 0.3) + min(0.1, volume / 50)

        # Convert signal_value to signal + confidence
        if signal_value > 0.2:
            signal = "buy"
            confidence = min(0.95, 0.5 + abs(signal_value) * 0.35 + conf_boost)
        elif signal_value < -0.2:
            signal = "sell"
            confidence = min(0.95, 0.5 + abs(signal_value) * 0.35 + conf_boost)
        else:
            signal = "hold"
            confidence = 0.5 + conf_boost

        reason = "; ".join(reasons) if reasons else "Mixed sentiment signals"
        return signal, round(confidence, 4), reason

    async def _analyze_with_llm(
        self,
        symbol: str,
        features: Dict[str, float],
        headlines: list,
        summary: dict,
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM for enhanced sentiment analysis.

        Token Optimization:
            - Sends only top 5 headlines + key metrics
            - Structured JSON output
        """
        # Build concise prompt
        top_headlines = headlines[:5]
        headlines_text = "\n".join(f"  - {h}" for h in top_headlines)

        user_prompt = f"""Symbol: {symbol}
Sentiment Score: {features.get('sentiment_score', 0):.3f}
Momentum: {features.get('sentiment_momentum', 0):.3f}
Articles: {features.get('sentiment_volume', 0)}
Divergence: {features.get('sentiment_divergence', 0):.1f}

Top Headlines:
{headlines_text}

Analyze and provide: {{"signal": "buy/sell/hold", "confidence": 0.0-1.0, "reason": "brief"}}"""

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
            "reason": response.get("reason", "LLM sentiment analysis"),
            "sentiment_summary": summary or self._make_summary(features),
            "analysis_type": "llm_enhanced",
        }

    @staticmethod
    def _make_summary(features: Dict[str, float]) -> Dict[str, Any]:
        """Create a summary dict from raw features."""
        score = features.get("sentiment_score", 0.0)
        if score > 0.15:
            label = "bullish"
        elif score < -0.15:
            label = "bearish"
        else:
            label = "neutral"

        return {
            "overall": label,
            "score": round(score, 4),
            "magnitude": round(features.get("sentiment_magnitude", 0.0), 4),
            "article_count": features.get("sentiment_volume", 0),
            "momentum": round(features.get("sentiment_momentum", 0.0), 4),
        }
