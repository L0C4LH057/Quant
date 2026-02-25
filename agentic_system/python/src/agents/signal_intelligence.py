"""
Signal Intelligence Module — Market regime detection, signal transition tracking, and signal arbitration.

This module provides three core components that work together to produce robust
trading signals by combining RL ensemble and specialist agent outputs:

1. MarketRegimeDetector: Classify the market as trending, consolidating, or volatile.
2. SignalTransitionDetector: Track signal history and alert on regime/signal changes.
3. UnifiedSignalArbiter: Merge RL ensemble + specialist signals into a final signal
   with market-regime-aware weighting and consolidation HOLD bias.
"""
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MarketRegime:
    """Result of market regime analysis."""

    regime: str  # "trending", "consolidating", "volatile"
    consolidation_score: float  # 0.0 (strong trend) – 1.0 (deep consolidation)
    adx: float
    atr_ratio: float  # ATR / price  (normalised volatility)
    bb_squeeze: float  # Bollinger Band width ratio (narrow = squeeze)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalTransition:
    """A detected change in signal direction."""

    previous_signal: str
    new_signal: str
    timestamp: str
    severity: str  # "info", "warning", "critical"
    message: str


@dataclass
class ArbitrationResult:
    """Unified output of the signal arbiter."""

    final_signal: str  # "buy", "sell", "hold"
    final_confidence: float
    rl_signal: str
    rl_confidence: float
    specialist_signal: str
    specialist_confidence: float
    regime: MarketRegime
    source: str  # "agreement", "rl_dominant", "specialist_dominant", "consolidation_hold"
    alerts: List[SignalTransition] = field(default_factory=list)
    reason: str = ""


# ---------------------------------------------------------------------------
# MarketRegimeDetector
# ---------------------------------------------------------------------------

class MarketRegimeDetector:
    """
    Detect whether the market is trending, consolidating, or volatile.

    Uses three complementary indicators:
        - ADX (Average Directional Index): < 20 ⇒ consolidating, > 25 ⇒ trending
        - ATR compression: normalised ATR ratio; low ⇒ consolidation
        - Bollinger Band squeeze: narrow bands ⇒ consolidation

    The consolidation_score is a weighted blend: 50% ADX + 25% ATR + 25% BB.
    """

    def __init__(
        self,
        adx_period: int = 14,
        atr_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        adx_trend_threshold: float = 25.0,
        adx_consolidation_threshold: float = 20.0,
    ):
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_consolidation_threshold = adx_consolidation_threshold

    def detect(self, df: pd.DataFrame) -> MarketRegime:
        """
        Analyse a DataFrame (must have high, low, close) and return the regime.

        Args:
            df: OHLCV DataFrame with at least ``adx_period + bb_period`` rows.

        Returns:
            MarketRegime with classified regime and scores.
        """
        if len(df) < max(self.adx_period, self.bb_period) + 10:
            return MarketRegime(
                regime="consolidating",
                consolidation_score=0.5,
                adx=0,
                atr_ratio=0,
                bb_squeeze=0,
                details={"reason": "insufficient_data"},
            )

        # --- ADX ---
        adx_val = self._compute_adx(df)

        # --- ATR ratio ---
        atr_val = self._compute_atr(df)
        current_price = float(df["close"].iloc[-1])
        atr_ratio = atr_val / current_price if current_price > 0 else 0

        # --- Bollinger Band squeeze ---
        bb_squeeze = self._compute_bb_squeeze(df)

        # --- Consolidation score (0 = trending, 1 = deep consolidation) ---
        # ADX component: linear scale from trend threshold (0) to 0 (1)
        if adx_val >= self.adx_trend_threshold:
            adx_score = 0.0
        elif adx_val <= self.adx_consolidation_threshold:
            adx_score = 1.0
        else:
            adx_score = 1.0 - (adx_val - self.adx_consolidation_threshold) / (
                self.adx_trend_threshold - self.adx_consolidation_threshold
            )

        # ATR component: lower ATR ratio ⇒ more consolidation
        # Normalize: typical forex ATR/price ranges 0.001–0.02
        atr_score = max(0.0, min(1.0, 1.0 - (atr_ratio / 0.015)))

        # BB squeeze component: lower squeeze ⇒ more consolidation
        bb_score = max(0.0, min(1.0, 1.0 - bb_squeeze))

        consolidation_score = round(
            0.50 * adx_score + 0.25 * atr_score + 0.25 * bb_score, 4
        )

        # --- Classify ---
        if consolidation_score >= 0.65:
            regime = "consolidating"
        elif atr_ratio > 0.015 or adx_val > 30:
            regime = "trending" if adx_val > self.adx_trend_threshold else "volatile"
        else:
            regime = "trending" if adx_val > self.adx_trend_threshold else "consolidating"

        return MarketRegime(
            regime=regime,
            consolidation_score=consolidation_score,
            adx=round(adx_val, 2),
            atr_ratio=round(atr_ratio, 6),
            bb_squeeze=round(bb_squeeze, 4),
            details={
                "adx_score": round(adx_score, 4),
                "atr_score": round(atr_score, 4),
                "bb_score": round(bb_score, 4),
            },
        )

    # ---- internal helpers ----

    def _compute_adx(self, df: pd.DataFrame) -> float:
        """Compute ADX manually (no external dependency needed)."""
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        close = df["close"].values.astype(float)
        n = self.adx_period

        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )

        # +DM and -DM
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Smoothed averages (Wilder's smoothing)
        def wilder_smooth(arr: np.ndarray, period: int) -> np.ndarray:
            out = np.zeros_like(arr)
            out[period - 1] = arr[:period].sum()
            for i in range(period, len(arr)):
                out[i] = out[i - 1] - out[i - 1] / period + arr[i]
            return out

        atr_smooth = wilder_smooth(tr, n)
        plus_di_smooth = wilder_smooth(plus_dm, n)
        minus_di_smooth = wilder_smooth(minus_dm, n)

        # +DI and -DI
        safe_atr = np.where(atr_smooth > 0, atr_smooth, 1.0)
        plus_di = 100.0 * plus_di_smooth / safe_atr
        minus_di = 100.0 * minus_di_smooth / safe_atr

        # DX
        di_sum = plus_di + minus_di
        di_diff = np.abs(plus_di - minus_di)
        safe_sum = np.where(di_sum > 0, di_sum, 1.0)
        dx = 100.0 * di_diff / safe_sum

        # ADX (smoothed DX)
        adx = wilder_smooth(dx, n)
        return float(adx[-1]) if len(adx) > 0 else 0.0

    def _compute_atr(self, df: pd.DataFrame) -> float:
        """Compute latest ATR."""
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        close = df["close"].values.astype(float)

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        # Simple moving average of TR for the period
        if len(tr) >= self.atr_period:
            return float(np.mean(tr[-self.atr_period:]))
        return float(np.mean(tr)) if len(tr) > 0 else 0.0

    def _compute_bb_squeeze(self, df: pd.DataFrame) -> float:
        """
        Compute Bollinger Band Width ratio (squeeze indicator).

        Low value ⇒ squeeze ⇒ consolidation.
        """
        close = df["close"].values.astype(float)
        if len(close) < self.bb_period:
            return 0.5

        sma = np.mean(close[-self.bb_period:])
        std = np.std(close[-self.bb_period:])

        if sma <= 0:
            return 0.5

        upper = sma + self.bb_std * std
        lower = sma - self.bb_std * std
        width = (upper - lower) / sma
        return float(width)


# ---------------------------------------------------------------------------
# SignalTransitionDetector
# ---------------------------------------------------------------------------

class SignalTransitionDetector:
    """
    Track recent signals and detect transitions between buy/sell/hold.

    Maintains a sliding window of recent signals and detects when
    a transition occurs, classifying it by severity:
        - info: hold → buy, hold → sell (entering a position)
        - warning: buy → hold, sell → hold (exiting a position)
        - critical: buy → sell, sell → buy (reversal)

    Also computes signal momentum to measure how rapidly signals shift.
    """

    # Severity mapping for transitions
    TRANSITION_SEVERITY = {
        ("hold", "buy"): "info",
        ("hold", "sell"): "info",
        ("buy", "hold"): "warning",
        ("sell", "hold"): "warning",
        ("buy", "sell"): "critical",
        ("sell", "buy"): "critical",
    }

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.signal_history: deque = deque(maxlen=window_size)

    def update(self, signal: str, confidence: float = 0.0) -> List[SignalTransition]:
        """
        Record a new signal and return any detected transitions.

        Args:
            signal: The new signal ("buy", "sell", or "hold").
            confidence: Confidence of the new signal (0–1).

        Returns:
            List of SignalTransition alerts (empty if no transition).
        """
        signal = signal.lower()
        now = datetime.now().strftime("%H:%M:%S")
        alerts: List[SignalTransition] = []

        if self.signal_history:
            prev_signal = self.signal_history[-1]["signal"]
            if prev_signal != signal:
                severity = self.TRANSITION_SEVERITY.get(
                    (prev_signal, signal), "info"
                )
                message = self._format_message(prev_signal, signal, severity, confidence)
                alerts.append(
                    SignalTransition(
                        previous_signal=prev_signal,
                        new_signal=signal,
                        timestamp=now,
                        severity=severity,
                        message=message,
                    )
                )

        self.signal_history.append({
            "signal": signal,
            "confidence": confidence,
            "time": now,
        })

        return alerts

    def get_momentum(self) -> Dict[str, Any]:
        """
        Calculate signal momentum — how rapidly signals are shifting.

        Returns:
            {
                "changes_in_window": int,
                "stability": float (0-1, 1 = all same),
                "dominant_signal": str,
                "distribution": {"buy": int, "sell": int, "hold": int},
            }
        """
        if not self.signal_history:
            return {
                "changes_in_window": 0,
                "stability": 1.0,
                "dominant_signal": "hold",
                "distribution": {"buy": 0, "sell": 0, "hold": 0},
            }

        signals = [s["signal"] for s in self.signal_history]
        changes = sum(1 for i in range(1, len(signals)) if signals[i] != signals[i - 1])
        stability = 1.0 - (changes / max(len(signals) - 1, 1))

        dist = {"buy": 0, "sell": 0, "hold": 0}
        for s in signals:
            dist[s] = dist.get(s, 0) + 1
        dominant = max(dist, key=dist.get)

        return {
            "changes_in_window": changes,
            "stability": round(stability, 4),
            "dominant_signal": dominant,
            "distribution": dist,
        }

    def reset(self) -> None:
        """Clear signal history."""
        self.signal_history.clear()

    @staticmethod
    def _format_message(
        prev: str, new: str, severity: str, confidence: float
    ) -> str:
        icons = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}
        icon = icons.get(severity, "ℹ️")
        return (
            f"{icon} Signal transition: {prev.upper()} → {new.upper()} "
            f"(confidence: {confidence:.0%})"
        )


# ---------------------------------------------------------------------------
# UnifiedSignalArbiter
# ---------------------------------------------------------------------------

class UnifiedSignalArbiter:
    """
    Merge RL ensemble and specialist agent signals into a single final signal.

    Weighting logic:
        - Agreement: use agreed signal with boosted confidence.
        - Disagreement: weighted comparison (RL 60% / specialist 40%)
          adjusted by market regime.
        - Consolidation bias: require higher confidence to issue buy/sell;
          default to HOLD when uncertain.
        - Signal instability: if signals are changing rapidly, bias toward HOLD.

    Args:
        rl_weight: base weight for RL signal (default 0.60)
        specialist_weight: base weight for specialist signal (default 0.40)
        consolidation_confidence_threshold: min confidence to override HOLD
            during consolidation (default 0.75)
        instability_hold_threshold: if signal stability < this, force HOLD
            (default 0.4)
    """

    def __init__(
        self,
        rl_weight: float = 0.60,
        specialist_weight: float = 0.40,
        consolidation_confidence_threshold: float = 0.75,
        instability_hold_threshold: float = 0.40,
    ):
        self.rl_weight = rl_weight
        self.specialist_weight = specialist_weight
        self.consolidation_confidence_threshold = consolidation_confidence_threshold
        self.instability_hold_threshold = instability_hold_threshold

    def arbitrate(
        self,
        rl_signal: str,
        rl_confidence: float,
        specialist_signal: str,
        specialist_confidence: float,
        regime: MarketRegime,
        signal_momentum: Optional[Dict[str, Any]] = None,
    ) -> ArbitrationResult:
        """
        Produce a final trading signal by merging RL and specialist outputs.

        Args:
            rl_signal: Signal from RL ensemble ("buy", "sell", "hold").
            rl_confidence: RL signal confidence (0–1).
            specialist_signal: Signal from specialist agent.
            specialist_confidence: Specialist signal confidence.
            regime: Market regime from MarketRegimeDetector.
            signal_momentum: Output of SignalTransitionDetector.get_momentum()

        Returns:
            ArbitrationResult with final signal and breakdown.
        """
        rl_signal = rl_signal.lower()
        specialist_signal = specialist_signal.lower()

        # 1. Check signal instability — too many rapid changes ⇒ HOLD
        if signal_momentum and signal_momentum.get("stability", 1.0) < self.instability_hold_threshold:
            return ArbitrationResult(
                final_signal="hold",
                final_confidence=0.5,
                rl_signal=rl_signal,
                rl_confidence=rl_confidence,
                specialist_signal=specialist_signal,
                specialist_confidence=specialist_confidence,
                regime=regime,
                source="instability_hold",
                reason=(
                    f"Signal instability detected (stability="
                    f"{signal_momentum['stability']:.2f}). Holding to avoid whipsaw."
                ),
            )

        # 2. Consolidation bias — if market is consolidating, require higher confidence
        if regime.regime == "consolidating":
            max_conf = max(rl_confidence, specialist_confidence)
            if max_conf < self.consolidation_confidence_threshold:
                return ArbitrationResult(
                    final_signal="hold",
                    final_confidence=regime.consolidation_score,
                    rl_signal=rl_signal,
                    rl_confidence=rl_confidence,
                    specialist_signal=specialist_signal,
                    specialist_confidence=specialist_confidence,
                    regime=regime,
                    source="consolidation_hold",
                    reason=(
                        f"Market consolidating (score={regime.consolidation_score:.2f}). "
                        f"Max confidence {max_conf:.2%} < threshold "
                        f"{self.consolidation_confidence_threshold:.2%}. Emphasizing HOLD."
                    ),
                )

        # 3. Agreement — both agree
        if rl_signal == specialist_signal:
            boosted_confidence = min(
                1.0,
                (rl_confidence * self.rl_weight + specialist_confidence * self.specialist_weight) * 1.15,
            )
            return ArbitrationResult(
                final_signal=rl_signal,
                final_confidence=round(boosted_confidence, 4),
                rl_signal=rl_signal,
                rl_confidence=rl_confidence,
                specialist_signal=specialist_signal,
                specialist_confidence=specialist_confidence,
                regime=regime,
                source="agreement",
                reason=f"Both RL and specialist agree on {rl_signal.upper()} — boosted confidence.",
            )

        # 4. One says hold, the other has a directional signal
        if rl_signal == "hold" or specialist_signal == "hold":
            # Use the non-hold signal if its confidence is high enough
            if rl_signal == "hold":
                active_signal = specialist_signal
                active_conf = specialist_confidence
                source = "specialist_dominant"
            else:
                active_signal = rl_signal
                active_conf = rl_confidence
                source = "rl_dominant"

            # In consolidation, require higher bar
            threshold = (
                self.consolidation_confidence_threshold
                if regime.regime == "consolidating"
                else 0.60
            )
            if active_conf >= threshold:
                return ArbitrationResult(
                    final_signal=active_signal,
                    final_confidence=round(active_conf * 0.9, 4),  # slight discount
                    rl_signal=rl_signal,
                    rl_confidence=rl_confidence,
                    specialist_signal=specialist_signal,
                    specialist_confidence=specialist_confidence,
                    regime=regime,
                    source=source,
                    reason=(
                        f"{source.replace('_', ' ').title()}: {active_signal.upper()} "
                        f"at {active_conf:.2%} (other says HOLD)."
                    ),
                )
            else:
                return ArbitrationResult(
                    final_signal="hold",
                    final_confidence=0.5,
                    rl_signal=rl_signal,
                    rl_confidence=rl_confidence,
                    specialist_signal=specialist_signal,
                    specialist_confidence=specialist_confidence,
                    regime=regime,
                    source="disagree_hold",
                    reason=(
                        f"Disagreement (RL={rl_signal}, Specialist={specialist_signal}). "
                        f"Active confidence {active_conf:.2%} too low. Holding."
                    ),
                )

        # 5. Full disagreement (buy vs sell) — pick higher weighted confidence
        rl_weighted = rl_confidence * self._regime_adjusted_weight(regime, "rl")
        spec_weighted = specialist_confidence * self._regime_adjusted_weight(regime, "specialist")

        if rl_weighted >= spec_weighted:
            winner_signal = rl_signal
            winner_confidence = rl_confidence
            source = "rl_dominant"
        else:
            winner_signal = specialist_signal
            winner_confidence = specialist_confidence
            source = "specialist_dominant"

        # Disagreement penalty — reduce confidence
        final_confidence = round(winner_confidence * 0.75, 4)

        # In consolidation or high disagreement, default to hold
        if final_confidence < 0.55:
            return ArbitrationResult(
                final_signal="hold",
                final_confidence=final_confidence,
                rl_signal=rl_signal,
                rl_confidence=rl_confidence,
                specialist_signal=specialist_signal,
                specialist_confidence=specialist_confidence,
                regime=regime,
                source="disagree_hold",
                reason=(
                    f"RL says {rl_signal.upper()}, Specialist says {specialist_signal.upper()}. "
                    f"Weighted confidence {final_confidence:.2%} too low after disagreement penalty. Holding."
                ),
            )

        return ArbitrationResult(
            final_signal=winner_signal,
            final_confidence=final_confidence,
            rl_signal=rl_signal,
            rl_confidence=rl_confidence,
            specialist_signal=specialist_signal,
            specialist_confidence=specialist_confidence,
            regime=regime,
            source=source,
            reason=(
                f"Disagreement resolved: {source.replace('_', ' ').title()} wins. "
                f"{winner_signal.upper()} at {final_confidence:.2%} (penalty applied)."
            ),
        )

    def _regime_adjusted_weight(self, regime: MarketRegime, agent_type: str) -> float:
        """Adjust weight based on market regime."""
        base_weight = self.rl_weight if agent_type == "rl" else self.specialist_weight

        if regime.regime == "trending":
            # RL tends to learn trend-following; boost it
            return base_weight * (1.1 if agent_type == "rl" else 0.9)
        elif regime.regime == "volatile":
            # Specialist rules are more interpretable in volatile conditions
            return base_weight * (0.9 if agent_type == "rl" else 1.1)
        else:
            return base_weight

