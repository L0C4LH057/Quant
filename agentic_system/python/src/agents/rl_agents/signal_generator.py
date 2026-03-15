"""
RL Signal Generator — high-level trading signal interface.

Design Decision:
    RL agents output raw continuous actions [-1, 1]. This is opaque for
    downstream trading systems. The RLSignalGenerator bridges the gap:

    1. Takes raw market data (OHLCV)
    2. Computes technical indicators automatically
    3. Feeds to the EnsembleAgent for voting
    4. Produces structured signals with:
        - Signal direction (buy/sell/hold)
        - Confidence (agreement ratio)
        - ATR-based stop-loss/take-profit
        - Position sizing based on confidence

    The stop-loss is calculated using ATR (Average True Range) which
    adapts to market volatility. High volatility → wider stops.
    Take-profit uses a configurable risk-reward ratio (default 2:1).

    Minimum confidence threshold (default 0.65) prevents acting on
    low-quality signals — when agents disagree, we hold.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .ensemble_agent import EnsembleAgent
from ...features.technical_indicators import add_all_indicators
from ...environments.trading_env import TradingEnv

logger = logging.getLogger(__name__)


class RLSignalGenerator:
    """
    High-level interface for generating trading signals from RL agents.

    Combines:
        - Technical indicator computation
        - Ensemble RL prediction
        - ATR-based risk management (stop-loss, take-profit)
        - Confidence-weighted position sizing

    Example:
        >>> generator = RLSignalGenerator(ensemble)
        >>> signal = generator.generate(market_data_df, current_price=1.0850)
        >>> print(signal["signal"], signal["confidence"])
        buy 0.85
        >>> print(signal["stop_loss"], signal["take_profit"])
        1.0810 1.0930
    """

    def __init__(
        self,
        ensemble: EnsembleAgent,
        min_confidence: float = 0.65,
        risk_reward_ratio: float = 2.0,
        atr_stop_multiplier: float = 1.5,
        max_position_pct: float = 0.05,
        window_size: int = 30,
        use_technical_indicators: bool = True,
    ):
        """
        Initialize signal generator.

        Args:
            ensemble: Trained EnsembleAgent with loaded models
            min_confidence: Minimum confidence to emit buy/sell (default: 0.65)
            risk_reward_ratio: Take-profit / stop-loss ratio (default: 2.0)
            atr_stop_multiplier: ATR multiplier for stop-loss distance
                (default: 1.5, meaning SL = 1.5 * ATR from entry)
            max_position_pct: Maximum position size as fraction of account (default: 5%)
            window_size: Observation window matching TradingEnv (default: 30)
            use_technical_indicators: Whether to calculate and add technical indicators (default: True).
                                      Set to False if models were trained on raw OHLCV data.
        """
        if ensemble.agent_count == 0:
            raise ValueError("EnsembleAgent has no agents loaded")

        self.ensemble = ensemble
        self.min_confidence = min_confidence
        self.risk_reward_ratio = risk_reward_ratio
        self.atr_stop_multiplier = atr_stop_multiplier
        self.max_position_pct = max_position_pct
        self.window_size = window_size
        self.use_technical_indicators = use_technical_indicators

        logger.info(
            f"RLSignalGenerator: min_conf={min_confidence}, "
            f"RR={risk_reward_ratio}, ATR_mult={atr_stop_multiplier}, "
            f"agents={ensemble.agent_count}, indicators={use_technical_indicators}"
        )

    def generate(
        self,
        df: pd.DataFrame,
        current_price: Optional[float] = None,
        symbol: str = "UNKNOWN",
    ) -> Dict[str, Any]:
        """
        Generate a trading signal from market data.

        Pipeline:
            1. Add technical indicators (optional)
            2. Build observation vector (same as TradingEnv format)
            3. Feed to EnsembleAgent for voting
            4. Calculate stop-loss/take-profit from ATR
            5. Size position based on confidence

        Args:
            df: Market data DataFrame with OHLCV columns.
                Must have at least window_size + 50 rows for
                indicator calculation (if enabled).
            current_price: Current price override. If None, uses
                last close from df.
            symbol: Trading symbol for the signal output.

        Returns:
            {
                "symbol": str,
                "signal": "buy" | "sell" | "hold",
                "confidence": float (0.0-1.0),
                "current_price": float,
                "stop_loss": float | None,
                "take_profit": float | None,
                "position_size_pct": float,
                "risk_reward_ratio": float,
                "agent_votes": {...},
                "indicators": {...},
            }
        """
        # Step 1: Add indicators (if enabled)
        try:
            if self.use_technical_indicators:
                df_with_indicators = add_all_indicators(df.copy())
            else:
                df_with_indicators = df.copy()
        except Exception as e:
            logger.error(f"Failed to compute indicators: {e}")
            return self._hold_signal(symbol, current_price or 0.0, reason=str(e))

        if len(df_with_indicators) < self.window_size:
            return self._hold_signal(
                symbol,
                current_price or 0.0,
                reason="Insufficient data after indicator calculation",
            )

        # Step 2: Get current price
        if current_price is None:
            current_price = float(df_with_indicators["close"].iloc[-1])

        # Step 3: Build observation vector
        obs = self._build_observation(df_with_indicators)

        # Step 4: Get ensemble prediction
        ensemble_result = self.ensemble.predict(obs, deterministic=True)

        signal = ensemble_result["signal"]
        confidence = ensemble_result["confidence"]

        # Step 5: Apply minimum confidence filter
        if signal != "hold" and confidence < self.min_confidence:
            logger.info(
                f"Signal '{signal}' rejected: confidence {confidence:.2%} "
                f"< threshold {self.min_confidence:.2%}"
            )
            signal = "hold"

        # Step 6: Calculate stop-loss and take-profit
        stop_loss = None
        take_profit = None
        atr_value = self._get_atr(df_with_indicators)

        if signal != "hold" and atr_value is not None:
            sl_distance = atr_value * self.atr_stop_multiplier
            tp_distance = sl_distance * self.risk_reward_ratio

            if signal == "buy":
                stop_loss = round(current_price - sl_distance, 5)
                take_profit = round(current_price + tp_distance, 5)
            elif signal == "sell":
                stop_loss = round(current_price + sl_distance, 5)
                take_profit = round(current_price - tp_distance, 5)

        # Step 7: Position sizing based on confidence
        if signal != "hold":
            # Scale position by confidence: higher confidence → larger position
            position_size_pct = self.max_position_pct * confidence
        else:
            position_size_pct = 0.0

        # Collect indicator snapshot
        indicators = self._get_indicator_snapshot(df_with_indicators)

        result = {
            "symbol": symbol,
            "signal": signal,
            "confidence": round(confidence, 4),
            "current_price": current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size_pct": round(position_size_pct, 6),
            "risk_reward_ratio": self.risk_reward_ratio,
            "agent_votes": ensemble_result.get("agent_votes", {}),
            "agreement_pct": ensemble_result.get("agreement_pct", 0.0),
            "raw_action": ensemble_result.get("raw_action", 0.0),
            "indicators": indicators,
        }

        logger.info(
            f"Signal: {symbol} {signal} @ {current_price} "
            f"(conf={confidence:.2%}, SL={stop_loss}, TP={take_profit}, "
            f"size={position_size_pct:.4%})"
        )

        return result

    def _build_observation(self, df: pd.DataFrame) -> np.ndarray:
        """
        Build observation vector matching TradingEnv format.

        Format: [price_window (normalized)] + [indicator_features] + [cash_ratio, holdings_ratio]

        Since we don't know the portfolio state, we use neutral values
        for cash_ratio (1.0) and holdings_ratio (0.0) — representing
        a fully-in-cash position. This is appropriate because we're
        generating a new signal, not managing an existing position.
        """
        # Price window (last window_size closes, normalized)
        prices = df["close"].values[-self.window_size:]
        if len(prices) > 1 and prices.std() > 0:
            prices_norm = (prices - prices.mean()) / prices.std()
        else:
            prices_norm = np.zeros(self.window_size)

        # Feature values (last row) — must match TradingEnv's feature detection
        exclude = {
            "date", "time", "timestamp", "datetime",
            "open", "high", "low", "close", "volume",
        }
        feature_cols = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]
        features = []
        last_row = df.iloc[-1]
        for col in feature_cols:
            val = float(last_row[col]) if not pd.isna(last_row[col]) else 0.0
            features.append(val)

        # Portfolio state (neutral: all cash, no holdings)
        portfolio_state = [1.0, 0.0]

        obs = np.concatenate([
            prices_norm.astype(np.float32),
            np.array(features, dtype=np.float32),
            np.array(portfolio_state, dtype=np.float32),
        ])

        return obs

    def _get_atr(self, df: pd.DataFrame) -> Optional[float]:
        """Get the latest ATR value from the dataframe."""
        atr_cols = [c for c in df.columns if c.startswith("atr_")]
        if atr_cols:
            val = float(df[atr_cols[0]].iloc[-1])
            return val if not np.isnan(val) and val > 0 else None
        return None

    def _get_indicator_snapshot(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get the latest indicator values."""
        exclude = {
            "date", "time", "timestamp", "datetime",
            "open", "high", "low", "close", "volume",
        }
        last_row = df.iloc[-1]
        indicators = {}
        for col in df.columns:
            if col not in exclude and pd.api.types.is_numeric_dtype(df[col]):
                val = float(last_row[col]) if not pd.isna(last_row[col]) else 0.0
                indicators[col] = round(val, 6)
        return indicators

    def _hold_signal(
        self,
        symbol: str,
        price: float,
        reason: str = "",
    ) -> Dict[str, Any]:
        """Return a safe hold signal."""
        return {
            "symbol": symbol,
            "signal": "hold",
            "confidence": 0.0,
            "current_price": price,
            "stop_loss": None,
            "take_profit": None,
            "position_size_pct": 0.0,
            "risk_reward_ratio": self.risk_reward_ratio,
            "agent_votes": {},
            "agreement_pct": 0.0,
            "raw_action": 0.0,
            "indicators": {},
            "reason": reason,
        }
