"""
Tests for the Signal Intelligence module.
"""
import pytest
import numpy as np
import pandas as pd

from src.agents.signal_intelligence import (
    MarketRegimeDetector,
    MarketRegime,
    SignalTransitionDetector,
    SignalTransition,
    UnifiedSignalArbiter,
    ArbitrationResult,
)


# ───────── Fixtures ─────────

@pytest.fixture
def trending_data():
    """Generate data with a clear upward trend."""
    np.random.seed(42)
    n = 100
    base = np.linspace(100, 130, n)  # strong upward trend
    noise = np.random.randn(n) * 0.2
    close = base + noise
    return pd.DataFrame({
        "open": close - np.abs(np.random.randn(n) * 0.1),
        "high": close + np.abs(np.random.randn(n) * 0.5),
        "low": close - np.abs(np.random.randn(n) * 0.5),
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })


@pytest.fixture
def consolidating_data():
    """Generate perfectly flat, range-bound data with negligible variation."""
    np.random.seed(99)
    n = 100
    # Almost constant close prices — negligible movement
    close = np.full(n, 100.0)
    # High and low are identical to close (no range at all)
    return pd.DataFrame({
        "open": close,
        "high": close + 0.001,    # Virtually zero range
        "low": close - 0.001,
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    })


@pytest.fixture
def volatile_data():
    """Generate highly volatile data."""
    np.random.seed(77)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 3.0)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 1.5,
        "high": close + np.abs(np.random.randn(n) * 4.0),
        "low": close - np.abs(np.random.randn(n) * 4.0),
        "close": close,
        "volume": np.random.randint(1000, 50000, n),
    })


@pytest.fixture
def regime_detector():
    return MarketRegimeDetector()


@pytest.fixture
def transition_detector():
    return SignalTransitionDetector(window_size=10)


@pytest.fixture
def arbiter():
    return UnifiedSignalArbiter()


# ───────── MarketRegimeDetector Tests ─────────

class TestMarketRegimeDetector:
    """Test market regime classification."""

    def test_detect_consolidating_market(self, regime_detector, consolidating_data):
        """Flat data should be classified as consolidating."""
        result = regime_detector.detect(consolidating_data)
        assert isinstance(result, MarketRegime)
        assert result.regime == "consolidating"
        assert result.consolidation_score >= 0.5

    def test_detect_trending_market(self, regime_detector, trending_data):
        """Trending data should have low consolidation score."""
        result = regime_detector.detect(trending_data)
        assert isinstance(result, MarketRegime)
        # Trending markets should have lower consolidation scores
        assert result.consolidation_score < 0.65

    def test_insufficient_data_defaults(self, regime_detector):
        """Short data should return default consolidating regime."""
        short_df = pd.DataFrame({
            "open": [100, 101], "high": [102, 103],
            "low": [99, 100], "close": [101, 102],
        })
        result = regime_detector.detect(short_df)
        assert result.regime == "consolidating"
        assert result.details.get("reason") == "insufficient_data"

    def test_regime_fields_populated(self, regime_detector, trending_data):
        """All MarketRegime fields should be populated."""
        result = regime_detector.detect(trending_data)
        assert result.adx >= 0
        assert result.atr_ratio >= 0
        assert result.bb_squeeze >= 0
        assert 0 <= result.consolidation_score <= 1

    def test_volatile_market_detected(self, regime_detector, volatile_data):
        """Highly volatile data should be identified as volatile or trending, not consolidating."""
        result = regime_detector.detect(volatile_data)
        assert result.regime in ("volatile", "trending")
        assert result.consolidation_score < 0.65


# ───────── SignalTransitionDetector Tests ─────────

class TestSignalTransitionDetector:
    """Test signal transition detection and alerts."""

    def test_no_transition_on_first_signal(self, transition_detector):
        """First signal should not generate an alert."""
        alerts = transition_detector.update("buy", 0.8)
        assert alerts == []

    def test_same_signal_no_transition(self, transition_detector):
        """Repeated same signal should not generate alerts."""
        transition_detector.update("buy", 0.8)
        alerts = transition_detector.update("buy", 0.7)
        assert alerts == []

    def test_hold_to_buy_is_info(self, transition_detector):
        """hold → buy transition should be info severity."""
        transition_detector.update("hold", 0.5)
        alerts = transition_detector.update("buy", 0.8)
        assert len(alerts) == 1
        assert alerts[0].severity == "info"
        assert alerts[0].previous_signal == "hold"
        assert alerts[0].new_signal == "buy"

    def test_buy_to_sell_is_critical(self, transition_detector):
        """buy → sell transition should be critical severity."""
        transition_detector.update("buy", 0.8)
        alerts = transition_detector.update("sell", 0.7)
        assert len(alerts) == 1
        assert alerts[0].severity == "critical"

    def test_buy_to_hold_is_warning(self, transition_detector):
        """buy → hold transition should be warning severity."""
        transition_detector.update("buy", 0.8)
        alerts = transition_detector.update("hold", 0.5)
        assert len(alerts) == 1
        assert alerts[0].severity == "warning"

    def test_momentum_stable_signals(self, transition_detector):
        """All same signals should have stability 1.0."""
        for _ in range(5):
            transition_detector.update("buy", 0.8)
        momentum = transition_detector.get_momentum()
        assert momentum["stability"] == 1.0
        assert momentum["dominant_signal"] == "buy"
        assert momentum["changes_in_window"] == 0

    def test_momentum_rapid_changes(self, transition_detector):
        """Alternating signals should have low stability."""
        for i in range(10):
            sig = "buy" if i % 2 == 0 else "sell"
            transition_detector.update(sig, 0.5)
        momentum = transition_detector.get_momentum()
        assert momentum["stability"] < 0.2
        assert momentum["changes_in_window"] >= 8

    def test_reset_clears_history(self, transition_detector):
        """Reset should clear all history."""
        transition_detector.update("buy", 0.8)
        transition_detector.update("sell", 0.7)
        transition_detector.reset()
        momentum = transition_detector.get_momentum()
        assert momentum["changes_in_window"] == 0


# ───────── UnifiedSignalArbiter Tests ─────────

class TestUnifiedSignalArbiter:
    """Test signal arbitration logic."""

    def _make_regime(self, regime_type: str, score: float = 0.5) -> MarketRegime:
        """Helper to build a MarketRegime."""
        return MarketRegime(
            regime=regime_type,
            consolidation_score=score,
            adx=25 if regime_type == "trending" else 15,
            atr_ratio=0.005,
            bb_squeeze=0.02,
        )

    def test_agreement_boosts_confidence(self, arbiter):
        """When both agree, confidence should be boosted."""
        regime = self._make_regime("trending", 0.2)
        result = arbiter.arbitrate("buy", 0.8, "buy", 0.7, regime)
        assert result.final_signal == "buy"
        assert result.source == "agreement"
        assert result.final_confidence > 0.7  # boosted

    def test_consolidation_forces_hold(self, arbiter):
        """In consolidation with low confidence, should force HOLD."""
        regime = self._make_regime("consolidating", 0.8)
        result = arbiter.arbitrate("buy", 0.6, "buy", 0.5, regime)
        assert result.final_signal == "hold"
        assert result.source == "consolidation_hold"

    def test_consolidation_overridden_by_high_confidence(self, arbiter):
        """In consolidation, very high confidence should still trade."""
        regime = self._make_regime("consolidating", 0.7)
        result = arbiter.arbitrate("buy", 0.85, "buy", 0.80, regime)
        assert result.final_signal == "buy"
        assert result.source == "agreement"

    def test_instability_forces_hold(self, arbiter):
        """Rapid signal changes should force HOLD via instability check."""
        regime = self._make_regime("trending", 0.2)
        momentum = {"stability": 0.2, "changes_in_window": 8}
        result = arbiter.arbitrate("buy", 0.8, "sell", 0.7, regime, momentum)
        assert result.final_signal == "hold"
        assert result.source == "instability_hold"

    def test_one_hold_other_directional(self, arbiter):
        """When one says hold and other has direction, use directional if confident."""
        regime = self._make_regime("trending", 0.3)
        result = arbiter.arbitrate("buy", 0.8, "hold", 0.5, regime)
        assert result.final_signal == "buy"
        assert result.source == "rl_dominant"

    def test_disagreement_buy_vs_sell(self, arbiter):
        """Full disagreement (buy vs sell) should resolve via weighted confidence."""
        regime = self._make_regime("trending", 0.3)
        result = arbiter.arbitrate("buy", 0.9, "sell", 0.5, regime)
        # RL has higher confidence and gets a trending boost
        assert result.final_signal == "buy"
        assert result.source == "rl_dominant"

    def test_disagreement_low_confidence_holds(self, arbiter):
        """Full disagreement with low confidence should default to HOLD."""
        regime = self._make_regime("consolidating", 0.7)
        result = arbiter.arbitrate("buy", 0.5, "sell", 0.4, regime)
        assert result.final_signal == "hold"

    def test_result_structure(self, arbiter):
        """Verify all fields are present in ArbitrationResult."""
        regime = self._make_regime("trending", 0.3)
        result = arbiter.arbitrate("buy", 0.8, "buy", 0.7, regime)
        assert isinstance(result, ArbitrationResult)
        assert hasattr(result, "final_signal")
        assert hasattr(result, "final_confidence")
        assert hasattr(result, "rl_signal")
        assert hasattr(result, "specialist_signal")
        assert hasattr(result, "regime")
        assert hasattr(result, "source")
        assert hasattr(result, "alerts")
        assert hasattr(result, "reason")
