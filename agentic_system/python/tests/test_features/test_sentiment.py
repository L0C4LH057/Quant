"""
Tests for the sentiment analysis modules.

Covers:
    - SentimentAnalyzer: VADER scoring, feature engineering, summaries
    - SentimentAnalysisAgent: rule-based signal generation
    - UnifiedSignalArbiter: 3-way merge and backward compatibility
"""
import asyncio
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_news_df():
    """Create a sample news DataFrame."""
    now = datetime.now()
    return pd.DataFrame({
        "datetime": [
            now - timedelta(hours=i) for i in range(10)
        ],
        "headline": [
            "Company reports strong quarterly earnings beating expectations",
            "Stock market rallies on positive economic data",
            "Investors remain cautious amid uncertainty",
            "Tech sector sees massive selloff",
            "Federal Reserve signals potential rate pause",
            "New product launch exceeds sales targets",
            "Global recession fears grow among analysts",
            "Market volatility hits yearly high",
            "Employment numbers better than expected",
            "Trade tensions escalate between major economies",
        ],
        "source": ["Reuters"] * 10,
        "url": [f"https://example.com/{i}" for i in range(10)],
        "raw_sentiment": [0.0] * 10,
        "relevance_score": [0.8] * 10,
    })


@pytest.fixture
def sample_market_df():
    """Create a sample market OHLCV DataFrame."""
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(10))
    return pd.DataFrame({
        "date": dates,
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.random.randint(1000, 10000, 10),
    })


@pytest.fixture
def bullish_features():
    """Sentiment features indicating bullish sentiment."""
    return {
        "sentiment_score": 0.45,
        "sentiment_magnitude": 0.5,
        "sentiment_volume": 15,
        "sentiment_momentum": 0.1,
        "sentiment_divergence": 0.0,
    }


@pytest.fixture
def bearish_features():
    """Sentiment features indicating bearish sentiment."""
    return {
        "sentiment_score": -0.40,
        "sentiment_magnitude": 0.45,
        "sentiment_volume": 12,
        "sentiment_momentum": -0.1,
        "sentiment_divergence": 0.0,
    }


@pytest.fixture
def neutral_features():
    """Sentiment features indicating neutral sentiment."""
    return {
        "sentiment_score": 0.02,
        "sentiment_magnitude": 0.1,
        "sentiment_volume": 5,
        "sentiment_momentum": 0.0,
        "sentiment_divergence": 0.0,
    }


# ---------------------------------------------------------------------------
# Tests: SentimentAnalyzer — VADER scoring
# ---------------------------------------------------------------------------

class TestVaderScoring:
    """Tests for VADER headline scoring."""

    def test_positive_headline_scores_positive(self):
        from python.src.features.sentiment_analyzer import score_headlines_vader

        scores = score_headlines_vader(["Company reports fantastic earnings growth"])
        assert len(scores) == 1
        assert scores[0]["compound"] > 0.0, "Positive headline should score positive"

    def test_negative_headline_scores_negative(self):
        from python.src.features.sentiment_analyzer import score_headlines_vader

        scores = score_headlines_vader(["Stock market crashes amid widespread panic"])
        assert len(scores) == 1
        assert scores[0]["compound"] < 0.0, "Negative headline should score negative"

    def test_empty_headline_scores_neutral(self):
        from python.src.features.sentiment_analyzer import score_headlines_vader

        scores = score_headlines_vader([""])
        assert scores[0]["compound"] == 0.0

    def test_none_headline_handled(self):
        from python.src.features.sentiment_analyzer import score_headlines_vader

        scores = score_headlines_vader([None])
        assert scores[0]["compound"] == 0.0

    def test_batch_scoring(self):
        from python.src.features.sentiment_analyzer import score_headlines_vader

        headlines = [
            "Great earnings report",
            "Terrible market crash",
            "Weather is cloudy today",
        ]
        scores = score_headlines_vader(headlines)
        assert len(scores) == 3
        assert all("compound" in s for s in scores)


# ---------------------------------------------------------------------------
# Tests: SentimentAnalyzer — feature engineering
# ---------------------------------------------------------------------------

class TestSentimentFeatures:
    """Tests for sentiment feature engineering."""

    def test_compute_sentiment_features_adds_score(self, sample_news_df):
        from python.src.features.sentiment_analyzer import compute_sentiment_features

        result = compute_sentiment_features(sample_news_df, model="vader")
        assert "sentiment_score" in result.columns
        assert len(result) == len(sample_news_df)
        # Scores should be in [-1, 1] range
        assert result["sentiment_score"].min() >= -1.0
        assert result["sentiment_score"].max() <= 1.0

    def test_compute_sentiment_features_empty_df(self):
        from python.src.features.sentiment_analyzer import compute_sentiment_features

        empty_df = pd.DataFrame(columns=["headline", "datetime"])
        result = compute_sentiment_features(empty_df, model="vader")
        assert "sentiment_score" in result.columns
        assert len(result) == 0

    def test_add_sentiment_features_produces_all_columns(
        self, sample_market_df, sample_news_df
    ):
        from python.src.features.sentiment_analyzer import add_sentiment_features

        result = add_sentiment_features(sample_market_df, sample_news_df)
        expected_cols = [
            "sentiment_score",
            "sentiment_magnitude",
            "sentiment_volume",
            "sentiment_momentum",
            "sentiment_divergence",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_add_sentiment_features_empty_news(self, sample_market_df):
        from python.src.features.sentiment_analyzer import add_sentiment_features

        empty_news = pd.DataFrame(columns=[
            "datetime", "headline", "source", "url",
            "raw_sentiment", "relevance_score",
        ])
        result = add_sentiment_features(sample_market_df, empty_news)
        # Should fill with neutral defaults
        assert (result["sentiment_score"] == 0.0).all()
        assert (result["sentiment_volume"] == 0).all()

    def test_get_sentiment_summary(self, sample_news_df):
        from python.src.features.sentiment_analyzer import (
            compute_sentiment_features,
            get_sentiment_summary,
        )

        scored = compute_sentiment_features(sample_news_df, model="vader")
        summary = get_sentiment_summary(scored)
        assert "mean_sentiment" in summary
        assert "article_count" in summary
        assert summary["article_count"] == 10
        assert 0.0 <= summary["bullish_pct"] <= 1.0
        assert 0.0 <= summary["bearish_pct"] <= 1.0


# ---------------------------------------------------------------------------
# Tests: SentimentAnalysisAgent — rule-based
# ---------------------------------------------------------------------------

class TestSentimentAgent:
    """Tests for the sentiment analysis agent."""

    def test_bullish_signal(self, bullish_features):
        from python.src.agents.specialized.sentiment_analysis_agent import (
            SentimentAnalysisAgent,
        )

        agent = SentimentAnalysisAgent()
        result = asyncio.get_event_loop().run_until_complete(
            agent.process({
                "symbol": "AAPL",
                "sentiment_features": bullish_features,
            })
        )
        assert result["signal"] == "buy"
        assert 0.0 <= result["confidence"] <= 1.0

    def test_bearish_signal(self, bearish_features):
        from python.src.agents.specialized.sentiment_analysis_agent import (
            SentimentAnalysisAgent,
        )

        agent = SentimentAnalysisAgent()
        result = asyncio.get_event_loop().run_until_complete(
            agent.process({
                "symbol": "AAPL",
                "sentiment_features": bearish_features,
            })
        )
        assert result["signal"] == "sell"
        assert 0.0 <= result["confidence"] <= 1.0

    def test_neutral_signal(self, neutral_features):
        from python.src.agents.specialized.sentiment_analysis_agent import (
            SentimentAnalysisAgent,
        )

        agent = SentimentAnalysisAgent()
        result = asyncio.get_event_loop().run_until_complete(
            agent.process({
                "symbol": "AAPL",
                "sentiment_features": neutral_features,
            })
        )
        assert result["signal"] == "hold"

    def test_low_volume_forces_hold(self):
        from python.src.agents.specialized.sentiment_analysis_agent import (
            SentimentAnalysisAgent,
        )

        agent = SentimentAnalysisAgent()
        features = {
            "sentiment_score": 0.8,  # Very bullish but...
            "sentiment_magnitude": 0.9,
            "sentiment_volume": 1,   # ... only 1 article
            "sentiment_momentum": 0.2,
            "sentiment_divergence": 0.0,
        }
        result = asyncio.get_event_loop().run_until_complete(
            agent.process({"symbol": "AAPL", "sentiment_features": features})
        )
        assert result["signal"] == "hold"
        assert result["confidence"] < 0.5

    def test_output_format(self, bullish_features):
        from python.src.agents.specialized.sentiment_analysis_agent import (
            SentimentAnalysisAgent,
        )

        agent = SentimentAnalysisAgent()
        result = asyncio.get_event_loop().run_until_complete(
            agent.process({
                "symbol": "AAPL",
                "sentiment_features": bullish_features,
            })
        )
        assert "signal" in result
        assert "confidence" in result
        assert "reason" in result
        assert "sentiment_summary" in result
        assert result["signal"] in ("buy", "sell", "hold")


# ---------------------------------------------------------------------------
# Tests: UnifiedSignalArbiter — 3-way merge + backward compatibility
# ---------------------------------------------------------------------------

class TestArbiter3Way:
    """Tests for the 3-way signal arbiter integration."""

    def _make_regime(self, regime_type="trending"):
        from python.src.agents.signal_intelligence import MarketRegime

        return MarketRegime(
            regime=regime_type,
            consolidation_score=0.3 if regime_type == "trending" else 0.8,
            adx=30.0 if regime_type == "trending" else 15.0,
            atr_ratio=0.01,
            bb_squeeze=0.03,
        )

    def test_backward_compatibility_no_sentiment(self):
        """Arbiter works exactly as before when no sentiment is provided."""
        from python.src.agents.signal_intelligence import UnifiedSignalArbiter

        arbiter = UnifiedSignalArbiter()
        regime = self._make_regime("trending")

        result = arbiter.arbitrate(
            rl_signal="buy",
            rl_confidence=0.8,
            specialist_signal="buy",
            specialist_confidence=0.7,
            regime=regime,
        )
        assert result.final_signal == "buy"
        assert result.final_confidence > 0.5
        assert result.sentiment_signal == "hold"
        assert result.sentiment_confidence == 0.0

    def test_unanimous_3way_agreement(self):
        """All three signals agree → extra confidence boost."""
        from python.src.agents.signal_intelligence import UnifiedSignalArbiter

        arbiter = UnifiedSignalArbiter()
        regime = self._make_regime("trending")

        result = arbiter.arbitrate(
            rl_signal="buy",
            rl_confidence=0.8,
            specialist_signal="buy",
            specialist_confidence=0.7,
            sentiment_signal="buy",
            sentiment_confidence=0.6,
            regime=regime,
        )
        assert result.final_signal == "buy"
        assert result.source == "unanimous_agreement"
        assert result.sentiment_signal == "buy"

    def test_sentiment_dissent(self):
        """RL + specialist agree but sentiment disagrees → smaller boost."""
        from python.src.agents.signal_intelligence import UnifiedSignalArbiter

        arbiter = UnifiedSignalArbiter()
        regime = self._make_regime("trending")

        # Without sentiment dissent
        result_no_sent = arbiter.arbitrate(
            rl_signal="buy",
            rl_confidence=0.8,
            specialist_signal="buy",
            specialist_confidence=0.7,
            regime=regime,
        )

        # With sentiment dissent
        result_dissent = arbiter.arbitrate(
            rl_signal="buy",
            rl_confidence=0.8,
            specialist_signal="buy",
            specialist_confidence=0.7,
            sentiment_signal="sell",
            sentiment_confidence=0.6,
            regime=regime,
        )

        assert result_dissent.final_signal == "buy"
        assert result_dissent.source == "agreement_sentiment_dissent"
        # Dissent should give less confidence than no sentiment
        assert result_dissent.final_confidence <= result_no_sent.final_confidence

    def test_sentiment_confirms_hold_signal(self):
        """Sentiment confirms the active directional signal in one-hold scenario."""
        from python.src.agents.signal_intelligence import UnifiedSignalArbiter

        arbiter = UnifiedSignalArbiter()
        regime = self._make_regime("trending")

        result = arbiter.arbitrate(
            rl_signal="buy",
            rl_confidence=0.7,
            specialist_signal="hold",
            specialist_confidence=0.5,
            sentiment_signal="buy",
            sentiment_confidence=0.65,
            regime=regime,
        )
        # Sentiment confirms RL's buy → should proceed
        assert result.final_signal == "buy"

    def test_result_has_sentiment_fields(self):
        """ArbitrationResult always includes sentiment fields."""
        from python.src.agents.signal_intelligence import UnifiedSignalArbiter

        arbiter = UnifiedSignalArbiter()
        regime = self._make_regime()

        result = arbiter.arbitrate(
            rl_signal="hold",
            rl_confidence=0.5,
            specialist_signal="hold",
            specialist_confidence=0.5,
            regime=regime,
        )
        assert hasattr(result, "sentiment_signal")
        assert hasattr(result, "sentiment_confidence")

    def test_instability_hold_with_sentiment(self):
        """Instability check works even when sentiment is provided."""
        from python.src.agents.signal_intelligence import UnifiedSignalArbiter

        arbiter = UnifiedSignalArbiter()
        regime = self._make_regime("trending")

        momentum = {
            "stability": 0.2,  # Very unstable
            "changes_in_window": 8,
            "dominant_signal": "buy",
            "distribution": {"buy": 5, "sell": 3, "hold": 2},
        }

        result = arbiter.arbitrate(
            rl_signal="buy",
            rl_confidence=0.9,
            specialist_signal="buy",
            specialist_confidence=0.85,
            sentiment_signal="buy",
            sentiment_confidence=0.8,
            regime=regime,
            signal_momentum=momentum,
        )
        assert result.final_signal == "hold"
        assert result.source == "instability_hold"


# ---------------------------------------------------------------------------
# Tests: NewsSentimentFetcher — unit tests (mocking network calls)
# ---------------------------------------------------------------------------

class TestNewsSentimentFetcher:
    """Tests for the news sentiment fetcher (no network calls)."""

    @pytest.fixture(autouse=True)
    def _mock_yfinance(self, monkeypatch):
        """Mock yfinance to avoid ImportError in CI environments."""
        import sys
        import types

        if "yfinance" not in sys.modules:
            mock_yf = types.ModuleType("yfinance")
            mock_yf.Ticker = lambda *a, **kw: None
            monkeypatch.setitem(sys.modules, "yfinance", mock_yf)

    def test_empty_dataframe_schema(self):
        from python.src.data.sentiment_fetcher import NewsSentimentFetcher

        df = NewsSentimentFetcher._empty_dataframe()
        expected_cols = [
            "datetime", "headline", "source", "url",
            "raw_sentiment", "relevance_score",
        ]
        assert list(df.columns) == expected_cols
        assert len(df) == 0

    def test_normalize_symbol(self):
        from python.src.data.sentiment_fetcher import NewsSentimentFetcher

        assert NewsSentimentFetcher._normalize_symbol("EURUSD=X") == "EURUSD"
        assert NewsSentimentFetcher._normalize_symbol("AAPL") == "AAPL"

    def test_symbol_to_query(self):
        from python.src.data.sentiment_fetcher import NewsSentimentFetcher

        assert "forex" in NewsSentimentFetcher._symbol_to_query("EURUSD=X")
        assert NewsSentimentFetcher._symbol_to_query("AAPL") == "AAPL"

    def test_extract_ticker_sentiment_found(self):
        from python.src.data.sentiment_fetcher import NewsSentimentFetcher

        sentiments = [
            {"ticker": "AAPL", "ticker_sentiment_score": "0.35", "relevance_score": "0.9"},
            {"ticker": "GOOGL", "ticker_sentiment_score": "-0.2", "relevance_score": "0.5"},
        ]
        result = NewsSentimentFetcher._extract_ticker_sentiment(sentiments, "AAPL")
        assert result["score"] == 0.35
        assert result["relevance"] == 0.9

    def test_extract_ticker_sentiment_not_found(self):
        from python.src.data.sentiment_fetcher import NewsSentimentFetcher

        sentiments = [{"ticker": "GOOGL", "ticker_sentiment_score": "0.1", "relevance_score": "0.3"}]
        result = NewsSentimentFetcher._extract_ticker_sentiment(sentiments, "AAPL")
        assert result["score"] == 0.0

    def test_parse_av_datetime(self):
        from python.src.data.sentiment_fetcher import NewsSentimentFetcher

        result = NewsSentimentFetcher._parse_av_datetime("20240115T143022")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

