"""
Sentiment analysis and feature engineering for news data.

Converts raw news articles into numerical sentiment features that can be
used by the trading agents, analogous to technical_indicators.py for OHLCV.

Supports:
    - VADER (rule-based, no API calls, fast — default)
    - FinBERT (transformer-based, optional — requires transformers + torch)

Token Optimization:
    - Each function is self-contained
    - Clear feature names for prompt readability
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded sentiment backends
# ---------------------------------------------------------------------------

_vader_analyzer = None
_finbert_pipeline = None


def _get_vader():
    """Lazy-load VADER sentiment analyzer."""
    global _vader_analyzer
    if _vader_analyzer is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            _vader_analyzer = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyzer loaded")
        except ImportError:
            raise ImportError(
                "vaderSentiment is required for VADER analysis. "
                "Install it with: pip install vaderSentiment"
            )
    return _vader_analyzer


def _get_finbert():
    """Lazy-load FinBERT transformer pipeline."""
    global _finbert_pipeline
    if _finbert_pipeline is None:
        try:
            from transformers import pipeline

            _finbert_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT sentiment pipeline loaded")
        except ImportError:
            raise ImportError(
                "transformers and torch are required for FinBERT. "
                "Install with: pip install transformers torch"
            )
    return _finbert_pipeline


# ---------------------------------------------------------------------------
# Core scoring functions
# ---------------------------------------------------------------------------


def score_headlines_vader(headlines: List[str]) -> List[Dict[str, float]]:
    """
    Score a list of headlines using VADER.

    Args:
        headlines: List of news headline strings.

    Returns:
        List of dicts with keys: compound, pos, neg, neu
        Compound ranges from -1.0 (most negative) to +1.0 (most positive).
    """
    analyzer = _get_vader()
    results = []
    for headline in headlines:
        if not headline or not isinstance(headline, str):
            results.append({"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0})
            continue
        scores = analyzer.polarity_scores(headline)
        results.append(scores)
    return results


def score_headlines_finbert(headlines: List[str]) -> List[Dict[str, float]]:
    """
    Score a list of headlines using FinBERT.

    Args:
        headlines: List of news headline strings.

    Returns:
        List of dicts with keys: compound (mapped to -1/0/+1 scale),
        label ("positive"/"negative"/"neutral"), raw_score.
    """
    pipe = _get_finbert()
    results = []
    for headline in headlines:
        if not headline or not isinstance(headline, str):
            results.append({"compound": 0.0, "label": "neutral", "raw_score": 0.0})
            continue
        try:
            out = pipe(headline[:512])[0]
            label = out["label"].lower()
            score = out["score"]
            # Map to compound: positive → +score, negative → -score, neutral → 0
            if label == "positive":
                compound = score
            elif label == "negative":
                compound = -score
            else:
                compound = 0.0
            results.append({
                "compound": round(compound, 4),
                "label": label,
                "raw_score": round(score, 4),
            })
        except Exception as e:
            logger.warning(f"FinBERT error on headline: {e}")
            results.append({"compound": 0.0, "label": "neutral", "raw_score": 0.0})
    return results


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def compute_sentiment_features(
    news_df: pd.DataFrame,
    model: str = "auto",
) -> pd.DataFrame:
    """
    Compute sentiment scores for a news DataFrame.

    Adds a 'sentiment_score' column (compound score) to the DataFrame.

    UPGRADE-05: ``model="auto"`` defaults to FinBERT when torch is available,
    otherwise falls back to VADER.

    Args:
        news_df: DataFrame with at least 'headline' and 'datetime' columns.
        model: "vader", "finbert", or "auto" (default).

    Returns:
        news_df with 'sentiment_score' column added.
    """
    if news_df.empty:
        news_df["sentiment_score"] = pd.Series(dtype=float)
        return news_df

    if model == "auto":
        try:
            import torch  # noqa: F401
            model = "finbert"
        except ImportError:
            model = "vader"

    headlines = news_df["headline"].fillna("").tolist()

    if model == "finbert":
        scores = score_headlines_finbert(headlines)
    else:
        scores = score_headlines_vader(headlines)

    news_df = news_df.copy()
    news_df["sentiment_score"] = [s["compound"] for s in scores]
    return news_df


def add_sentiment_features(
    market_df: pd.DataFrame,
    news_df: pd.DataFrame,
    lookback_windows: Optional[List[int]] = None,
    model: str = "vader",
) -> pd.DataFrame:
    """
    Merge sentiment features into a market OHLCV DataFrame.

    For each row in market_df, aggregates news sentiment over lookback windows
    and adds the following features:

        - sentiment_score:       Mean sentiment in the trailing window
        - sentiment_magnitude:   Mean absolute sentiment (strength)
        - sentiment_volume:      Number of articles in the window
        - sentiment_momentum:    Rate of change of sentiment
        - sentiment_divergence:  Divergence between sentiment and price direction

    Args:
        market_df: OHLCV DataFrame with 'date' or 'datetime' column and 'close'.
        news_df: News DataFrame with at least 'headline' and 'datetime'.
        lookback_windows: List of trailing windows in days (default: [1, 3, 7]).
        model: Sentiment model to use ("vader" or "finbert").

    Returns:
        market_df enriched with sentiment feature columns.

    Example:
        >>> enriched = add_sentiment_features(ohlcv_df, news_df)
        >>> print([c for c in enriched.columns if 'sentiment' in c])
        ['sentiment_score', 'sentiment_magnitude', 'sentiment_volume',
         'sentiment_momentum', 'sentiment_divergence']
    """
    if lookback_windows is None:
        lookback_windows = [1, 3, 7]

    market_df = market_df.copy()

    # Ensure news has sentiment scores
    if "sentiment_score" not in news_df.columns:
        news_df = compute_sentiment_features(news_df, model=model)

    # Normalise datetime columns
    date_col = "date" if "date" in market_df.columns else "datetime"
    market_df[date_col] = pd.to_datetime(market_df[date_col], errors="coerce")
    news_df["datetime"] = pd.to_datetime(news_df["datetime"], errors="coerce")
    news_df = news_df.dropna(subset=["datetime"])

    if news_df.empty:
        # No news — fill with neutral defaults
        market_df["sentiment_score"] = 0.0
        market_df["sentiment_magnitude"] = 0.0
        market_df["sentiment_volume"] = 0
        market_df["sentiment_momentum"] = 0.0
        market_df["sentiment_divergence"] = 0.0
        return market_df

    # Use the primary lookback window for the main features
    primary_window = lookback_windows[0]

    sent_scores = []
    sent_magnitudes = []
    sent_volumes = []

    for _, row in market_df.iterrows():
        row_date = row[date_col]
        if pd.isna(row_date):
            sent_scores.append(0.0)
            sent_magnitudes.append(0.0)
            sent_volumes.append(0)
            continue

        # Articles in the lookback window
        window_start = row_date - pd.Timedelta(days=primary_window)
        mask = (news_df["datetime"] >= window_start) & (
            news_df["datetime"] <= row_date
        )
        window_articles = news_df.loc[mask]

        if window_articles.empty:
            sent_scores.append(0.0)
            sent_magnitudes.append(0.0)
            sent_volumes.append(0)
        else:
            scores = window_articles["sentiment_score"]
            sent_scores.append(round(float(scores.mean()), 4))
            sent_magnitudes.append(round(float(scores.abs().mean()), 4))
            sent_volumes.append(len(window_articles))

    market_df["sentiment_score"] = sent_scores
    market_df["sentiment_magnitude"] = sent_magnitudes
    market_df["sentiment_volume"] = sent_volumes

    # Sentiment momentum: rate of change over the last few rows
    market_df["sentiment_momentum"] = (
        market_df["sentiment_score"]
        .diff()
        .fillna(0.0)
        .round(4)
    )

    # Sentiment divergence: sentiment direction vs. price direction
    if "close" in market_df.columns:
        price_change = market_df["close"].pct_change().fillna(0.0)
        sentiment_dir = np.sign(market_df["sentiment_score"])
        price_dir = np.sign(price_change)
        # Divergence = 1 when they disagree, 0 when they agree, 0.5 if either is neutral
        market_df["sentiment_divergence"] = np.where(
            (sentiment_dir == 0) | (price_dir == 0),
            0.0,
            np.where(sentiment_dir != price_dir, 1.0, 0.0),
        )
    else:
        market_df["sentiment_divergence"] = 0.0

    logger.info(
        f"Added sentiment features — {sum(v > 0 for v in sent_volumes)}/{len(market_df)} "
        f"rows have news data (window={primary_window}d)"
    )

    return market_df


def get_sentiment_summary(news_df: pd.DataFrame) -> Dict[str, float]:
    """
    Produce a concise sentiment summary for agent consumption.

    Args:
        news_df: News DataFrame with 'sentiment_score' column.

    Returns:
        Dict with keys: mean_sentiment, magnitude, article_count,
        bullish_pct, bearish_pct, neutral_pct
    """
    if news_df.empty or "sentiment_score" not in news_df.columns:
        return {
            "mean_sentiment": 0.0,
            "magnitude": 0.0,
            "article_count": 0,
            "bullish_pct": 0.0,
            "bearish_pct": 0.0,
            "neutral_pct": 1.0,
        }

    scores = news_df["sentiment_score"]
    total = len(scores)
    bullish = (scores > 0.05).sum()
    bearish = (scores < -0.05).sum()
    neutral = total - bullish - bearish

    return {
        "mean_sentiment": round(float(scores.mean()), 4),
        "magnitude": round(float(scores.abs().mean()), 4),
        "article_count": total,
        "bullish_pct": round(bullish / total, 4) if total > 0 else 0.0,
        "bearish_pct": round(bearish / total, 4) if total > 0 else 0.0,
        "neutral_pct": round(neutral / total, 4) if total > 0 else 1.0,
    }
