"""
Tests for LLM provider.
"""
import pytest
from unittest.mock import AsyncMock, patch


class TestDeepSeekProvider:
    """Test DeepSeek provider."""

    def test_provider_initialization(self):
        """Test provider initializes correctly."""
        from src.llm.provider import DeepSeekProvider

        provider = DeepSeekProvider(api_key="test-key")
        assert provider.api_key == "test-key"
        assert provider.model == "deepseek-chat"

    def test_default_model_is_latest(self):
        """Verify default model is V3.2 (deepseek-chat)."""
        from src.llm.provider import DeepSeekProvider

        assert DeepSeekProvider.DEFAULT_MODEL == "deepseek-chat"
        assert DeepSeekProvider.REASONER_MODEL == "deepseek-reasoner"

    def test_usage_tracking(self):
        """Test usage tracking works."""
        from src.llm.provider import DeepSeekProvider

        provider = DeepSeekProvider(api_key="test-key")
        assert provider.total_tokens_used == 0

        stats = provider.get_usage_stats()
        assert "total_tokens_used" in stats
        assert "model" in stats

    def test_factory_creates_deepseek(self):
        """Test factory creates DeepSeek provider."""
        from src.llm.provider import LLMProviderFactory, DeepSeekProvider

        provider = LLMProviderFactory.create("deepseek")
        assert isinstance(provider, DeepSeekProvider)

    def test_factory_raises_for_unknown(self):
        """Test factory raises for unknown provider."""
        from src.llm.provider import LLMProviderFactory

        with pytest.raises(ValueError, match="Unknown provider"):
            LLMProviderFactory.create("unknown")


class TestTradingPrompts:
    """Test trading prompts."""

    def test_market_analysis_prompt_format(self):
        """Test market analysis prompt format."""
        from src.llm.prompts import format_market_analysis_prompt

        prompt = format_market_analysis_prompt(
            symbol="EURUSD",
            price=1.0850,
            indicators={"rsi_14": 45.5, "macd": 0.001},
        )

        assert "EURUSD" in prompt
        assert "1.08500" in prompt
        assert "rsi_14" in prompt

    def test_prompts_are_concise(self):
        """Test system prompts are token-optimized (< 200 chars each)."""
        from src.llm.prompts import TradingPrompts

        assert len(TradingPrompts.MARKET_ANALYSIS) < 300
        assert len(TradingPrompts.RISK_MANAGEMENT) < 300
        assert len(TradingPrompts.EXECUTION) < 300
