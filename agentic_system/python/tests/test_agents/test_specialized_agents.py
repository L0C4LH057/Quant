"""
Tests for specialized agents.
"""
import pytest


@pytest.mark.asyncio
class TestMarketAnalysisAgent:
    """Test market analysis agent."""

    async def test_process_generates_signal(self):
        """Test that process generates a valid signal."""
        from src.agents.specialized import MarketAnalysisAgent

        agent = MarketAnalysisAgent()

        result = await agent.process({
            "symbol": "EURUSD",
            "current_price": 1.0850,
            "indicators": {
                "rsi_14": 35,  # Near oversold
                "macd": 0.001,
                "macd_signal": -0.001,
                "sma_20": 1.08,
                "sma_50": 1.075,
                "close": 1.085,
            },
        })

        assert "signal" in result
        assert result["signal"] in ["buy", "sell", "hold"]
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1


@pytest.mark.asyncio
class TestRiskManagementAgent:
    """Test risk management agent."""

    async def test_calculates_position_size(self):
        """Test position size calculation."""
        from src.agents.specialized import RiskManagementAgent

        agent = RiskManagementAgent()

        result = await agent.process({
            "signal": "buy",
            "confidence": 0.8,
            "current_price": 1.0850,
            "account_balance": 100000,
            "atr": 0.0050,
        })

        assert result["approved"] is True
        assert result["position_size"] > 0
        assert result["stop_loss"] > 0

    async def test_hold_signal_not_approved(self):
        """Test that hold signal returns not approved."""
        from src.agents.specialized import RiskManagementAgent

        agent = RiskManagementAgent()

        result = await agent.process({
            "signal": "hold",
            "confidence": 0.5,
            "current_price": 1.0850,
            "account_balance": 100000,
        })

        assert result["approved"] is False
        assert result["position_size"] == 0
