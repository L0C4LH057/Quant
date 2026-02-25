#!/usr/bin/env python3
"""
Test script to validate DeepSeek API integration.

Run with: python scripts/test_deepseek.py
"""
import asyncio
import sys
import os

# Add project root to path BEFORE importing
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Also need to load .env from correct location
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(project_root), '.env')
load_dotenv(env_path)


async def test_basic_generation():
    """Test basic text generation."""
    from src.llm.provider import DeepSeekProvider

    print("=" * 50)
    print("Testing DeepSeek API Integration")
    print("=" * 50)

    provider = DeepSeekProvider()

    print(f"\n✓ Provider initialized")
    print(f"  Model: {provider.model}")
    print(f"  Base URL: {provider.base_url}")
    print(f"  API Key configured: {'Yes' if provider.api_key else 'No'}")

    if not provider.api_key:
        print("\n✗ Error: DEEPSEEK_API_KEY not set in .env")
        return False

    print("\n--- Test 1: Basic Generation ---")
    try:
        response = await provider.generate(
            system_prompt="You are a helpful assistant. Be very brief.",
            user_prompt="What is 2+2? Reply in one word.",
            max_tokens=10,
        )
        print(f"✓ Response: {response}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n--- Test 2: Structured JSON Output ---")
    try:
        response = await provider.generate_structured(
            system_prompt="You are a trading analyst. Output valid JSON only.",
            user_prompt="Analyze EURUSD with RSI=35. Output: {\"signal\": \"buy/sell/hold\", \"confidence\": 0.0-1.0}",
            max_tokens=50,
        )
        print(f"✓ Response: {response}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n--- Test 3: Market Analysis Agent ---")
    try:
        from src.agents.specialized import MarketAnalysisAgent
        from src.llm.provider import get_llm_provider

        llm = get_llm_provider("deepseek")
        agent = MarketAnalysisAgent(llm_provider=llm)

        result = await agent.process({
            "symbol": "EURUSD",
            "current_price": 1.0850,
            "indicators": {
                "rsi_14": 32,
                "macd": 0.0015,
                "macd_signal": -0.001,
                "sma_20": 1.082,
                "sma_50": 1.078,
                "close": 1.085,
            },
            "use_llm": True,
        })

        print(f"✓ Signal: {result['signal']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Reason: {result['reason']}")
        print(f"  Analysis Type: {result.get('analysis_type', 'unknown')}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n--- Token Usage ---")
    stats = provider.get_usage_stats()
    print(f"Total tokens used: {stats['total_tokens_used']}")

    print("\n" + "=" * 50)
    print("✓ All tests passed!")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = asyncio.run(test_basic_generation())
    sys.exit(0 if success else 1)
