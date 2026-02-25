"""
LLM Provider abstraction layer.

Supports:
    - DeepSeek (primary) - V3.2 latest model
    - OpenAI-compatible fallback

Token Optimization:
    - Structured outputs for parsing
    - Token counting for budget management
    - Response caching (optional)
"""
import logging
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from ..config.base import get_config

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM response wrapper."""

    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.usage.get("total_tokens", 0)


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        """Generate completion."""
        pass

    @abstractmethod
    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Dict[str, Any],
        max_tokens: int = 500,
    ) -> Dict[str, Any]:
        """Generate structured JSON output."""
        pass


class DeepSeekProvider(BaseLLMProvider):
    """
    DeepSeek LLM Provider.

    Uses DeepSeek-V3.2 (latest) - Reasoning-first model built for agents.
    OpenAI-compatible API format.

    Models available:
        - deepseek-chat: General chat (V3.2)
        - deepseek-reasoner: Extended reasoning (R1)

    Example:
        >>> provider = DeepSeekProvider()
        >>> response = await provider.generate(
        ...     system_prompt="You are a trading analyst.",
        ...     user_prompt="Analyze EURUSD trend."
        ... )
    """

    # Latest model (Dec 2025)
    DEFAULT_MODEL = "deepseek-chat"  # Maps to V3.2
    REASONER_MODEL = "deepseek-reasoner"  # For complex reasoning

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
    ):
        """
        Initialize DeepSeek provider.

        Args:
            api_key: DeepSeek API key (from env if not provided)
            base_url: API base URL
            model: Model to use (default: deepseek-chat/V3.2)
            timeout: Request timeout in seconds
        """
        config = get_config()

        self.api_key = api_key or config.deepseek_api_key
        self.base_url = base_url or config.deepseek_base_url
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout

        if not self.api_key:
            logger.warning("DeepSeek API key not configured")

        # Track usage
        self.total_tokens_used = 0

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate text completion.

        Args:
            system_prompt: System instruction
            user_prompt: User message
            max_tokens: Maximum response tokens
            temperature: Sampling temperature (0-1)

        Returns:
            Generated text content
        """
        response = await self._call_api(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return response.content

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[Dict[str, Any]] = None,
        max_tokens: int = 500,
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output.

        Args:
            system_prompt: System instruction (include JSON format)
            user_prompt: User message
            response_format: JSON schema for response
            max_tokens: Maximum response tokens

        Returns:
            Parsed JSON response
        """
        # Enhance system prompt for JSON output
        json_prompt = f"{system_prompt}\n\nRespond with valid JSON only. No markdown."

        response = await self._call_api(
            messages=[
                {"role": "system", "content": json_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,  # Lower temp for structured output
            response_format=response_format,
        )

        # Parse JSON response
        try:
            content = response.content.strip()
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return {"error": "Invalid JSON response", "raw": response.content}

    async def generate_with_thinking(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1000,
    ) -> str:
        """
        Generate with extended thinking (uses reasoner model).

        Best for complex analysis and multi-step reasoning.

        Args:
            system_prompt: System instruction
            user_prompt: User message
            max_tokens: Maximum response tokens

        Returns:
            Generated text with reasoning
        """
        original_model = self.model
        self.model = self.REASONER_MODEL

        try:
            response = await self._call_api(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.5,
            )
            return response.content
        finally:
            self.model = original_model

    async def _call_api(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.7,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """Make API call to DeepSeek."""
        if not self.api_key:
            raise ValueError("DeepSeek API key not configured")

        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Add response format if specified (for JSON mode)
        if response_format:
            payload["response_format"] = {"type": "json_object"}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

            # Extract response
            choice = data["choices"][0]
            usage = data.get("usage", {})

            # Track usage
            self.total_tokens_used += usage.get("total_tokens", 0)

            logger.debug(
                f"DeepSeek API call: model={self.model}, "
                f"tokens={usage.get('total_tokens', 0)}"
            )

            return LLMResponse(
                content=choice["message"]["content"],
                model=data.get("model", self.model),
                usage=usage,
                finish_reason=choice.get("finish_reason", "stop"),
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"DeepSeek API error: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            raise

    def get_usage_stats(self) -> Dict[str, int]:
        """Get token usage statistics."""
        return {
            "total_tokens_used": self.total_tokens_used,
            "model": self.model,
        }


class LLMProviderFactory:
    """Factory for creating LLM providers."""

    @staticmethod
    def create(provider: str = "deepseek") -> BaseLLMProvider:
        """
        Create LLM provider instance.

        Args:
            provider: Provider name (deepseek)

        Returns:
            LLM provider instance
        """
        if provider == "deepseek":
            return DeepSeekProvider()
        else:
            raise ValueError(f"Unknown provider: {provider}")


# Default provider instance
def get_llm_provider(provider: str = "deepseek") -> BaseLLMProvider:
    """Get LLM provider instance."""
    return LLMProviderFactory.create(provider)
