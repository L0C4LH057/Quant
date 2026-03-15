"""
LLM Provider abstraction layer.

Supported providers (all pre-configured — only an API key needed to switch):

    ┌─────────────┬──────────────────────────────────┬────────────────────────────┐
    │ Provider    │ Env key                          │ Default model              │
    ├─────────────┼──────────────────────────────────┼────────────────────────────┤
    │ deepseek    │ DEEPSEEK_API_KEY                 │ deepseek-chat              │
    │ openai      │ OPENAI_API_KEY                   │ gpt-4o-mini                │
    │ anthropic   │ ANTHROPIC_API_KEY                │ claude-sonnet-4-20250514   │
    │ gemini      │ GOOGLE_API_KEY  (or GEMINI_*)    │ gemini-2.0-flash           │
    │ kimi        │ KIMI_API_KEY    (Moonshot AI)     │ moonshot-v1-8k             │
    │ grok        │ XAI_API_KEY     (xAI)            │ grok-2-latest              │
    └─────────────┴──────────────────────────────────┴────────────────────────────┘

Switching providers requires only:
    1. Set the provider's API key env var.
    2. Set ``LLM_PROVIDER=<name>`` (or pass ``provider=`` to factory/get_llm_provider).

Token Optimization:
    - Structured outputs for parsing
    - Token counting for budget management
    - Shared OpenAI-compatible base reduces code duplication
"""
from __future__ import annotations

import json
import logging
import os
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

    #: Human-readable provider name — set by each subclass
    PROVIDER_NAME: str = "base"

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
        response_format: Optional[Dict[str, Any]] = None,
        max_tokens: int = 500,
    ) -> Dict[str, Any]:
        """Generate structured JSON output."""
        pass

    def get_usage_stats(self) -> Dict[str, Any]:
        """Return token usage statistics."""
        return {"total_tokens_used": getattr(self, "total_tokens_used", 0), "model": getattr(self, "model", "unknown")}


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI-Compatible base
#
# DeepSeek, OpenAI, Kimi (Moonshot) and Grok (xAI) all speak the same
# /v1/chat/completions protocol.  This shared base eliminates duplication.
# ─────────────────────────────────────────────────────────────────────────────

class OpenAICompatibleProvider(BaseLLMProvider):
    """
    Shared base for any provider that speaks the OpenAI chat / completions API.

    Subclasses only need to set ``BASE_URL``, ``DEFAULT_MODEL``, ``PROVIDER_NAME``,
    and the env-var name for their key.  All HTTP logic lives here.
    """

    BASE_URL: str = "https://api.openai.com/v1"
    DEFAULT_MODEL: str = "gpt-4o-mini"
    PROVIDER_NAME: str = "openai_compatible"
    _API_KEY_ENV: str = "OPENAI_API_KEY"    # override in subclass
    _MODEL_ENV: str = "OPENAI_MODEL"         # override in subclass

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key or os.getenv(self._API_KEY_ENV, "")
        self.model = model or os.getenv(self._MODEL_ENV, self.DEFAULT_MODEL)
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.timeout = timeout
        self.total_tokens_used = 0

        if not self.api_key:
            logger.warning("%s API key not configured (env: %s)", self.PROVIDER_NAME, self._API_KEY_ENV)

    # ── Public interface ──────────────────────────────────────────────────────

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
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
        json_prompt = f"{system_prompt}\n\nRespond with valid JSON only. No markdown."
        response = await self._call_api(
            messages=[
                {"role": "system", "content": json_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
            json_mode=True,
        )
        return _parse_json(response.content)

    # ── Internal HTTP call ────────────────────────────────────────────────────

    async def _call_api(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> "LLMResponse":
        if not self.api_key:
            raise ValueError(f"{self.PROVIDER_NAME} API key not configured (env: {self._API_KEY_ENV})")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()

            choice = data["choices"][0]
            usage = data.get("usage", {})
            self.total_tokens_used += usage.get("total_tokens", 0)
            logger.debug("%s: model=%s tokens=%s", self.PROVIDER_NAME, self.model, usage.get("total_tokens", 0))

            return LLMResponse(
                content=choice["message"]["content"],
                model=data.get("model", self.model),
                usage=usage,
                finish_reason=choice.get("finish_reason", "stop"),
            )
        except httpx.HTTPStatusError as e:
            logger.error("%s API error %s: %s", self.PROVIDER_NAME, e.response.status_code, e.response.text[:200])
            raise
        except Exception as e:
            logger.error("%s API error: %s", self.PROVIDER_NAME, e)
            raise


# ─────────────────────────────────────────────────────────────────────────────
# DeepSeek
# ─────────────────────────────────────────────────────────────────────────────

class DeepSeekProvider(OpenAICompatibleProvider):
    """
    DeepSeek LLM Provider (deepseek-chat / deepseek-reasoner).

    OpenAI-compatible wire format.  Set ``DEEPSEEK_API_KEY`` and optionally
    ``DEEPSEEK_MODEL`` to switch model (default: ``deepseek-chat`` / V3.2).

    Extra: ``generate_with_thinking()`` switches to the R1 reasoner mid-call.

    Example::

        provider = DeepSeekProvider()
        text = await provider.generate(system_prompt=..., user_prompt=...)
    """

    BASE_URL = "https://api.deepseek.com/v1"
    DEFAULT_MODEL = "deepseek-chat"
    REASONER_MODEL = "deepseek-reasoner"
    PROVIDER_NAME = "deepseek"
    _API_KEY_ENV = "DEEPSEEK_API_KEY"
    _MODEL_ENV = "DEEPSEEK_MODEL"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        config = get_config()
        super().__init__(
            api_key=api_key or config.deepseek_api_key,
            model=model,
            base_url=base_url or config.deepseek_base_url,
            timeout=timeout,
        )

    async def generate_with_thinking(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1000,
    ) -> str:
        """Use the DeepSeek-R1 reasoner for complex multi-step analysis."""
        orig = self.model
        self.model = self.REASONER_MODEL
        try:
            return await self.generate(system_prompt, user_prompt, max_tokens=max_tokens, temperature=0.5)
        finally:
            self.model = orig


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI
# ─────────────────────────────────────────────────────────────────────────────

class OpenAIProvider(OpenAICompatibleProvider):
    """
    OpenAI LLM Provider.

    Set ``OPENAI_API_KEY``.  Optionally ``OPENAI_MODEL`` (default: ``gpt-4o-mini``).

    Common models:
        * ``gpt-4o``        — best quality
        * ``gpt-4o-mini``   — fast & cheap  ← default
        * ``gpt-4-turbo``   — large context
        * ``o1``            — reasoning
        * ``o3-mini``       — fast reasoning

    Example::

        provider = OpenAIProvider()                     # gpt-4o-mini
        provider = OpenAIProvider(model="gpt-4o")       # override
    """

    BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODEL = "gpt-4o-mini"
    PROVIDER_NAME = "openai"
    _API_KEY_ENV = "OPENAI_API_KEY"
    _MODEL_ENV = "OPENAI_MODEL"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, timeout: float = 60.0) -> None:
        config = get_config()
        super().__init__(api_key=api_key or config.openai_api_key, model=model, timeout=timeout)


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic (Claude)
# ─────────────────────────────────────────────────────────────────────────────

class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude LLM Provider.

    Uses the native Anthropic Messages API (not OpenAI-compatible).
    Set ``ANTHROPIC_API_KEY``.  Optionally ``ANTHROPIC_MODEL``.

    Common models:
        * ``claude-opus-4-20250514``   — most capable
        * ``claude-sonnet-4-20250514`` — balanced  ← default
        * ``claude-haiku-3-5``         — fast & cheap

    Example::

        provider = AnthropicProvider()
        text = await provider.generate(system_prompt=..., user_prompt=...)
    """

    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    PROVIDER_NAME = "anthropic"
    _API_KEY_ENV = "ANTHROPIC_API_KEY"
    _MODEL_ENV = "ANTHROPIC_MODEL"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        config = get_config()
        self.api_key = api_key or config.anthropic_api_key or os.getenv(self._API_KEY_ENV, "")
        self.model = model or os.getenv(self._MODEL_ENV, self.DEFAULT_MODEL)
        self.timeout = timeout
        self.total_tokens_used = 0

        if not self.api_key:
            logger.warning("Anthropic API key not configured (env: ANTHROPIC_API_KEY)")

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        response = await self._call_api(system_prompt, user_prompt, max_tokens, temperature)
        return response.content

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[Dict[str, Any]] = None,
        max_tokens: int = 500,
    ) -> Dict[str, Any]:
        json_prompt = f"{system_prompt}\n\nRespond with valid JSON only. No markdown."
        response = await self._call_api(json_prompt, user_prompt, max_tokens, temperature=0.3)
        return _parse_json(response.content)

    async def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> "LLMResponse":
        if not self.api_key:
            raise ValueError("Anthropic API key not configured")

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()

            content = data["content"][0]["text"]
            usage = data.get("usage", {})
            total = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            self.total_tokens_used += total
            logger.debug("anthropic: model=%s tokens=%s", self.model, total)

            return LLMResponse(
                content=content,
                model=data.get("model", self.model),
                usage={"prompt_tokens": usage.get("input_tokens", 0), "completion_tokens": usage.get("output_tokens", 0), "total_tokens": total},
                finish_reason=data.get("stop_reason", "end_turn"),
            )
        except httpx.HTTPStatusError as e:
            logger.error("Anthropic API error %s: %s", e.response.status_code, e.response.text[:200])
            raise
        except Exception as e:
            logger.error("Anthropic API error: %s", e)
            raise


# ─────────────────────────────────────────────────────────────────────────────
# Google Gemini
# ─────────────────────────────────────────────────────────────────────────────

class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini LLM Provider.

    Uses Google's ``generativelanguage.googleapis.com`` REST API directly
    (no SDK required).  Set ``GOOGLE_API_KEY``.  Optionally ``GEMINI_MODEL``.

    Common models:
        * ``gemini-2.0-flash``         — fast & cheap  ← default
        * ``gemini-2.5-pro-preview``   — most capable
        * ``gemini-1.5-flash``         — legacy fast

    Example::

        provider = GeminiProvider()
        text = await provider.generate(system_prompt=..., user_prompt=...)
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    DEFAULT_MODEL = "gemini-2.0-flash"
    PROVIDER_NAME = "gemini"
    _API_KEY_ENV = "GOOGLE_API_KEY"
    _MODEL_ENV = "GEMINI_MODEL"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        config = get_config()
        self.api_key = api_key or config.google_api_key or os.getenv(self._API_KEY_ENV, "")
        self.model = model or os.getenv(self._MODEL_ENV, self.DEFAULT_MODEL)
        self.base_url = self.BASE_URL
        self.timeout = timeout
        self.total_tokens_used = 0

        if not self.api_key:
            logger.warning("Google API key not configured (env: GOOGLE_API_KEY)")

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        response = await self._call_api(system_prompt, user_prompt, max_tokens, temperature)
        return response.content

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[Dict[str, Any]] = None,
        max_tokens: int = 500,
    ) -> Dict[str, Any]:
        json_prompt = f"{system_prompt}\n\nRespond with valid JSON only. No markdown."
        response = await self._call_api(json_prompt, user_prompt, max_tokens, temperature=0.3)
        return _parse_json(response.content)

    async def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> "LLMResponse":
        if not self.api_key:
            raise ValueError("Google API key not configured (env: GOOGLE_API_KEY)")

        url = f"{self.base_url}/models/{self.model}:generateContent"
        params = {"key": self.api_key}

        payload: Dict[str, Any] = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, params=params, json=payload)
                resp.raise_for_status()
                data = resp.json()

            candidate = data["candidates"][0]
            content = candidate["content"]["parts"][0]["text"]
            usage = data.get("usageMetadata", {})
            total = usage.get("totalTokenCount", 0)
            self.total_tokens_used += total
            logger.debug("gemini: model=%s tokens=%s", self.model, total)

            return LLMResponse(
                content=content,
                model=self.model,
                usage={
                    "prompt_tokens": usage.get("promptTokenCount", 0),
                    "completion_tokens": usage.get("candidatesTokenCount", 0),
                    "total_tokens": total,
                },
                finish_reason=candidate.get("finishReason", "STOP"),
            )
        except httpx.HTTPStatusError as e:
            logger.error("Gemini API error %s: %s", e.response.status_code, e.response.text[:200])
            raise
        except Exception as e:
            logger.error("Gemini API error: %s", e)
            raise


# ─────────────────────────────────────────────────────────────────────────────
# Kimi (Moonshot AI)
# ─────────────────────────────────────────────────────────────────────────────

class KimiProvider(OpenAICompatibleProvider):
    """
    Kimi LLM Provider (Moonshot AI).

    OpenAI-compatible wire format.  Set ``KIMI_API_KEY``.
    Optionally ``KIMI_MODEL`` (default: ``moonshot-v1-8k``).

    Common models:
        * ``moonshot-v1-8k``   — standard  ← default
        * ``moonshot-v1-32k``  — long context
        * ``moonshot-v1-128k`` — very long context

    Example::

        provider = KimiProvider()
        text = await provider.generate(system_prompt=..., user_prompt=...)
    """

    BASE_URL = "https://api.moonshot.cn/v1"
    DEFAULT_MODEL = "moonshot-v1-8k"
    PROVIDER_NAME = "kimi"
    _API_KEY_ENV = "KIMI_API_KEY"
    _MODEL_ENV = "KIMI_MODEL"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, timeout: float = 60.0) -> None:
        config = get_config()
        super().__init__(api_key=api_key or config.kimi_api_key, model=model, timeout=timeout)


# ─────────────────────────────────────────────────────────────────────────────
# Grok (xAI)
# ─────────────────────────────────────────────────────────────────────────────

class GrokProvider(OpenAICompatibleProvider):
    """
    Grok LLM Provider (xAI).

    OpenAI-compatible wire format.  Set ``XAI_API_KEY``.
    Optionally ``GROK_MODEL`` (default: ``grok-2-latest``).

    Common models:
        * ``grok-2-latest``   ← default
        * ``grok-2``
        * ``grok-3-mini``

    Example::

        provider = GrokProvider()
        text = await provider.generate(system_prompt=..., user_prompt=...)
    """

    BASE_URL = "https://api.x.ai/v1"
    DEFAULT_MODEL = "grok-2-latest"
    PROVIDER_NAME = "grok"
    _API_KEY_ENV = "XAI_API_KEY"
    _MODEL_ENV = "GROK_MODEL"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, timeout: float = 60.0) -> None:
        config = get_config()
        super().__init__(api_key=api_key or config.xai_api_key, model=model, timeout=timeout)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json(content: str) -> Dict[str, Any]:
    """Strip Markdown fences and parse JSON from an LLM response."""
    text = content.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON response: %s", e)
        return {"error": "Invalid JSON response", "raw": content}


# ─────────────────────────────────────────────────────────────────────────────
# Factory & convenience function
# ─────────────────────────────────────────────────────────────────────────────

#: Registry mapping provider name → class.  Add new providers here.
_PROVIDER_REGISTRY: Dict[str, type] = {
    "deepseek":  DeepSeekProvider,
    "openai":    OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini":    GeminiProvider,
    "kimi":      KimiProvider,
    "grok":      GrokProvider,
}


class LLMProviderFactory:
    """
    Factory for creating LLM provider instances.

    Provider selection order:
        1. ``provider`` argument passed to ``create()``
        2. ``LLM_PROVIDER`` environment variable
        3. Falls back to ``"deepseek"``
    """

    @staticmethod
    def create(provider: Optional[str] = None, **kwargs: Any) -> BaseLLMProvider:
        """
        Create an LLM provider instance.

        Args:
            provider: Provider name.  If ``None``, reads ``LLM_PROVIDER`` env var.
                      Available: ``deepseek``, ``openai``, ``anthropic``,
                      ``gemini``, ``kimi``, ``grok``.
            **kwargs: Forwarded to the provider constructor
                      (e.g. ``model="gpt-4o"``, ``api_key=...``).

        Returns:
            Configured ``BaseLLMProvider`` instance.

        Raises:
            ValueError: Unknown provider name.
        """
        name = (provider or os.getenv("LLM_PROVIDER", "deepseek")).lower().strip()
        cls = _PROVIDER_REGISTRY.get(name)
        if cls is None:
            available = ", ".join(sorted(_PROVIDER_REGISTRY))
            raise ValueError(f"Unknown LLM provider '{name}'. Available: {available}")
        logger.info("Creating LLM provider: %s", name)
        return cls(**kwargs)

    @staticmethod
    def available() -> List[str]:
        """Return list of registered provider names."""
        return sorted(_PROVIDER_REGISTRY)


def get_llm_provider(provider: Optional[str] = None, **kwargs: Any) -> BaseLLMProvider:
    """
    Convenience wrapper around ``LLMProviderFactory.create()``.

    Reads ``LLM_PROVIDER`` env var when *provider* is ``None``.

    Example::

        # Use whatever LLM_PROVIDER env var says
        llm = get_llm_provider()

        # Explicit override
        llm = get_llm_provider("openai", model="gpt-4o")

        # Script-level — pick from env
        LLM_PROVIDER=gemini GOOGLE_API_KEY=... python my_script.py
    """
    return LLMProviderFactory.create(provider, **kwargs)

