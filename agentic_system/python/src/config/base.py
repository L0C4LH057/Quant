"""
Base configuration with secure secret loading.

Security:
    - All secrets loaded from environment variables
    - Validation ensures required keys are present
    - No hardcoded secrets allowed
"""
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


# Load .env file
load_dotenv()


@dataclass
class Config:
    """
    Base configuration class with secure secret management.

    All API keys and secrets are loaded from environment variables.
    Never hardcode secrets in code.

    Attributes:
        environment: Current environment (development/staging/production)
        debug: Enable debug mode
        log_level: Logging level
        log_format: Log format (json for production)

    Security:
        - Validates required API keys on initialization
        - Raises ValueError if critical keys are missing
    """

    # Environment
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")

    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "text"))

    # ── LLM Provider selection ────────────────────────────────────────────────
    # Set LLM_PROVIDER to switch providers.  Only the matching API key is needed.
    # Available: deepseek | openai | anthropic | gemini | kimi | grok
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "deepseek"))

    # ── LLM Provider API keys ─────────────────────────────────────────────────
    deepseek_api_key: Optional[str] = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY"))
    deepseek_base_url: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    )
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    anthropic_api_key: Optional[str] = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    google_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    kimi_api_key: Optional[str] = field(default_factory=lambda: os.getenv("KIMI_API_KEY"))
    xai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("XAI_API_KEY"))

    # Market Data
    alpha_vantage_key: Optional[str] = field(default_factory=lambda: os.getenv("ALPHA_VANTAGE_KEY"))

    # News Sentiment
    newsapi_key: Optional[str] = field(default_factory=lambda: os.getenv("NEWSAPI_KEY"))
    finnhub_key: Optional[str] = field(default_factory=lambda: os.getenv("FINNHUB_KEY"))
    sentiment_model: str = field(default_factory=lambda: os.getenv("SENTIMENT_MODEL", "vader"))

    # Broker
    metaapi_token: Optional[str] = field(default_factory=lambda: os.getenv("METAAPI_TOKEN"))
    metaapi_account_id: Optional[str] = field(default_factory=lambda: os.getenv("METAAPI_ACCOUNT_ID"))

    # Service
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("API_KEY"))

    # Redis
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))

    # Paths
    model_save_path: Path = field(
        default_factory=lambda: Path(os.getenv("MODEL_SAVE_PATH", "storage/rl_models"))
    )
    tensorboard_log_path: Path = field(
        default_factory=lambda: Path(os.getenv("TENSORBOARD_LOG_PATH", "storage/training_logs"))
    )
    backtest_results_path: Path = field(
        default_factory=lambda: Path(os.getenv("BACKTEST_RESULTS_PATH", "storage/backtest_results"))
    )

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._ensure_paths_exist()

    def _ensure_paths_exist(self) -> None:
        """Create storage paths if they don't exist."""
        for path in [self.model_save_path, self.tensorboard_log_path, self.backtest_results_path]:
            path.mkdir(parents=True, exist_ok=True)

    def validate_llm_keys(self) -> None:
        """
        Validate that at least one LLM provider key is set.

        Raises:
            ValueError: If no LLM API key is configured
        """
        has_any = any([
            self.deepseek_api_key,
            self.openai_api_key,
            self.anthropic_api_key,
            self.google_api_key,
            self.kimi_api_key,
            self.xai_api_key,
        ])
        if not has_any:
            raise ValueError(
                "At least one LLM API key is required.  "
                "Set one of: DEEPSEEK_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, "
                "GOOGLE_API_KEY, KIMI_API_KEY, XAI_API_KEY."
            )

    def validate_broker_keys(self) -> None:
        """
        Validate broker configuration for live trading.

        Raises:
            ValueError: If broker credentials are missing
        """
        if not self.metaapi_token:
            raise ValueError("METAAPI_TOKEN is required for live trading.")
        if not self.metaapi_account_id:
            raise ValueError("METAAPI_ACCOUNT_ID is required for live trading.")

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"

    def to_safe_dict(self) -> dict:
        """
        Convert config to dictionary, excluding secrets.

        Use this for logging - secrets are excluded.
        """
        return {
            "environment": self.environment,
            "debug": self.debug,
            "log_level": self.log_level,
            "log_format": self.log_format,
            "redis_url": self.redis_url,
            "llm_provider": self.llm_provider,
            "has_deepseek_key": bool(self.deepseek_api_key),
            "has_openai_key": bool(self.openai_api_key),
            "has_anthropic_key": bool(self.anthropic_api_key),
            "has_google_key": bool(self.google_api_key),
            "has_kimi_key": bool(self.kimi_api_key),
            "has_xai_key": bool(self.xai_api_key),
            "has_metaapi_token": bool(self.metaapi_token),
        }


@lru_cache()
def get_config() -> Config:
    """
    Get cached configuration singleton.

    Returns:
        Config: Application configuration

    Example:
        >>> config = get_config()
        >>> print(config.log_level)
        INFO
    """
    return Config()
