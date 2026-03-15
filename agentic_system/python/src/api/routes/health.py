"""
Health check endpoints (GAP-07 fix: dependency verification).
"""
import logging
from typing import Dict, Any

from fastapi import APIRouter

from ...config.base import get_config

router = APIRouter()
logger = logging.getLogger(__name__)


async def _check_redis() -> Dict[str, Any]:
    """Verify Redis connectivity."""
    try:
        import redis

        config = get_config()
        r = redis.from_url(config.redis_url, socket_timeout=2)
        r.ping()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


async def _check_models() -> Dict[str, Any]:
    """Check if at least one trained model exists."""
    try:
        config = get_config()
        model_dir = config.model_save_path
        if model_dir.exists():
            zip_files = list(model_dir.rglob("*.zip"))
            return {"status": "ok", "model_count": len(zip_files)}
        return {"status": "ok", "model_count": 0}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


async def _check_llm() -> Dict[str, Any]:
    """Verify at least one LLM API key is configured."""
    config = get_config()
    if config.deepseek_api_key or config.anthropic_api_key:
        return {"status": "ok"}
    return {"status": "degraded", "error": "No LLM API key configured"}


@router.get("/health")
async def health_check():
    """
    Health check endpoint with dependency verification.

    Returns:
        Overall status + per-dependency status
    """
    redis_check = await _check_redis()
    model_check = await _check_models()
    llm_check = await _check_llm()

    checks = {
        "redis": redis_check,
        "models": model_check,
        "llm": llm_check,
    }

    overall = "healthy"
    for name, result in checks.items():
        if result.get("status") != "ok":
            overall = "degraded"
            break

    return {"status": overall, "service": "pipflow-ai", "checks": checks}


@router.get("/ready")
async def readiness_check():
    """
    Readiness check — returns ready only when critical deps are reachable.
    """
    llm_check = await _check_llm()
    ready = llm_check.get("status") == "ok"
    return {"ready": ready}
