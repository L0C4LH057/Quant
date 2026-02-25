"""
FastAPI application for the RL Trading Service.

Security:
    - API key authentication
    - Input validation (Pydantic)
    - CORS configured
    - Error handling
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from ..config.base import get_config
from ..utils.logger import setup_logger

# Routes
from .routes import agents, training, backtest, health, trading

logger = logging.getLogger(__name__)

# API Key security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Verify API key from request header.

    Security:
        - Raises 403 if key is missing or invalid
        - Key loaded from environment, never hardcoded
    """
    config = get_config()
    expected_key = config.api_key

    if not expected_key:
        # No key configured - development mode
        logger.warning("No API_KEY configured - running in development mode")
        return "dev"

    if not api_key:
        raise HTTPException(status_code=403, detail="API key required")

    if api_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    config = get_config()
    logger.info(f"Starting PipFlow AI service in {config.environment} mode")

    yield

    # Shutdown
    logger.info("Shutting down PipFlow AI service")


def create_app() -> FastAPI:
    """
    Create configured FastAPI application.

    Returns:
        FastAPI app instance
    """
    config = get_config()

    # Setup logging
    setup_logger(
        "pipflow",
        json_format=config.log_format == "json",
    )

    app = FastAPI(
        title="PipFlow AI - RL Trading Service",
        description="Multi-agent reinforcement learning trading system",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if config.debug else ["http://localhost:8080"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(
        agents.router,
        prefix="/api/agents",
        tags=["Agents"],
        dependencies=[Depends(verify_api_key)],
    )
    app.include_router(
        training.router,
        prefix="/api/training",
        tags=["Training"],
        dependencies=[Depends(verify_api_key)],
    )
    app.include_router(
        backtest.router,
        prefix="/api/backtest",
        tags=["Backtesting"],
        dependencies=[Depends(verify_api_key)],
    )
    # Trading routes (no auth for development dashboard)
    app.include_router(
        trading.router,
        prefix="/api/trading",
        tags=["Trading"],
    )

    # Static files for dashboard
    static_dir = Path(__file__).parent.parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/dashboard", include_in_schema=False)
        async def dashboard():
            """Serve trading dashboard."""
            return FileResponse(str(static_dir / "dashboard" / "index.html"))

    return app


# Create default app instance
app = create_app()
