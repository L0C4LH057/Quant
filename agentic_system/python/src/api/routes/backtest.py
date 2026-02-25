"""
Backtesting endpoints.
"""
import logging
from typing import Optional, List

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

router = APIRouter()
logger = logging.getLogger(__name__)


class BacktestConfig(BaseModel):
    """Backtest configuration."""

    agent_id: str
    symbol: str = Field(default="EURUSD=X")
    start_date: str = Field(default="2023-01-01")
    end_date: str = Field(default="2024-01-01")
    initial_balance: float = Field(default=100000, ge=1000)


class BacktestResult(BaseModel):
    """Backtest result."""

    backtest_id: str
    agent_id: str
    status: str
    metrics: Optional[dict] = None
    trades_count: int = 0


# In-memory storage
_backtest_jobs: dict = {}


@router.post("/run")
async def run_backtest(
    config: BacktestConfig,
    background_tasks: BackgroundTasks,
):
    """
    Run a backtest.

    Backtesting runs in background.
    """
    import uuid

    backtest_id = str(uuid.uuid4())[:8]

    job = {
        "backtest_id": backtest_id,
        "agent_id": config.agent_id,
        "status": "running",
        "metrics": None,
        "trades_count": 0,
    }

    _backtest_jobs[backtest_id] = job

    # Add background task (placeholder)
    # background_tasks.add_task(execute_backtest, backtest_id, config)

    logger.info(f"Backtest started: {backtest_id}")

    return {"backtest_id": backtest_id, "status": "running"}


@router.get("/{backtest_id}/results", response_model=BacktestResult)
async def get_backtest_results(backtest_id: str):
    """Get backtest results."""
    if backtest_id not in _backtest_jobs:
        raise HTTPException(status_code=404, detail="Backtest not found")

    return _backtest_jobs[backtest_id]


@router.get("/", response_model=List[BacktestResult])
async def list_backtests():
    """List all backtest jobs."""
    return list(_backtest_jobs.values())
