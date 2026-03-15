"""
Backtesting endpoints.

BUG-03 fix: Implements execute_backtest and wires it to background_tasks.
BUG-08 fix: Persists backtest jobs to disk JSON files so they survive restarts.
"""
import json
import logging
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from ...config.base import get_config

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Pydantic models (renamed to avoid collision with engine.py) ──────────


class BacktestRequest(BaseModel):
    """Backtest configuration submitted by the caller."""

    agent_id: str
    symbol: str = Field(default="EURUSD=X")
    start_date: str = Field(default="2023-01-01")
    end_date: str = Field(default="2024-01-01")
    initial_balance: float = Field(default=100_000, ge=1000)


class BacktestJobStatus(BaseModel):
    """Status / result for a backtest job."""

    backtest_id: str
    agent_id: str
    status: str
    metrics: Optional[dict] = None
    trades_count: int = 0
    error: Optional[str] = None


# ── Persistent job store helpers ─────────────────────────────────────────


def _jobs_dir() -> Path:
    config = get_config()
    d = config.backtest_results_path / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_job(job: dict) -> None:
    path = _jobs_dir() / f"{job['backtest_id']}.json"
    path.write_text(json.dumps(job, indent=2))


def _load_job(backtest_id: str) -> Optional[dict]:
    path = _jobs_dir() / f"{backtest_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def _load_all_jobs() -> List[dict]:
    jobs = []
    for p in sorted(_jobs_dir().glob("*.json")):
        try:
            jobs.append(json.loads(p.read_text()))
        except Exception:
            continue
    return jobs


# ── Background task implementation ───────────────────────────────────────


async def execute_backtest(backtest_id: str, config: BacktestRequest) -> None:
    """Run backtest in background and persist result."""
    try:
        from ...data.fetcher import MarketDataFetcher
        from ...backtesting.engine import BacktestEngine, BacktestConfig as EngineConfig
        from ...backtesting.metrics import MetricsCalculator

        # 1. Fetch data
        fetcher = MarketDataFetcher()
        df = fetcher.fetch(config.symbol, config.start_date, config.end_date)

        if df is None or df.empty:
            raise ValueError(f"No data for {config.symbol}")

        # 2. Create engine and run
        engine_config = EngineConfig(
            initial_balance=config.initial_balance,
            transaction_cost_pct=0.001,
        )
        engine = BacktestEngine(config=engine_config)
        result = engine.run(df)

        # 3. Compute metrics
        equity = result.get("equity_curve", [config.initial_balance])
        trades_df = result.get("trades")
        calculator = MetricsCalculator(equity=equity, trades=trades_df)
        metrics = calculator.calculate_all()

        # 4. Save completed job
        job = {
            "backtest_id": backtest_id,
            "agent_id": config.agent_id,
            "status": "completed",
            "metrics": metrics.to_dict(),
            "trades_count": metrics.total_trades,
            "error": None,
        }
        _save_job(job)
        logger.info(f"Backtest {backtest_id} completed — trades={metrics.total_trades}")

    except Exception as e:
        logger.exception(f"Backtest {backtest_id} failed: {e}")
        job = {
            "backtest_id": backtest_id,
            "agent_id": config.agent_id,
            "status": "failed",
            "metrics": None,
            "trades_count": 0,
            "error": str(e),
        }
        _save_job(job)


# ── Routes ───────────────────────────────────────────────────────────────


@router.post("/run")
async def run_backtest(
    config: BacktestRequest,
    background_tasks: BackgroundTasks,
):
    """Start a backtest in the background."""
    backtest_id = str(uuid.uuid4())[:8]

    job = {
        "backtest_id": backtest_id,
        "agent_id": config.agent_id,
        "status": "running",
        "metrics": None,
        "trades_count": 0,
        "error": None,
    }
    _save_job(job)

    background_tasks.add_task(execute_backtest, backtest_id, config)

    logger.info(f"Backtest started: {backtest_id}")
    return {"backtest_id": backtest_id, "status": "running"}


@router.get("/{backtest_id}/results", response_model=BacktestJobStatus)
async def get_backtest_results(backtest_id: str):
    """Get backtest results."""
    job = _load_job(backtest_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Backtest not found")
    return job


@router.get("/", response_model=List[BacktestJobStatus])
async def list_backtests():
    """List all backtest jobs."""
    return _load_all_jobs()
