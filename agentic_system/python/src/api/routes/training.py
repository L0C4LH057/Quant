"""
Training endpoints.
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

router = APIRouter()
logger = logging.getLogger(__name__)


class TrainingConfig(BaseModel):
    """Training configuration."""

    agent_id: str
    total_timesteps: int = Field(default=10000, ge=1000, le=1000000)
    symbol: str = Field(default="EURUSD=X")
    start_date: str = Field(default="2023-01-01")
    end_date: str = Field(default="2024-01-01")


class TrainingStatus(BaseModel):
    """Training status."""

    training_id: str
    agent_id: str
    status: str
    progress: float
    current_timestep: int
    total_timesteps: int
    best_reward: Optional[float] = None


# In-memory tracking (replace with database)
_training_jobs: dict = {}


@router.post("/start")
async def start_training(
    config: TrainingConfig,
    background_tasks: BackgroundTasks,
):
    """
    Start training a new agent.

    Training runs in background.
    """
    import uuid

    training_id = str(uuid.uuid4())[:8]

    job = {
        "training_id": training_id,
        "agent_id": config.agent_id,
        "status": "queued",
        "progress": 0.0,
        "current_timestep": 0,
        "total_timesteps": config.total_timesteps,
        "best_reward": None,
    }

    _training_jobs[training_id] = job

    # Add background task (placeholder)
    # background_tasks.add_task(run_training, training_id, config)

    logger.info(f"Training queued: {training_id} for agent {config.agent_id}")

    return {"training_id": training_id, "status": "queued"}


@router.get("/{training_id}/status", response_model=TrainingStatus)
async def get_training_status(training_id: str):
    """Get training status."""
    if training_id not in _training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")

    return _training_jobs[training_id]


@router.post("/{training_id}/stop")
async def stop_training(training_id: str):
    """Stop training."""
    if training_id not in _training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")

    job = _training_jobs[training_id]
    job["status"] = "stopped"

    logger.info(f"Training stopped: {training_id}")

    return {"training_id": training_id, "status": "stopped"}
