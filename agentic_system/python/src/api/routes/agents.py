"""
Agent management endpoints.
"""
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()
logger = logging.getLogger(__name__)


class AgentConfig(BaseModel):
    """Agent configuration."""

    name: str = Field(..., min_length=1, max_length=100)
    algorithm: str = Field(..., pattern="^(PPO|A2C|SAC|TD3|DQN)$")
    config: dict = Field(default_factory=dict)
    description: Optional[str] = None


class AgentResponse(BaseModel):
    """Agent response model."""

    id: str
    name: str
    algorithm: str
    status: str
    config: dict


# In-memory storage (replace with database)
_agents: dict = {}


@router.get("/", response_model=List[AgentResponse])
async def list_agents():
    """List all agents."""
    return list(_agents.values())


@router.post("/", response_model=AgentResponse)
async def create_agent(config: AgentConfig):
    """
    Create a new RL agent.

    Args:
        config: Agent configuration
    """
    import uuid

    agent_id = str(uuid.uuid4())[:8]

    agent = {
        "id": agent_id,
        "name": config.name,
        "algorithm": config.algorithm,
        "status": "idle",
        "config": config.config,
    }

    _agents[agent_id] = agent
    logger.info(f"Created agent: {agent_id} ({config.algorithm})")

    return agent


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str):
    """Get agent by ID."""
    if agent_id not in _agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    return _agents[agent_id]


@router.delete("/{agent_id}")
async def delete_agent(agent_id: str):
    """Delete agent."""
    if agent_id not in _agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    del _agents[agent_id]
    logger.info(f"Deleted agent: {agent_id}")

    return {"status": "deleted", "id": agent_id}


@router.get("/{agent_id}/status")
async def get_agent_status(agent_id: str):
    """Get agent status."""
    if agent_id not in _agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = _agents[agent_id]
    return {
        "id": agent_id,
        "status": agent["status"],
        "algorithm": agent["algorithm"],
    }
