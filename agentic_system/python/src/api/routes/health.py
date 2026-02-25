"""
Health check endpoints.
"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns:
        {"status": "healthy"}
    """
    return {"status": "healthy", "service": "pipflow-ai"}


@router.get("/ready")
async def readiness_check():
    """
    Readiness check endpoint.

    Returns:
        {"ready": true}
    """
    return {"ready": True}
