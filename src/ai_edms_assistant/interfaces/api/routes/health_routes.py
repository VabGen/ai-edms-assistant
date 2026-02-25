# src/ai_edms_assistant/interfaces/api/routes/health_routes.py
"""Health-check and readiness probe endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..dependencies import _agent

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", summary="Liveness probe — сервис жив")
async def health() -> dict:
    """
    Always returns 200 if the process is running.
    Used by container orchestrators (K8s liveness probe).
    """
    return {
        "status": "ok",
        "agent": "ready" if _agent is not None else "initializing",
    }


@router.get("/ready", summary="Readiness probe — агент готов принимать запросы")
async def ready() -> dict:
    """
    Returns 200 only when EdmsDocumentAgent is fully initialized.
    Used as K8s readiness probe — traffic is held until this passes.

    Raises:
        HTTPException 503: Agent not yet initialized.
    """
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")
    return {"status": "ready"}
