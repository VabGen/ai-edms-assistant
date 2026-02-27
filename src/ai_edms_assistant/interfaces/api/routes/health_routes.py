# src/ai_edms_assistant/interfaces/api/routes/health_routes.py
"""Health-check and readiness probe endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..dependencies import get_agent

router = APIRouter(prefix="/health", tags=["health"])


def _is_agent_ready() -> bool:
    """Check agent availability without raising an exception.

    Wraps ``get_agent()`` in a try/except so the health endpoint
    never raises a 503 when it should return 200 (liveness vs readiness).

    Returns:
        ``True`` when agent singleton is initialised and ready.
    """
    try:
        get_agent()
        return True
    except HTTPException:
        return False


@router.get("", summary="Liveness probe — сервис жив")
async def health() -> dict:
    """Always returns 200 if the process is running.

    Used by container orchestrators (K8s liveness probe).
    Liveness ≠ readiness: even when the agent is not yet initialised,
    the process is alive and should not be restarted.

    Returns:
        Dict with ``status`` and ``agent`` fields.
    """
    return {
        "status": "ok",
        "agent": "ready" if _is_agent_ready() else "initializing",
    }


@router.get("/ready", summary="Readiness probe — агент готов принимать запросы")
async def ready() -> dict:
    """Returns 200 only when EdmsDocumentAgent is fully initialized.

    Used as K8s readiness probe — traffic is held until this passes.

    Returns:
        Dict with ``status: "ready"`` when agent is up.

    Raises:
        HTTPException 503: Agent not yet initialized.
    """
    if not _is_agent_ready():
        raise HTTPException(status_code=503, detail="Agent not ready")
    return {"status": "ready"}