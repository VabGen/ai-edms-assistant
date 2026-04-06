# orchestrator/api/routes/settings.py
"""GET/PATCH /api/settings — настройки сессии пользователя."""
from __future__ import annotations

from fastapi import APIRouter, Body

from edms_ai_assistant.config import settings

router = APIRouter(prefix="/api/settings", tags=["Settings"])


@router.get("", summary="Получить публичные настройки")
async def get_settings_endpoint() -> dict:
    data = {
        "agent_max_iterations": settings.AGENT_MAX_ITERATIONS,
        "agent_timeout": settings.AGENT_TIMEOUT,
        "cache_ttl_seconds": settings.CACHE_TTL_SECONDS,
        "environment": settings.ENVIRONMENT,
    }
    if settings.SETTINGS_PANEL_SHOW_TECHNICAL:
        data["llm_model"] = settings.LLM_GENERATIVE_MODEL
        data["mcp_url"] = str(settings.MCP_URL)
    return data


@router.patch("", summary="Обновить настройки сессии (только user-level)")
async def patch_settings(body: dict = Body(default={})) -> dict:
    # User-level настройки хранятся в Redis-сессии — здесь заглушка
    return {"status": "ok", "applied": list(body.keys())}
