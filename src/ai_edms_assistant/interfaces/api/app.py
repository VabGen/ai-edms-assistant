# src/ai_edms_assistant/interfaces/api/app.py
"""
FastAPI application factory.

Extracts all setup logic from the monolithic main.py into a clean
factory function. The module-level ``app`` singleton is picked up by
uvicorn/gunicorn automatically.

Entry point::

    uvicorn ai_edms_assistant.interfaces.api.app:app --reload
"""

from __future__ import annotations

import shutil
import structlog
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .dependencies import set_agent, set_http_client, UPLOAD_DIR
from .middleware import register_middleware
from .routes.agent_routes import router as agent_router
from .routes.document_routes import router as document_router
from .routes.task_routes import router as task_router
from .routes.health_routes import router as health_router

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Startup order:
        1. Create temp upload directory
        2. Initialise shared EdmsHttpClient
        3. Initialise EdmsDocumentAgent (LangGraph compilation)

    Shutdown:
        4. Remove temp upload directory
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    from ...infrastructure.edms_api.http_client import EdmsHttpClient
    from ...application.agents import EdmsDocumentAgent

    http_client = EdmsHttpClient()
    set_http_client(http_client)
    logger.info("edms_http_client_ready")

    try:
        agent = EdmsDocumentAgent()
        set_agent(agent)
        logger.info("edms_agent_ready")
    except Exception as exc:
        logger.error("edms_agent_init_failed", error=str(exc))
        # Стартуем без агента — health /ready вернёт 503

    yield

    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        logger.info("upload_dir_cleaned")


def create_app() -> FastAPI:
    """
    FastAPI application factory.

    Registers middleware, routes and the lifespan manager.

    Returns:
        Fully configured FastAPI instance.
    """
    app = FastAPI(
        title="EDMS AI Assistant API",
        version="3.0.0",
        description="AI-powered assistant for EDMS document management",
        lifespan=lifespan,
    )

    register_middleware(app)

    app.include_router(health_router)
    app.include_router(agent_router)
    app.include_router(document_router)
    app.include_router(task_router)

    return app


# Module-level singleton for uvicorn / gunicorn
app = create_app()
