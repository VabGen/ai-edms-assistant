# src/ai_edms_assistant/interfaces/api/middleware.py
"""
FastAPI middleware stack.

All middleware is registered through register_middleware() called once
from app.py. Middleware is applied in reverse registration order by Starlette.
"""

from __future__ import annotations

import time
import uuid
import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

logger = structlog.get_logger(__name__)


def register_middleware(app: FastAPI) -> None:
    """
    Attach all middleware to the FastAPI application instance.

    Registered in order (applied in reverse by Starlette):
        1. CORS — allow all origins (tighten in prod via settings)
        2. Request-ID injection + structured access log

    Args:
        app: The FastAPI application to configure.
    """
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request-ID + structured access logging
    @app.middleware("http")
    async def request_id_logging(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()

        response = await call_next(request)

        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=duration_ms,
            request_id=request_id,
        )
        response.headers["X-Request-ID"] = request_id
        return response
