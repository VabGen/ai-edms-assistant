# src/ai_edms_assistant/main.py
"""
EDMS AI Assistant — точка входа (compatibility shim).

Этот файл сохраняет обратную совместимость с uvicorn-командой вида::

    uvicorn main:app --reload

Вся логика приложения перенесена в:
    - interfaces/api/app.py        ← FastAPI factory + lifespan
    - interfaces/api/routes/       ← роуты
    - interfaces/api/dependencies.py ← DI контейнер
    - interfaces/api/middleware.py  ← CORS, logging
"""
from __future__ import annotations

import uvicorn

from ai_edms_assistant.interfaces.api.app import app  # noqa: F401

if __name__ == "__main__":
    from ai_edms_assistant.shared.config import settings

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOGGING_LEVEL.lower(),
    )