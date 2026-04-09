# edms_ai_assistant/orchestrator/main_entrypoint.py
"""
Точка входа для Windows-совместимого запуска оркестратора.

На Windows Python 3.12+ по умолчанию использует ProactorEventLoop,
который несовместим с psycopg (async postgres driver).
Этот файл переключает event loop на SelectorEventLoop ДО запуска uvicorn.

Запуск:
    python -m edms_ai_assistant.orchestrator.main_entrypoint
    # или через этот файл:
    python edms_ai_assistant/orchestrator/main_entrypoint.py
"""
from __future__ import annotations

import sys


def _patch_event_loop_windows() -> None:
    """Переключает event loop на SelectorEventLoop на Windows.

    ProactorEventLoop (дефолт Windows) несовместим с:
    - psycopg AsyncConnection (langgraph-checkpoint-postgres)
    - некоторыми версиями asyncpg

    SelectorEventLoop работает на всех платформах.
    """
    if sys.platform != "win32":
        return

    import asyncio
    import selectors

    # Проверяем, не переключён ли уже
    current = asyncio.get_event_loop_policy()
    if isinstance(current, asyncio.WindowsSelectorEventLoopPolicy):
        return

    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )

    # Пересоздаём loop с правильным selector
    loop = asyncio.SelectorEventLoop(selectors.SelectSelector())
    asyncio.set_event_loop(loop)


# Патч ДОЛЖЕН быть применён до любых импортов uvicorn/fastapi/psycopg
_patch_event_loop_windows()

import uvicorn  # noqa: E402
from edms_ai_assistant.config import settings  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(
        "edms_ai_assistant.orchestrator.main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOGGING_LEVEL.lower(),
        # loop="asyncio" гарантирует использование нашего SelectorEventLoop
        loop="asyncio",
    )