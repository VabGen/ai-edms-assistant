# edms_ai_assistant/mcp_server/run_server.py
"""
Точка запуска EDMS MCP Server.

fastmcp 2.x изменил транспортную модель:
  - mcp.run()            → stdio-транспорт (для Claude Desktop, не для HTTP)
  - mcp.run_http_async() → HTTP-транспорт (fastmcp 2.x, если есть)
  - uvicorn + mcp.app    → HTTP через ASGI (универсальный fallback)

Этот файл автоматически определяет доступный способ запуска.

Запуск:
    python -m edms_ai_assistant.mcp_server.run_server
    python edms_ai_assistant/mcp_server/run_server.py
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import os
import sys
from pathlib import Path

# ── sys.path ──────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _get_transport_strategy(mcp_obj: object) -> str:
    """Определяет доступный HTTP-транспорт для данной версии fastmcp.

    Returns:
        "http_async"  — есть run_http_async(host, port)
        "streamable"  — есть run_streamable_http_async(host, port)
        "uvicorn"     — нет HTTP-методов, используем uvicorn + ASGI
    """
    for method_name in ("run_http_async", "run_streamable_http_async"):
        method = getattr(mcp_obj, method_name, None)
        if method is None:
            continue
        params = set(inspect.signature(method).parameters.keys())
        if "host" in params and "port" in params:
            return "http_async" if method_name == "run_http_async" else "streamable"
    return "uvicorn"


def _get_asgi_app(mcp_obj: object) -> object:
    """Извлекает ASGI-приложение из объекта FastMCP."""
    return (
        getattr(mcp_obj, "app", None)
        or getattr(mcp_obj, "_app", None)
        or getattr(mcp_obj, "http_app", None)
        or mcp_obj
    )


def _run_via_uvicorn(mcp_obj: object, host: str, port: int) -> None:
    """Запускает HTTP-сервер через uvicorn + ASGI.

    Универсальный fallback: работает с любой версией fastmcp,
    предоставляющей ASGI-совместимый объект.
    """
    import uvicorn

    asgi_app = _get_asgi_app(mcp_obj)
    logger.info("Transport: uvicorn ASGI (app type: %s)", type(asgi_app).__name__)

    # Windows требует SelectorEventLoop для asyncio-совместимости
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    uvicorn.run(
        asgi_app,
        host=host,
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )


def main() -> None:
    """Запускает EDMS MCP Server с автоопределением транспорта."""
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8001"))

    logger.info("Starting EDMS MCP Server on %s:%d", host, port)
    logger.info("EDMS Base URL: %s", os.getenv("EDMS_BASE_URL", "not set"))

    from edms_ai_assistant.mcp_server.edms_mcp_server import mcp  # noqa: PLC0415

    strategy = _get_transport_strategy(mcp)
    logger.info("Выбран транспорт: %s", strategy)

    try:
        if strategy == "http_async":
            logger.info("Transport: run_http_async")
            asyncio.run(mcp.run_http_async(host=host, port=port))  # type: ignore[attr-defined]
        elif strategy == "streamable":
            logger.info("Transport: run_streamable_http_async")
            asyncio.run(mcp.run_streamable_http_async(host=host, port=port))  # type: ignore[attr-defined]
        else:
            _run_via_uvicorn(mcp, host, port)

    except KeyboardInterrupt:
        logger.info("MCP Server stopped")
    except Exception as exc:
        logger.critical("MCP Server failed: %s", exc, exc_info=True)
        sys.exit(1)


# ── ASGI app для прямого uvicorn ──────────────────────────────────────────────
# uvicorn edms_ai_assistant.mcp_server.run_server:app --host 0.0.0.0 --port 8001
try:
    from edms_ai_assistant.mcp_server.edms_mcp_server import mcp as _mcp  # noqa: PLC0415
    app = _get_asgi_app(_mcp)
except Exception as _err:
    logger.warning("Не удалось создать ASGI app: %s", _err)
    app = None  # type: ignore[assignment]


if __name__ == "__main__":
    main()