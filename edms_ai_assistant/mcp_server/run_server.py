# edms_ai_assistant/mcp_server/run_server.py
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if sys.platform == "win32":
    import selectors

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    _loop = asyncio.SelectorEventLoop(selectors.SelectSelector())
    asyncio.set_event_loop(_loop)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main() -> None:
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8001"))
    logger.info("Starting EDMS MCP Server on %s:%d", host, port)

    from edms_ai_assistant.mcp_server.edms_mcp_server import mcp

    _run_via_uvicorn(mcp, host, port)


def _run_via_uvicorn(mcp_obj: object, host: str, port: int) -> None:
    """Запускает MCP-сервер с /health эндпоинтом.

    Правильный паттерн согласно документации fastmcp:
        mcp_app = mcp.http_app()
        app = Starlette(..., lifespan=mcp_app.lifespan)
        app.mount("/", mcp_app)
    """
    import uvicorn
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Mount, Route

    # Получаем StarletteWithLifespan один раз
    mcp_app = mcp_obj.http_app()  # type: ignore[attr-defined]
    logger.info("MCP app obtained: %s", type(mcp_app).__name__)

    async def health_endpoint(request):
        return JSONResponse({"status": "ok", "service": "mcp-server"})

    app = Starlette(
        routes=[
            Route("/health", health_endpoint, methods=["GET"]),
            Mount("/", app=mcp_app),
        ],
        lifespan=mcp_app.lifespan,
    )

    logger.info("Transport: Starlette+lifespan wrapping %s", type(mcp_app).__name__)

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        loop="asyncio",
    )


# ASGI app для прямого запуска через uvicorn:
#   uvicorn edms_ai_assistant.mcp_server.run_server:app --port 8001
try:
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Mount, Route

    from edms_ai_assistant.mcp_server.edms_mcp_server import mcp as _mcp

    _mcp_app = _mcp.http_app()

    async def _health(request):
        return JSONResponse({"status": "ok", "service": "mcp-server"})

    app = Starlette(
        routes=[
            Route("/health", _health, methods=["GET"]),
            Mount("/", app=_mcp_app),
        ],
        lifespan=_mcp_app.lifespan,
    )
except Exception as _err:
    logger.warning("Не удалось создать ASGI app: %s", _err)
    app = None  # type: ignore[assignment]


if __name__ == "__main__":
    main()
