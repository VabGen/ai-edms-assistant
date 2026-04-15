import argparse
import logging
import sys
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


def _run_via_uvicorn(mcp_obj, host, port):
    """
    Запускает MCP сервер как стандартное ASGI приложение через Uvicorn.
    Мы пропускаем метод .run() и отдаем приложение напрямую в Uvicorn.
    """
    app = getattr(mcp_obj, "app", mcp_obj)

    logger.info("Starting Uvicorn on %s:%d", host, port)
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    except KeyboardInterrupt:
        logger.info("Server stopped.")


def main():
    parser = argparse.ArgumentParser(description="EDMS MCP Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    args = parser.parse_args()

    try:
        from edms_ai_assistant.mcp_server.edms_mcp_server import mcp
    except ImportError:
        logger.error("Не удалось импортировать модуль edms_mcp_server.")
        sys.exit(1)

    logger.info("Starting EDMS MCP Server on %s:%d", args.host, args.port)

    try:
        _run_via_uvicorn(mcp, args.host, args.port)
    except Exception as e:
        logger.error("Server failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()