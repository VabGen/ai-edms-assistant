# mcp-server/run_server.py
"""
Точка запуска EDMS MCP Server.

Читает конфигурацию из переменных окружения.
Запускает FastMCP сервер на указанном хосте и порту.

CMD в Dockerfile:
    CMD ["python", "run_server.py"]
"""
from __future__ import annotations

import logging
import os
import sys

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Запускает EDMS MCP Server."""
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8001"))

    logger.info("Starting EDMS MCP Server on %s:%d", host, port)
    logger.info("EDMS API URL: %s", os.getenv("EDMS_API_URL", "not set"))

    from edms_mcp_server import mcp

    try:
        mcp.run(host=host, port=port)
    except KeyboardInterrupt:
        logger.info("MCP Server stopped")
    except Exception as exc:
        logger.critical("MCP Server failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
