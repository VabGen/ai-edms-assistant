"""
EDMS MCP Server — HTTP wrapper с /health эндпоинтом для Docker healthcheck.

Запускает FastMCP сервер в режиме SSE и одновременно поднимает
минимальный FastAPI сервер на том же порту для /health.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse

# Импортируем mcp instance из основного файла
from edms_mcp_server import mcp, EDMS_BASE_URL, logger

# ── Metrics ───────────────────────────────────────────────────────────────────
_tool_call_count: dict[str, int] = {}
_error_count: int = 0
_total_calls: int = 0


def _track_call(tool_name: str, success: bool) -> None:
    global _total_calls, _error_count
    _total_calls += 1
    _tool_call_count[tool_name] = _tool_call_count.get(tool_name, 0) + 1
    if not success:
        _error_count += 1


# ── FastAPI health app ────────────────────────────────────────────────────────
health_app = FastAPI(title="EDMS MCP Health", version="1.0.0")


@health_app.get("/health")
async def health() -> JSONResponse:
    """Healthcheck для Docker и оркестратора."""
    edms_ok = False
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(f"{EDMS_BASE_URL}/actuator/health")
            edms_ok = resp.status_code < 400
    except Exception:
        pass

    return JSONResponse({
        "status": "healthy",
        "mcp_server": "running",
        "edms_api": "healthy" if edms_ok else "unavailable",
        "edms_url": EDMS_BASE_URL,
        "tools_called": _total_calls,
        "errors": _error_count,
        "timestamp": datetime.now().isoformat(),
    })


@health_app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    """Метрики Prometheus."""
    lines = [
        f"# HELP edms_mcp_tool_calls_total Total MCP tool calls",
        f"# TYPE edms_mcp_tool_calls_total counter",
        f"edms_mcp_tool_calls_total {_total_calls}",
        f"",
        f"# HELP edms_mcp_errors_total Total MCP tool errors",
        f"# TYPE edms_mcp_errors_total counter",
        f"edms_mcp_errors_total {_error_count}",
    ]
    for tool, count in _tool_call_count.items():
        lines.append(f'edms_mcp_tool_calls{{tool="{tool}"}} {count}')
    return "\n".join(lines)


@health_app.get("/tools")
async def list_tools() -> JSONResponse:
    """Список зарегистрированных MCP инструментов."""
    try:
        tools_path = os.path.join(os.path.dirname(__file__), "tools_registry.json")
        with open(tools_path, encoding="utf-8") as f:
            registry = json.load(f)
        return JSONResponse(registry)
    except FileNotFoundError:
        return JSONResponse({"tools": [], "error": "tools_registry.json not found"})


if __name__ == "__main__":
    port = int(os.getenv("MCP_PORT", "8001"))
    uvicorn.run(health_app, host="0.0.0.0", port=port, log_level="info")
