#!/usr/bin/env python3
"""
EDMS MCP Server — точка входа.
Запускает FastAPI (health + metrics + /tools) + MCP SSE на одном порту.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse

EDMS_BASE_URL = os.getenv("EDMS_API_URL", "http://localhost:8098")
MCP_PORT = int(os.getenv("MCP_PORT", "8001"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/app/logs/mcp_server.log", encoding="utf-8"),
    ],
)

from edms_mcp_server import mcp

app = FastAPI(title="EDMS MCP Server", version="1.0.0")

try:
    app.mount("/sse", mcp.sse_app())
except Exception:
    pass  # SSE transport may not be available in all mcp versions

_metrics: dict[str, int] = {"calls_total": 0, "errors_total": 0}
_tool_counts: dict[str, int] = {}


@app.get("/health")
async def health() -> JSONResponse:
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
        "tools_called": _metrics["calls_total"],
        "errors": _metrics["errors_total"],
        "timestamp": datetime.now().isoformat(),
    })


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    lines = [
        "# HELP edms_mcp_calls_total Total MCP tool calls",
        "# TYPE edms_mcp_calls_total counter",
        f"edms_mcp_calls_total {_metrics['calls_total']}",
        "",
        "# HELP edms_mcp_errors_total Total MCP tool errors",
        "# TYPE edms_mcp_errors_total counter",
        f"edms_mcp_errors_total {_metrics['errors_total']}",
    ]
    for tool, count in _tool_counts.items():
        lines.append(f'edms_mcp_tool_calls_total{{tool="{tool}"}} {count}')
    return "\n".join(lines)


@app.get("/tools")
async def list_tools() -> JSONResponse:
    try:
        registry_path = os.path.join(os.path.dirname(__file__), "tools_registry.json")
        with open(registry_path, encoding="utf-8") as f:
            return JSONResponse(json.load(f))
    except FileNotFoundError:
        return JSONResponse({"tools": [], "error": "tools_registry.json not found"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=MCP_PORT, log_level=LOG_LEVEL.lower())
