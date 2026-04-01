"""
EDMS MCP Server HTTP Bridge — FastAPI приложение поверх MCP сервера.

Добавляет HTTP эндпоинты:
  POST /call-tool   — вызов MCP инструмента из оркестратора
  GET  /health      — healthcheck
  GET  /metrics     — Prometheus метрики
  GET  /tools       — список инструментов

Запуск: uvicorn mcp_http_bridge:app --host 0.0.0.0 --port 8001
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("mcp_bridge")

os.makedirs("/app/logs", exist_ok=True)
try:
    fh = logging.FileHandler("/app/logs/mcp_server.log", encoding="utf-8")
    logging.getLogger().addHandler(fh)
except Exception:
    pass

EDMS_BASE_URL = os.getenv("EDMS_API_URL", "http://localhost:8098")
PORT = int(os.getenv("MCP_PORT", "8001"))

# ── Import all tool functions ─────────────────────────────────────────────────
from edms_mcp_server import (
    get_document,
    search_documents,
    get_document_history,
    get_document_versions,
    get_document_statistics,
    search_employees,
    get_current_user,
    create_task,
    create_introduction,
    execute_document_operation,
    start_document_routing,
    set_document_control,
    send_notification,
    get_reference_data,
    health_check as mcp_health_check,
)

TOOL_MAP: dict[str, Any] = {
    "get_document": get_document,
    "search_documents": search_documents,
    "get_document_history": get_document_history,
    "get_document_versions": get_document_versions,
    "get_document_statistics": get_document_statistics,
    "search_employees": search_employees,
    "get_current_user": get_current_user,
    "create_task": create_task,
    "create_introduction": create_introduction,
    "execute_document_operation": execute_document_operation,
    "start_document_routing": start_document_routing,
    "set_document_control": set_document_control,
    "send_notification": send_notification,
    "get_reference_data": get_reference_data,
    "health_check": mcp_health_check,
}

# ── Metrics ───────────────────────────────────────────────────────────────────
_metrics: dict[str, Any] = {
    "tool_calls_total": 0,
    "tool_errors_total": 0,
    "by_tool": {},
}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EDMS MCP Server",
    version="1.0.0",
    description="HTTP bridge for EDMS MCP tools",
)


class ToolCallRequest(BaseModel):
    tool: str
    args: dict[str, Any] = {}


@app.post("/call-tool")
async def call_tool(req: ToolCallRequest) -> JSONResponse:
    """Вызвать MCP инструмент по имени с заданными аргументами."""
    tool_func = TOOL_MAP.get(req.tool)
    if not tool_func:
        raise HTTPException(
            status_code=404,
            detail=f"Tool '{req.tool}' not found. Available: {sorted(TOOL_MAP.keys())}",
        )

    _metrics["tool_calls_total"] += 1
    _metrics["by_tool"][req.tool] = _metrics["by_tool"].get(req.tool, 0) + 1

    start = time.monotonic()
    try:
        result = await tool_func(**req.args)
        latency_ms = int((time.monotonic() - start) * 1000)
        logger.info("Tool '%s' succeeded in %dms", req.tool, latency_ms)

        # Инструменты возвращают JSON-строку, парсим её
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                return JSONResponse(parsed)
            except json.JSONDecodeError:
                return JSONResponse({"result": result})
        return JSONResponse(result if isinstance(result, dict) else {"result": result})

    except TypeError as exc:
        # Неверные аргументы
        _metrics["tool_errors_total"] += 1
        logger.error("Tool '%s' bad args: %s", req.tool, exc)
        raise HTTPException(
            status_code=422,
            detail=f"Invalid arguments for tool '{req.tool}': {exc}",
        )
    except Exception as exc:
        _metrics["tool_errors_total"] += 1
        latency_ms = int((time.monotonic() - start) * 1000)
        logger.error("Tool '%s' failed in %dms: %s", req.tool, latency_ms, exc)
        return JSONResponse(
            {"success": False, "error": str(exc), "tool": req.tool},
            status_code=500,
        )


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
        "tools_available": len(TOOL_MAP),
        "calls_total": _metrics["tool_calls_total"],
        "errors_total": _metrics["tool_errors_total"],
        "timestamp": datetime.now().isoformat(),
    })


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    lines = [
        "# HELP edms_mcp_tool_calls_total Total tool invocations",
        "# TYPE edms_mcp_tool_calls_total counter",
        f"edms_mcp_tool_calls_total {_metrics['tool_calls_total']}",
        "",
        "# HELP edms_mcp_tool_errors_total Total tool errors",
        "# TYPE edms_mcp_tool_errors_total counter",
        f"edms_mcp_tool_errors_total {_metrics['tool_errors_total']}",
        "",
        "# HELP edms_mcp_calls_by_tool Per-tool call counts",
        "# TYPE edms_mcp_calls_by_tool counter",
    ]
    for tool, count in _metrics["by_tool"].items():
        lines.append(f'edms_mcp_calls_by_tool{{tool="{tool}"}} {count}')
    return "\n".join(lines)


@app.get("/tools")
async def list_tools() -> JSONResponse:
    tools_path = os.path.join(os.path.dirname(__file__), "tools_registry.json")
    try:
        with open(tools_path, encoding="utf-8") as f:
            return JSONResponse(json.load(f))
    except FileNotFoundError:
        return JSONResponse({
            "tools": [{"name": k} for k in TOOL_MAP.keys()],
        })


if __name__ == "__main__":
    logger.info("Starting EDMS MCP HTTP Bridge on port %d", PORT)
    uvicorn.run(
        "mcp_http_bridge:app",
        host="0.0.0.0",
        port=PORT,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
