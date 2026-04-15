"""
mcp_http_bridge.py — HTTP обёртка над MCP-сервером.

POST /call-tool  — вызов инструмента
GET  /tools      — список доступных инструментов
GET  /health     — состояние сервера
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("mcp_bridge")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="EDMS MCP HTTP Bridge", version="1.0.0")

# Загружаем реестр инструментов
_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "tools_registry.json")
try:
    with open(_REGISTRY_PATH, encoding="utf-8") as f:
        _REGISTRY: dict = json.load(f)
except FileNotFoundError:
    _REGISTRY = {"tools": []}

# Прямые вызовы через edms_mcp_server функции (in-process)
sys.path.insert(0, os.path.dirname(__file__))
import edms_mcp_server as _mcp_module

_TOOL_MAP: dict[str, Any] = {}
for _func_name in dir(_mcp_module):
    _obj = getattr(_mcp_module, _func_name)
    if callable(_obj) and not _func_name.startswith("_"):
        _TOOL_MAP[_func_name] = _obj


class ToolCallRequest(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = {}


class ToolCallResponse(BaseModel):
    success: bool
    result: Any = None
    error: str | None = None


@app.post("/call-tool", response_model=ToolCallResponse)
async def call_tool(req: ToolCallRequest) -> ToolCallResponse:
    """Вызвать инструмент MCP по имени."""
    func = _TOOL_MAP.get(req.tool_name)
    if func is None:
        raise HTTPException(status_code=404, detail=f"Инструмент '{req.tool_name}' не найден")
    try:
        if asyncio.iscoroutinefunction(func):
            result = await func(**req.arguments)
        else:
            result = func(**req.arguments)
        return ToolCallResponse(success=True, result=result)
    except TypeError as exc:
        logger.warning("[bridge] TypeError calling %s: %s", req.tool_name, exc)
        raise HTTPException(status_code=400, detail=f"Неверные аргументы: {exc}")
    except Exception as exc:
        logger.error("[bridge] Error calling %s: %s", req.tool_name, exc, exc_info=True)
        return ToolCallResponse(success=False, error=str(exc))


@app.get("../orchestrator/tools")
async def list_tools() -> dict:
    """Список доступных инструментов."""
    return _REGISTRY


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "mcp-bridge", "tools_count": len(_REGISTRY.get("../orchestrator/tools", []))}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("MCP_HTTP_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
