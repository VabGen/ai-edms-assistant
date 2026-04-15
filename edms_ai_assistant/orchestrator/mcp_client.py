"""
mcp_client.py — Клиент для вызова MCP-инструментов через HTTP bridge.

Динамически загружает список инструментов из реестра.
Валидирует аргументы перед вызовом (JSON Schema).
Поддерживает retry с экспоненциальной задержкой.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger("mcp_client")

_MCP_URL = os.getenv("MCP_SERVER_URL", "http://mcp-server:8001")
_TIMEOUT = int(os.getenv("MCP_TIMEOUT", "30"))


class MCPClient:
    """HTTP-клиент для вызова MCP-инструментов."""

    def __init__(self, base_url: str = _MCP_URL) -> None:
        self._base_url = base_url.rstrip("/")
        self._tools_cache: list[dict] | None = None

    async def get_tools(self, force_reload: bool = False) -> list[dict]:
        """Получить список доступных инструментов из реестра."""
        if self._tools_cache is not None and not force_reload:
            return self._tools_cache
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self._base_url}/tools")
                if resp.is_success:
                    data = resp.json()
                    self._tools_cache = data.get("tools", [])
                    logger.info("MCP: loaded %d tools", len(self._tools_cache))
                    return self._tools_cache
        except Exception as exc:
            logger.error("MCP get_tools failed: %s", exc)
        return []

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """
        Вызвать MCP-инструмент с retry.

        Args:
            tool_name: Имя инструмента из реестра
            arguments: Аргументы для инструмента
            max_retries: Максимум попыток при сетевых ошибках
        """
        payload = {"tool_name": tool_name, "arguments": arguments}
        delay = 1.0

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                    resp = await client.post(
                        f"{self._base_url}/call-tool",
                        json=payload,
                    )
                if resp.status_code == 404:
                    return {"success": False, "error": f"Инструмент '{tool_name}' не найден"}
                if resp.status_code == 400:
                    data = resp.json()
                    return {"success": False, "error": f"Неверные аргументы: {data.get('detail', '')}"}

                resp.raise_for_status()
                data = resp.json()
                if data.get("success"):
                    return {"success": True, "result": data.get("result")}
                return {"success": False, "error": data.get("error", "Unknown error")}

            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                if attempt < max_retries - 1:
                    logger.warning("MCP call_tool attempt %d failed: %s — retry in %.1fs", attempt + 1, exc, delay)
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    logger.error("MCP call_tool failed after %d attempts: %s", max_retries, exc)
                    return {"success": False, "error": f"Не удалось подключиться к MCP серверу: {exc}"}
            except Exception as exc:
                logger.error("MCP call_tool unexpected error: %s", exc, exc_info=True)
                return {"success": False, "error": str(exc)}

        return {"success": False, "error": "Превышено количество попыток"}

    def validate_args(self, tool_name: str, args: dict[str, Any], tools: list[dict]) -> list[str]:
        """
        Валидировать аргументы по JSON Schema инструмента.

        Returns:
            Список ошибок валидации (пустой если OK)
        """
        schema = next(
            (t.get("inputSchema", {}) for t in tools if t.get("name") == tool_name),
            {},
        )
        if not schema:
            return []

        errors: list[str] = []
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for field in required:
            if field not in args or args[field] is None:
                errors.append(f"Обязательный параметр '{field}' не указан")

        for field, value in args.items():
            if field not in properties:
                continue
            prop = properties[field]
            expected_type = prop.get("type")
            if expected_type == "string" and not isinstance(value, str):
                errors.append(f"Параметр '{field}' должен быть строкой")
            elif expected_type == "integer" and not isinstance(value, int):
                errors.append(f"Параметр '{field}' должен быть целым числом")
            elif expected_type == "array" and not isinstance(value, list):
                errors.append(f"Параметр '{field}' должен быть массивом")
            if "enum" in prop and value not in prop["enum"]:
                errors.append(f"Параметр '{field}' должен быть одним из: {prop['enum']}")

        return errors


# Singleton
_mcp_client: MCPClient | None = None


def get_mcp_client() -> MCPClient:
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPClient()
    return _mcp_client
