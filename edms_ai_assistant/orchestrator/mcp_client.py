"""
MCP HTTP Client — HTTP-клиент для вызова MCP инструментов из оркестратора.

Поскольку MCP сервер работает как отдельный сервис, оркестратор вызывает
инструменты через HTTP API, а не через MCP SDK напрямую.

Паттерн вызова:
  POST /call-tool  → {"tool": "search_documents", "args": {...}, "token": "jwt..."}
  GET  /tools      → список доступных инструментов
  GET  /health     → статус сервера
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

logger = logging.getLogger("mcp_client")

# Retry настройки
_MAX_RETRIES = 3
_RETRY_DELAYS = [0.5, 1.0, 2.0]  # секунды между попытками


class MCPClient:
    """
    HTTP-клиент для взаимодействия с EDMS MCP сервером.

    Поддерживает:
    - Вызов инструментов с автоматическим retry
    - Динамическое получение списка инструментов
    - Инъекцию токена авторизации
    - Структурированную обработку ошибок
    """

    def __init__(self, base_url: str, timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._tools_cache: list[dict] | None = None
        self._tools_cache_at: float = 0.0
        self._tools_cache_ttl: float = 60.0  # 1 минута

    async def call_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        token: str = "",
    ) -> dict[str, Any]:
        """
        Вызвать MCP инструмент с указанными аргументами.

        Args:
            tool_name: Имя инструмента (например, "search_documents")
            args: Аргументы инструмента
            token: JWT токен EDMS для передачи в инструмент

        Returns:
            Распарсенный JSON-ответ инструмента

        Raises:
            MCPToolError: При ошибке вызова инструмента
        """
        # Инжектируем токен в аргументы
        call_args = {**args, "token": token}

        last_error: Exception | None = None

        for attempt, delay in enumerate([(0, 0)] + list(enumerate(_RETRY_DELAYS))):
            if attempt > 0:
                import asyncio
                await asyncio.sleep(_RETRY_DELAYS[attempt - 1])

            try:
                start = time.monotonic()
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        f"{self.base_url}/call-tool",
                        json={"tool": tool_name, "args": call_args},
                        headers={"Content-Type": "application/json"},
                    )

                latency_ms = int((time.monotonic() - start) * 1000)
                logger.debug(
                    "MCP tool '%s' completed in %dms (attempt %d)",
                    tool_name, latency_ms, attempt + 1,
                )

                if resp.status_code == 404:
                    return {"error": f"Инструмент '{tool_name}' не найден на MCP сервере"}

                resp.raise_for_status()

                # Ответ инструмента — всегда строка (JSON или текст)
                raw = resp.json()
                if isinstance(raw, str):
                    try:
                        return json.loads(raw)
                    except json.JSONDecodeError:
                        return {"result": raw}
                return raw

            except httpx.ConnectError as exc:
                last_error = exc
                logger.warning(
                    "MCP server unavailable (attempt %d/%d): %s",
                    attempt + 1, _MAX_RETRIES, exc,
                )
            except httpx.TimeoutException as exc:
                last_error = exc
                logger.warning("MCP tool '%s' timed out (attempt %d)", tool_name, attempt + 1)
            except httpx.HTTPStatusError as exc:
                # 5xx — retry, 4xx — не retry
                if exc.response.status_code >= 500:
                    last_error = exc
                    logger.warning("MCP server 5xx (attempt %d): %s", attempt + 1, exc)
                else:
                    return {"error": f"MCP ошибка {exc.response.status_code}: {exc.response.text[:200]}"}

        # Все попытки исчерпаны
        return {
            "error": f"MCP сервер недоступен после {_MAX_RETRIES} попыток: {last_error}",
            "tool": tool_name,
        }

    async def call_tool_direct(
        self,
        tool_name: str,
        args: dict[str, Any],
        token: str = "",
    ) -> dict[str, Any]:
        """
        Вызов инструмента напрямую через Python импорт (если MCP в том же процессе).
        Используется для тестирования или когда MCP запущен как библиотека.
        """
        try:
            from edms_mcp_server import (
                get_document, search_documents, get_document_history,
                get_document_versions, get_document_statistics,
                search_employees, get_current_user, create_task,
                create_introduction, execute_document_operation,
                start_document_routing, set_document_control,
                send_notification, get_reference_data, health_check,
            )

            tool_map = {
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
                "health_check": health_check,
            }

            func = tool_map.get(tool_name)
            if not func:
                return {"error": f"Инструмент '{tool_name}' не найден"}

            call_args = {**args, "token": token}
            raw_result = await func(**call_args)

            if isinstance(raw_result, str):
                try:
                    return json.loads(raw_result)
                except json.JSONDecodeError:
                    return {"result": raw_result}
            return raw_result if isinstance(raw_result, dict) else {"result": raw_result}

        except Exception as exc:
            logger.error("Direct tool call failed '%s': %s", tool_name, exc)
            return {"error": str(exc)}

    async def list_tools(self) -> list[dict]:
        """
        Получить список доступных инструментов с кэшированием.

        Returns:
            Список словарей с описанием инструментов
        """
        now = time.monotonic()
        if self._tools_cache and (now - self._tools_cache_at) < self._tools_cache_ttl:
            return self._tools_cache

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self.base_url}/tools")
                resp.raise_for_status()
                data = resp.json()
                tools = data.get("tools", data if isinstance(data, list) else [])
                self._tools_cache = tools
                self._tools_cache_at = now
                logger.info("Loaded %d tools from MCP server", len(tools))
                return tools
        except Exception as exc:
            logger.warning("Could not load tools from MCP server: %s", exc)
            # Возвращаем список из реестра если сервер недоступен
            return _get_default_tool_names()

    async def health(self) -> dict:
        """Проверить состояние MCP сервера."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.base_url}/health")
                return resp.json()
        except Exception as exc:
            return {"status": "unavailable", "error": str(exc)}


def _get_default_tool_names() -> list[dict]:
    """Возвращает дефолтный список инструментов если сервер недоступен."""
    return [
        {"name": n} for n in [
            "get_document", "search_documents", "get_document_history",
            "get_document_versions", "get_document_statistics",
            "search_employees", "get_current_user", "create_task",
            "create_introduction", "execute_document_operation",
            "start_document_routing", "set_document_control",
            "send_notification", "get_reference_data", "health_check",
        ]
    ]
