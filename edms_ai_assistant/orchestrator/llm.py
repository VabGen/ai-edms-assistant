"""llm.py — LLM клиент с поддержкой лёгкой и тяжёлой моделей."""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger("llm")

_LIGHT_URL = os.getenv("LLM_LIGHT_URL", "http://localhost:11434/v1")
_LIGHT_MODEL = os.getenv("LLM_LIGHT_MODEL", "llama3.2:3b")
_HEAVY_URL = os.getenv("LLM_HEAVY_URL", "http://localhost:11434/v1")
_HEAVY_MODEL = os.getenv("LLM_HEAVY_MODEL", "gpt-oss:120b-cloud")
_API_KEY = os.getenv("LLM_API_KEY", "ollama")
_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))
_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))


async def call_llm(
    messages: list[dict[str, Any]],
    *,
    use_heavy: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
    tools: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Вызвать LLM (лёгкую или тяжёлую модель).

    Returns:
        {"content": str, "tool_calls": list | None, "model": str, "usage": dict}
    """
    base_url = _HEAVY_URL if use_heavy else _LIGHT_URL
    model = _HEAVY_MODEL if use_heavy else _LIGHT_MODEL
    url = f"{base_url.rstrip('/')}/chat/completions"

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature if temperature is not None else _TEMPERATURE,
        "max_tokens": max_tokens or _MAX_TOKENS,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_API_KEY}",
    }

    import asyncio
    delay = 1.0
    for attempt in range(_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            msg = choice["message"]
            return {
                "content": msg.get("content") or "",
                "tool_calls": msg.get("tool_calls"),
                "model": model,
                "usage": data.get("usage", {}),
                "finish_reason": choice.get("finish_reason"),
            }
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(delay)
                delay *= 2
            else:
                raise RuntimeError(f"LLM недоступен после {_MAX_RETRIES} попыток: {exc}") from exc
        except Exception as exc:
            logger.error("LLM error: %s", exc, exc_info=True)
            raise
    raise RuntimeError("LLM: превышено количество попыток")
