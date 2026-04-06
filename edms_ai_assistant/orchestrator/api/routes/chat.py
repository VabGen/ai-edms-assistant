# orchestrator/api/routes/chat.py
"""
Chat-роутер. Подключается в main.py через app.include_router().
Дублирует эндпоинты из main.py для тех, кто предпочитает роутеры.
"""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["Chat v1"])
# Эндпоинты определены непосредственно в main.py для простоты.
# Этот модуль — точка расширения для дополнительных v1-маршрутов.
