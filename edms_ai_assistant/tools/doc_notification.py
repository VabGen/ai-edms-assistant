# edms_ai_assistant/tools/doc_notification.py
"""
EDMS AI Assistant — Notification & Reminder Tool.

Слой: Infrastructure / Tool.
Отправляет уведомления и напоминания сотрудникам через EDMS API.

Поддерживаемые типы уведомлений:
  - REMINDER   — ручное напоминание о документе / дедлайне
  - DEADLINE   — предупреждение о приближающемся сроке поручения
  - CUSTOM     — произвольное сообщение по документу
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from edms_ai_assistant.clients.base_client import EdmsHttpClient

logger = logging.getLogger(__name__)


class NotificationType(StrEnum):
    """Types of notifications supported by EDMS."""

    REMINDER = "REMINDER"
    DEADLINE = "DEADLINE"
    CUSTOM = "CUSTOM"


class _NotificationClient(EdmsHttpClient):
    """Minimal EDMS client for notification endpoints."""

    async def send_notification(
        self,
        token: str,
        document_id: str,
        recipient_ids: list[str],
        notification_type: str,
        message: str,
        deadline: str | None,
    ) -> dict[str, Any]:
        """
        Sends a notification to the given employees.

        POST api/document/{document_id}/notification
        """
        payload: dict[str, Any] = {
            "recipientIds": recipient_ids,
            "type": notification_type,
            "message": message.strip(),
        }
        if deadline:
            payload["deadline"] = deadline

        result = await self._make_request(
            "POST",
            f"api/document/{document_id}/notification",
            token=token,
            json=payload,
            is_json_response=False,
        )
        # Некоторые EDMS-эндпоинты возвращают пустой 204 — нормализуем.
        return result if isinstance(result, dict) else {"sent": True}


# ──────────────────────────────────────────────────────────────────────────────
# Схема ввода
# ──────────────────────────────────────────────────────────────────────────────


class SendNotificationInput(BaseModel):
    """Validated input schema for sending notifications."""

    token: str = Field(..., description="JWT токен авторизации пользователя")
    document_id: str = Field(
        ..., description="UUID документа, к которому привязано уведомление"
    )
    recipient_ids: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Список UUID сотрудников-получателей уведомления. "
            "Используй employee_search_tool для получения UUID."
        ),
    )
    message: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Текст уведомления / напоминания",
    )
    notification_type: NotificationType = Field(
        NotificationType.REMINDER,
        description=(
            "Тип уведомления: "
            "REMINDER — напоминание, "
            "DEADLINE — предупреждение о сроке, "
            "CUSTOM — произвольное"
        ),
    )
    deadline: str | None = Field(
        None,
        description=(
            "Дедлайн, о котором напоминаем, в формате ISO 8601 "
            "(например: '2026-04-01T18:00:00Z')"
        ),
    )

    @field_validator("recipient_ids")
    @classmethod
    def validate_recipient_ids(cls, v: list[str]) -> list[str]:
        """Filters out empty/blank recipient IDs."""
        cleaned = [uid.strip() for uid in v if uid and uid.strip()]
        if not cleaned:
            raise ValueError("Список получателей не может быть пустым")
        return cleaned

    @field_validator("message")
    @classmethod
    def strip_message(cls, v: str) -> str:
        """Strips surrounding whitespace from message text."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Текст уведомления не может быть пустым")
        return stripped


# ──────────────────────────────────────────────────────────────────────────────
# Tool
# ──────────────────────────────────────────────────────────────────────────────


@tool("doc_send_notification", args_schema=SendNotificationInput)
async def doc_send_notification(
    token: str,
    document_id: str,
    recipient_ids: list[str],
    message: str,
    notification_type: NotificationType = NotificationType.REMINDER,
    deadline: str | None = None,
) -> dict[str, Any]:
    """
    Отправляет уведомление или напоминание сотрудникам по документу.

    Используй когда пользователь просит:
    - «Напомни Иванову о документе»
    - «Отправь уведомление исполнителям о дедлайне 15 апреля»
    - «Предупреди Петрова о приближающемся сроке»
    - «Сообщи всем исполнителям поручения что срок переносится»

    ВАЖНО: перед вызовом используй employee_search_tool для получения UUID
    сотрудников-получателей, если они не известны.

    Возвращает статус отправки и список адресатов.
    """
    logger.info(
        "Sending notification",
        extra={
            "document_id": document_id,
            "recipients_count": len(recipient_ids),
            "notification_type": notification_type,
            "has_deadline": bool(deadline),
        },
    )

    try:
        async with _NotificationClient() as client:
            result = await client.send_notification(
                token=token,
                document_id=document_id,
                recipient_ids=recipient_ids,
                notification_type=str(notification_type),
                message=message,
                deadline=deadline,
            )

        recipients_count = len(recipient_ids)
        deadline_note = f" (дедлайн: {deadline[:10]})" if deadline else ""

        logger.info(
            "Notification sent successfully",
            extra={
                "document_id": document_id,
                "recipients": recipients_count,
            },
        )

        return {
            "status": "success",
            "message": (
                f"Уведомление отправлено {recipients_count} "
                f"сотруднику(-ам){deadline_note}."
            ),
            "notification_type": str(notification_type),
            "recipients_count": recipients_count,
            "document_id": document_id,
            "result": result,
        }

    except Exception as exc:
        logger.error("Failed to send notification", exc_info=True)
        return {
            "status": "error",
            "message": f"Ошибка отправки уведомления: {exc}",
        }
