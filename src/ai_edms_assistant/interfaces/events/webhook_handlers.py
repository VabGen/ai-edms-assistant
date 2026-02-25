# src/ai_edms_assistant/interfaces/events/webhook_handlers.py
"""
Webhook event handlers for EDMS backend push notifications.

The EDMS backend can POST events (document status change, task completion,
new appeal) to this service.  Each handler validates the payload, logs
the event, and can trigger downstream agent actions.

Mount in app.py::

    from .events.webhook_handlers import webhook_router
    app.include_router(webhook_router)

Security note:
    In production enable HMAC-SHA256 signature validation.
    Set WEBHOOK_SECRET in .env and add a Depends(verify_signature) on each route.
"""

from __future__ import annotations

import structlog
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = structlog.get_logger(__name__)

webhook_router = APIRouter(prefix="/webhooks", tags=["webhooks"])


class WebhookPayload(BaseModel):
    """Common envelope for all EDMS webhook events."""

    event_type: str
    organization_id: str
    entity_id: UUID
    data: dict[str, Any] = {}


@webhook_router.post(
    "/document/status-changed",
    summary="Документ изменил статус",
)
async def on_document_status_changed(payload: WebhookPayload) -> dict:
    """
    Handles EDMS DOCUMENT_STATUS_CHANGED event.

    Potential downstream actions:
    - Notify responsible executor
    - Trigger auto-control check
    - Write to audit log
    """
    logger.info(
        "webhook_document_status_changed",
        entity_id=str(payload.entity_id),
        new_status=payload.data.get("newStatus"),
        org_id=payload.organization_id,
    )
    # TODO: integrate with agent notification workflow
    return {"received": True, "event": payload.event_type}


@webhook_router.post(
    "/task/completed",
    summary="Поручение выполнено",
)
async def on_task_completed(payload: WebhookPayload) -> dict:
    """
    Handles EDMS TASK_COMPLETED event.

    Potential downstream actions:
    - Check if all sibling tasks on document are done
    - Send completion summary to document author
    """
    logger.info(
        "webhook_task_completed",
        task_id=str(payload.entity_id),
        org_id=payload.organization_id,
    )
    return {"received": True, "event": payload.event_type}


@webhook_router.post(
    "/appeal/received",
    summary="Получено новое обращение",
)
async def on_appeal_received(payload: WebhookPayload) -> dict:
    """
    Handles EDMS APPEAL_RECEIVED event.

    Potential downstream actions:
    - Trigger auto-classification via agent
    - Suggest routing to responsible department
    """
    logger.info(
        "webhook_appeal_received",
        appeal_id=str(payload.entity_id),
        org_id=payload.organization_id,
    )
    return {"received": True, "event": payload.event_type}
