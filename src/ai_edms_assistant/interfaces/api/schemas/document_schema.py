# src/ai_edms_assistant/interfaces/api/schemas/document_schema.py
"""Pydantic response schemas for document and task endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel


class DocumentBriefResponse(BaseModel):
    """Краткое представление документа для списков."""

    id: UUID
    reg_number: Optional[str] = None
    short_summary: Optional[str] = None
    status: Optional[str] = None
    create_date: Optional[datetime] = None
    author_name: Optional[str] = None


class TaskBriefResponse(BaseModel):
    """Краткое представление поручения."""

    id: UUID
    text: str
    status: str
    deadline: Optional[datetime] = None
    responsible_name: Optional[str] = None
