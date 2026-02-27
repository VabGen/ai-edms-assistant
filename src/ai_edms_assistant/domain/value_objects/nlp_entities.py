# src/ai_edms_assistant/domain/value_objects/nlp_entities.py
"""NLP named entity value objects for the domain layer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class EntityType(StrEnum):
    """Types of extractable named entities from EDMS document text."""

    DATE = "date"
    DATETIME = "datetime"
    PERSON = "person"
    NUMBER = "number"
    MONEY = "money"
    DOCUMENT_ID = "document_id"
    DEPARTMENT = "department"
    DURATION = "duration"


@dataclass
class Entity:
    """Immutable named entity value object.

    Attributes:
        type: EntityType classification.
        value: Parsed Python value (datetime, float, dict, str).
        raw_text: Original text fragment that was matched.
        confidence: Confidence score 0.0–1.0.
        normalized_value: Normalized representation (ISO string, dict, etc.).
    """

    type: EntityType
    value: Any
    raw_text: str
    confidence: float = 1.0
    normalized_value: Any | None = None
