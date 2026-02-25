# src/ai_edms_assistant/application/dto/__init__.py
"""Data Transfer Objects for application layer.

DTOs provide a stable interface between application use cases and
the external world (HTTP endpoints, CLI, etc). They are transport-agnostic
Pydantic models that validate input and serialize output.

Document DTOs:
    DocumentSummaryDto: Lightweight document summary for use case outputs.
    TaskSummaryDto: Lightweight task summary for use case outputs.
    ComparisonResultDto: Document comparison results with changed fields.
    ExtractionResultDto: NLP extraction results with confidence scores.

Agent DTOs:
    AgentRequest: Validated incoming user request to agent.
    AgentResponse: Standardized agent response with status and content.
    AgentStatus: Status enum for agent execution (SUCCESS, ERROR, etc.).
    ActionType: Type of user action required (DISAMBIGUATION, etc.).
"""

from .agent import ActionType, AgentRequest, AgentResponse, AgentStatus
from .document_dto import (
    ComparisonResultDto,
    DocumentSummaryDto,
    ExtractionResultDto,
    TaskSummaryDto,
)

__all__ = [
    "DocumentSummaryDto",
    "TaskSummaryDto",
    "ComparisonResultDto",
    "ExtractionResultDto",
    "AgentRequest",
    "AgentResponse",
    "AgentStatus",
    "ActionType",
]
