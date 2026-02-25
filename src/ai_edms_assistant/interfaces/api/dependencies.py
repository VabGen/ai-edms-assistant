# src/ai_edms_assistant/interfaces/api/dependencies.py
"""
FastAPI DI container.

Singleton slots are populated by the lifespan manager in app.py.
Per-request dependencies (repositories) are created fresh each call
— they are stateless wrappers around the shared http_client.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated, Optional

from fastapi import Depends, HTTPException

# Отложенные импорты — предотвращают циклические зависимости при старте
from typing import TYPE_CHECKING

from ...application.agents import EdmsDocumentAgent

if TYPE_CHECKING:
    from ...infrastructure.edms_api.http_client import EdmsHttpClient
    from ...infrastructure.edms_api.repositories.edms_document_repository import (
        EdmsDocumentRepository,
    )
    from ...infrastructure.edms_api.repositories.edms_employee_repository import (
        EdmsEmployeeRepository,
    )
    from ...infrastructure.edms_api.repositories.edms_task_repository import (
        EdmsTaskRepository,
    )

# Shared upload directory (используется в agent_routes и app lifespan)
UPLOAD_DIR = Path(tempfile.gettempdir()) / "edms_ai_assistant_uploads"

# ── Singleton slots ──────────────────────────────────────────────────────────
_agent: Optional["EdmsDocumentAgent"] = None
_http_client: Optional["EdmsHttpClient"] = None


def set_agent(agent: "EdmsDocumentAgent") -> None:
    """Register the agent singleton. Called once from lifespan startup."""
    global _agent
    _agent = agent


def set_http_client(client: "EdmsHttpClient") -> None:
    """Register the HTTP client singleton. Called once from lifespan startup."""
    global _http_client
    _http_client = client


# ── Dependency providers ─────────────────────────────────────────────────────


def get_agent() -> "EdmsDocumentAgent":
    """
    Provide the singleton EdmsDocumentAgent.

    Raises:
        HTTPException 503: Agent failed to initialise at startup.
    """
    if _agent is None:
        raise HTTPException(status_code=503, detail="ИИ-Агент не инициализирован.")
    return _agent


def get_http_client() -> "EdmsHttpClient":
    """
    Provide the shared EdmsHttpClient.

    Raises:
        HTTPException 503: Client was not created at startup.
    """
    if _http_client is None:
        raise HTTPException(status_code=503, detail="HTTP-клиент не инициализирован.")
    return _http_client


def get_document_repository(
    client: Annotated["EdmsHttpClient", Depends(get_http_client)],
) -> "EdmsDocumentRepository":
    """Provide EdmsDocumentRepository. Stateless — safe to construct per-request."""
    from ...infrastructure.edms_api.repositories.edms_document_repository import (
        EdmsDocumentRepository,
    )

    return EdmsDocumentRepository(client)


def get_employee_repository(
    client: Annotated["EdmsHttpClient", Depends(get_http_client)],
) -> "EdmsEmployeeRepository":
    """Provide EdmsEmployeeRepository per-request."""
    from ...infrastructure.edms_api.repositories.edms_employee_repository import (
        EdmsEmployeeRepository,
    )

    return EdmsEmployeeRepository(client)


def get_task_repository(
    client: Annotated["EdmsHttpClient", Depends(get_http_client)],
) -> "EdmsTaskRepository":
    """Provide EdmsTaskRepository per-request."""
    from ...infrastructure.edms_api.repositories.edms_task_repository import (
        EdmsTaskRepository,
    )

    return EdmsTaskRepository(client)


# ── Annotated type aliases (удобные сокращения для подписей роутов) ──────────
AgentDep = Annotated["EdmsDocumentAgent", Depends(get_agent)]
DocumentRepoDep = Annotated["EdmsDocumentRepository", Depends(get_document_repository)]
EmployeeRepoDep = Annotated["EdmsEmployeeRepository", Depends(get_employee_repository)]
TaskRepoDep = Annotated["EdmsTaskRepository", Depends(get_task_repository)]
