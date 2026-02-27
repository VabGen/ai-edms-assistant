# src/ai_edms_assistant/interfaces/api/routes/task_routes.py
"""Task query endpoints — thin HTTP layer over TaskRepository."""

from __future__ import annotations

from uuid import UUID
from fastapi import APIRouter, Query

from ..dependencies import TaskRepoDep
from ..schemas.document_schema import TaskBriefResponse

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get(
    "/document/{document_id}",
    response_model=list[TaskBriefResponse],
    summary="Поручения по документу",
)
async def tasks_by_document(
    document_id: UUID,
    token: str = Query(..., description="JWT токен пользователя"),
    repo: TaskRepoDep = None,
) -> list[TaskBriefResponse]:
    """Return all tasks linked to a specific document.

    Uses ``TaskBriefResponse.from_entity()`` for consistent mapping —
    ensures ``task_number``, ``responsible_executor_name``, ``is_endless``,
    ``executors_count`` are always populated correctly.

    Previously used a manual constructor with the deprecated
    ``responsible_name`` field — this caused a silent mapping error
    because the audit-fixed schema renamed it to ``responsible_executor_name``.

    Args:
        document_id: Parent document UUID.
        token: JWT bearer token (query param).
        repo: Injected AbstractTaskRepository.

    Returns:
        List of ``TaskBriefResponse`` for all tasks on the document.
    """
    tasks = await repo.get_by_document_id(document_id, token)
    return [TaskBriefResponse.from_entity(t) for t in tasks]


@router.get("/{task_id}/tree", summary="Полное дерево поручений")
async def task_tree(
    task_id: UUID,
    token: str = Query(..., description="JWT токен пользователя"),
    repo: TaskRepoDep = None,
) -> list[dict]:
    """Return the full task tree: root ancestor + all children + siblings.

    Used for rendering task hierarchy diagrams on the frontend.

    Args:
        task_id: Root task UUID.
        token: JWT bearer token.
        repo: Injected AbstractTaskRepository.

    Returns:
        Flat list of task dicts with ``id``, ``text``, ``status``, ``parentId``.
    """
    tasks = await repo.find_task_tree(task_id, token)
    return [
        {
            "id": str(t.id),
            "text": t.text,
            "status": t.status.value,
            "parentId": str(t.parent_task_id) if t.parent_task_id else None,
        }
        for t in tasks
    ]