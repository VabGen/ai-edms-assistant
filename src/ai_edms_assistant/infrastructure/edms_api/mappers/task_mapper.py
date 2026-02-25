# src/ai_edms_assistant/infrastructure/edms_api/mappers/task_mapper.py
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from ....domain.entities.employee import UserInfo
from ....domain.entities.task import Task, TaskStatus, TaskType
from .employee_mapper import EmployeeMapper

logger = logging.getLogger(__name__)


class TaskMapper:
    """FULL mapper: EDMS TaskDto → domain Task (45+ fields)."""

    @staticmethod
    def from_dto(data: dict[str, Any]) -> Task:
        """Map complete TaskDto to domain Task."""

        # ── Parse status enum ──────────────────────────────────────────
        raw_status = data.get("taskStatus") or data.get("status")
        status = TaskStatus.NEW
        if raw_status:
            try:
                status = TaskStatus(raw_status)
            except ValueError:
                logger.warning(
                    f"unknown_task_status: {raw_status}, task_id: {data.get('id')}"
                )

        # ── Parse task type enum ───────────────────────────────────────
        raw_type = data.get("type") or data.get("taskType")
        task_type: TaskType | None = None
        if raw_type:
            try:
                task_type = TaskType(raw_type)
            except ValueError:
                logger.warning(
                    f"unknown_task_type: {raw_type}, task_id: {data.get('id')}"
                )
                task_type = TaskType.EXECUTION

        # ── Parse executors ────────────────────────────────────────────
        executors: list[UserInfo] = []
        for ex in data.get("executors", []):
            emp_raw = ex.get("executor") or ex
            u = EmployeeMapper.to_user_info(emp_raw)
            if u:
                executors.append(u)

        # ── Parse responsible executor ─────────────────────────────────
        responsible_executor = EmployeeMapper.to_user_info(
            data.get("responsibleExecutor")
        )

        # ── Parse timestamps ───────────────────────────────────────────
        def _dt(raw: str | int | float | None) -> datetime | None:
            if not raw:
                return None
            try:
                if isinstance(raw, str):
                    return datetime.fromisoformat(raw.replace("Z", "+00:00"))
                elif isinstance(raw, (int, float)):
                    return datetime.fromtimestamp(raw / 1000)
            except (ValueError, TypeError, OSError):
                return None

        # ── Parse IDs ──────────────────────────────────────────────────
        parent_raw = data.get("parentId") or data.get("parentTaskId")
        parent_task_id = None
        if parent_raw:
            try:
                parent_task_id = UUID(str(parent_raw))
            except (ValueError, TypeError):
                pass

        document_id = None
        if data.get("documentId"):
            try:
                document_id = UUID(data["documentId"])
            except (ValueError, TypeError):
                pass

        return Task(
            # ── Identity ───────────────────────────────────────────────
            id=UUID(data["id"]),
            task_number=data.get("taskNumber"),
            # ── Content ────────────────────────────────────────────────
            text=data.get("taskText") or data.get("text", ""),
            note=data.get("note"),
            # ── Status & Type ──────────────────────────────────────────
            status=status,
            task_type=task_type,
            # ── Deadlines ──────────────────────────────────────────────
            deadline=_dt(data.get("planedDateEnd") or data.get("deadline")),
            create_date=_dt(data.get("createDate")),
            complete_date=_dt(data.get("realDateEnd") or data.get("completeDate")),
            modify_date=_dt(data.get("modifyDate")),
            # ── Executors ──────────────────────────────────────────────
            author=EmployeeMapper.to_user_info(data.get("author")),
            responsible_executor=responsible_executor,
            executors=executors,
            # ── Relations ──────────────────────────────────────────────
            document_id=document_id,
            parent_task_id=parent_task_id,
            # ── Priority & Control ─────────────────────────────────────
            priority=data.get("priority", 0),
            is_controlled=data.get("onControl", False),
            # ── Periodic Task Fields ───────────────────────────────────
            endless=data.get("endless", False),
            periodic_task=data.get("periodTask", False),
            period_type=data.get("periodType"),
            period_days=data.get("periodDays"),
            period_day_week=data.get("periodDayWeek"),
            period_day_month=data.get("periodDayMonth"),
            period_month_year=data.get("periodMonthYear"),
            period_date_begin=_dt(data.get("periodDateBegin")),
            period_date_end=_dt(data.get("periodDateEnd")),
            # ── Counters ───────────────────────────────────────────────
            count_exec=data.get("countExec", 0),
            count_completed_exec=data.get("countCompletedExec", 0),
            # ── Flags ──────────────────────────────────────────────────
            revision=data.get("revision", False),
            can_change_date=data.get("canChangeDate", False),
            # ── Additional Info ────────────────────────────────────────
            days_execution=data.get("daysExecution"),
            organization_id=data.get("organizationId"),
        )

    @staticmethod
    def from_dto_list(items: list[dict[str, Any]]) -> list[Task]:
        """Map list of TaskDto dicts, skip malformed."""
        result: list[Task] = []
        for item in items or []:
            try:
                result.append(TaskMapper.from_dto(item))
            except (KeyError, ValueError) as exc:
                logger.warning(f"task_mapper_skip: {exc}, item_id: {item.get('id')}")
        return result
