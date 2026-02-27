# src/ai_edms_assistant/infrastructure/edms_api/mappers/employee_mapper.py
"""EmployeeDto dict → domain Employee / UserInfo mapper.

Responsibilities:
    1. Normalize Java type quirks (int→str, null→bool, nested→flat).
    2. Map normalized raw dict to domain Employee / UserInfo entities.

Architecture:
    Infrastructure Layer → Domain Layer
    Raw API dict → Pydantic domain entity

Java API confirmed quirks (Postman 2026-02-26):
    - ``postId: 23543``         → int, Employee.post_id is str | None
    - ``notify: null``          → None, Employee.notify is bool = True
    - ``post: {postName: ...}`` → nested object, Employee.post_name is flat str
    - ``department: {name: ..}`` → nested, Employee.department_name is flat str
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from ....domain.entities.employee import Employee, UserInfo

logger = logging.getLogger(__name__)

# ── Поля с гарантированным int → str преобразованием ─────────────────────────
_INT_TO_STR_FIELDS = (
    "postId",
    "externalId",
    "personalNumber",
    "ldapName",
    "sid",
    "uId",
    "getuId",
)

# ── bool поля с null-default ─────────────────────────────────────────
_BOOL_DEFAULTS: tuple[tuple[str, bool], ...] = (
    ("notify", True),
    ("fired", False),
    ("active", True),
    ("io", False),
    ("haveIo", False),
)


class EmployeeMapper:
    """Stateless mapper: EDMS EmployeeDto raw dict → domain entities.

    All methods are static — no state, safe for concurrent use.

    Normalization handles two Java response shapes:
        Flat  : ``postName``, ``departmentName`` at root level.
        Nested: ``post: {postName: ...}``, ``department: {name: ...}`` objects.
                Returned by GET /api/employee/{id}.
                fts-lastname always returns ``post=null``, ``department=null``.
    """

    @staticmethod
    def normalize(data: dict[str, Any]) -> dict[str, Any]:
        """Normalize raw Java API dict before Pydantic model_validate().

        Operates on a COPY of ``data`` — original is never mutated.
        This is the single source of truth for Java → Python type coercion.

        Fixes applied:
            1. ``int → str``: postId and other identifier fields.
            2. ``None → bool``: notify, fired, active, io, haveIo.
            3. ``post.postName → postName``: flat extraction from nested post.
            4. ``department.name → departmentName``: flat extraction from nested dept.

        Args:
            data: Raw dict from Java API response (not yet copied).

        Returns:
            New normalized dict safe for ``Employee.model_validate()``.
        """
        result = dict(data)

        # ── 1. int → str ──────────────────────────────────────────────────────
        for field in _INT_TO_STR_FIELDS:
            val = result.get(field)
            if isinstance(val, int):
                result[field] = str(val)

        # ── 2. None → bool defaults ───────────────────────────────────────────
        for field, default in _BOOL_DEFAULTS:
            if result.get(field) is None:
                result[field] = default

        # ── 3. Nested post → flat postName ────────────────────────────────────
        post_raw = result.get("post")
        if isinstance(post_raw, dict):
            if not result.get("postName"):
                result["postName"] = post_raw.get("postName") or post_raw.get("name")
            if not result.get("postId"):
                pid = post_raw.get("id")
                if pid is not None:
                    result["postId"] = str(pid) if isinstance(pid, int) else pid

        # ── 4. Nested department → flat departmentName ────────────────────────
        dept_raw = result.get("department")
        if isinstance(dept_raw, dict):
            if not result.get("departmentName"):
                result["departmentName"] = dept_raw.get("name") or dept_raw.get(
                    "departmentName"
                )
            if not result.get("departmentId"):
                did = dept_raw.get("id")
                if did:
                    result["departmentId"] = str(did)

        return result

    @staticmethod
    def to_employee(data: dict[str, Any] | None) -> Employee | None:
        """Map a single normalized EmployeeDto dict to domain Employee.

        Args:
            data: Raw dict from EDMS API (will be normalized internally).

        Returns:
            Domain Employee entity or None on mapping failure or empty input.
        """
        if not data or not isinstance(data, dict):
            return None

        try:
            return Employee.model_validate(EmployeeMapper.normalize(data))
        except Exception as exc:
            logger.warning(
                "employee_mapper_to_employee_failed",
                error=str(exc),
                employee_id=data.get("id"),
                keys=list(data.keys())[:10],
            )
            return None

    @staticmethod
    def to_employee_list(items: list[dict[str, Any]]) -> list[Employee]:
        """Map a list of EmployeeDto dicts, skipping malformed items.

        Args:
            items: List of raw employee dicts from Spring ``content`` array.

        Returns:
            List of successfully mapped Employee entities.
            Malformed items are skipped and logged at DEBUG level.
        """
        result: list[Employee] = []
        for item in items or []:
            employee = EmployeeMapper.to_employee(item)
            if employee is not None:
                result.append(employee)
        return result

    @staticmethod
    def to_user_info(data: dict[str, Any] | None) -> UserInfo | None:
        """Map a partial employee/user reference to lightweight UserInfo.

        Used when Document/Task DTOs embed partial employee data under keys
        like ``author``, ``responsibleExecutor``, ``initiator``.

        Handles both ``UserInfoDto`` (``employeeId`` key) and partial
        ``EmployeeDto`` (``id`` key) shapes.

        Args:
            data: Partial employee or user-info dict, or None.

        Returns:
            UserInfo value object or None when input is absent or invalid.
        """
        if not data or not isinstance(data, dict):
            return None

        # ── Resolve employee UUID (UserInfoDto vs EmployeeDto shape) ──────────
        raw_id = data.get("employeeId") or data.get("id")
        user_id: UUID | None = None
        if raw_id:
            try:
                user_id = UUID(str(raw_id))
            except (ValueError, TypeError):
                pass

        # ── Resolve department UUID ───────────────────────────────────────────
        raw_dept_id = data.get("authorDepartmentId") or data.get("departmentId")
        department_id: UUID | None = None
        if raw_dept_id:
            try:
                department_id = UUID(str(raw_dept_id))
            except (ValueError, TypeError):
                pass

        try:
            return UserInfo.model_validate(
                {
                    "id": user_id,
                    "firstName": data.get("firstName"),
                    "lastName": data.get("lastName"),
                    "middleName": data.get("middleName"),
                    "organizationId": (
                        data.get("organizationId") or data.get("employeeOrgId")
                    ),
                    "departmentId": department_id,
                    "departmentName": (
                        data.get("authorDepartmentName")  # UserInfoDto key
                        or data.get("departmentName")  # EmployeeDto key
                    ),
                    "postName": (
                        data.get("authorPost")  # UserInfoDto key
                        or data.get("postName")  # EmployeeDto key
                    ),
                    "email": data.get("email"),
                }
            )
        except Exception as exc:
            logger.warning(
                "employee_mapper_to_user_info_failed",
                error=str(exc),
                raw_id=raw_id,
            )
            return None
