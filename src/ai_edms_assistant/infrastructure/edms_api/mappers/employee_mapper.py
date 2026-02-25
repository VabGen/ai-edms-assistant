# src/ai_edms_assistant/infrastructure/edms_api/mappers/employee_mapper.py
"""EmployeeDto dict → domain Employee mapper.

Maps EDMS API employee representations to domain entities with complete
field coverage from Java DTO specification.

Architecture:
    Infrastructure Layer → Domain Layer
    Raw API dict → Immutable/Mutable domain entity
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from ....domain.entities.employee import Employee, EmployeeCreateType, UserInfo

logger = logging.getLogger(__name__)


class EmployeeMapper:
    """Stateless mapper: EDMS EmployeeDto dict → domain Employee/UserInfo.

    Handles two response shape variants:
    1. **Flat** — Deprecated GET /api/employee
       Fields at root: departmentName, postName, departmentId, postId

    2. **Nested** — POST /api/employee/search with includes=[POST, DEPARTMENT]
       Nested objects: department: {id, name}, post: {id, postName}

    Critical field mapping:
        uId (primary) / getuId / getUId → account_id
        departmentName / department.name → department_name
        postName / post.name / post.postName → post_name
    """

    @staticmethod
    def from_dto(data: dict[str, Any]) -> Employee:
        """Map a single EmployeeDto dict to domain Employee.

        Args:
            data: Raw dict from EDMS API in either flat or nested format.

        Returns:
            Populated domain Employee entity.

        Raises:
            KeyError: When mandatory ``id`` field is missing.
        """
        # ── Extract nested objects (if present) ───────────────────────────────
        post_raw: dict = data.get("post") or {}
        dept_raw: dict = data.get("department") or {}
        org_raw: dict = data.get("org") or {}

        # ── Resolve IDs with fallback to root-level fields ───────────────────
        dept_id_raw = dept_raw.get("id") or data.get("departmentId")
        post_id_raw = post_raw.get("id") or data.get("postId")

        # ── Parse department_id (handle both UUID and None) ───────────────────
        department_id: UUID | None = None
        if dept_id_raw:
            try:
                department_id = UUID(str(dept_id_raw))
            except (ValueError, TypeError):
                logger.debug(
                    "invalid_department_id",
                    extra={"value": dept_id_raw, "employee_id": data.get("id")},
                )

        # ── Parse create_date ─────────────────────────────────────────────────
        create_date = EmployeeMapper._parse_datetime(data.get("createDate"))
        last_avatar_upload = EmployeeMapper._parse_datetime(
            data.get("lastManualAvatarUploadDate")
        )

        # ── Parse create_type enum ────────────────────────────────────────────
        create_type_raw = data.get("createType")
        create_type: EmployeeCreateType | None = None
        if create_type_raw:
            try:
                create_type = EmployeeCreateType(create_type_raw)
            except ValueError:
                logger.warning(
                    "unknown_employee_create_type",
                    extra={"value": create_type_raw, "employee_id": data.get("id")},
                )

        # ── Parse organization_id (string, not UUID) ──────────────────────────
        organization_id = (
            org_raw.get("id") or data.get("organizationId") or data.get("orgId")
        )

        return Employee(
            # ── Identity ──────────────────────────────────────────────────────
            id=UUID(data["id"]) if data.get("id") else None,
            organization_id=organization_id,
            # ── Account identifiers ───────────────────────────────────────────
            # CRITICAL: uId is the PRIMARY field in Java, getuId is secondary
            account_id=(data.get("uId") or data.get("getuId") or data.get("getUId")),
            external_id=data.get("externalId"),
            personal_number=data.get("personalNumber"),
            ldap_name=data.get("ldapName"),
            sid=data.get("sid"),  # Windows SID for AD authentication
            # ── Name components ───────────────────────────────────────────────
            first_name=data.get("firstName"),
            last_name=data.get("lastName"),
            middle_name=data.get("middleName"),
            full_post_name=data.get("fullPostName"),  # Full "ФИО + должность"
            # ── Department (nested or flat) ───────────────────────────────────
            department_id=department_id,
            department_name=(
                dept_raw.get("name")
                or dept_raw.get("departmentName")
                or data.get("departmentName")
            ),
            department_code=dept_raw.get("departmentCode")
            or data.get("departmentCode"),
            # ── Post/Position (nested or flat) ────────────────────────────────
            post_id=str(post_id_raw) if post_id_raw else None,
            post_name=(
                post_raw.get("postName") or post_raw.get("name") or data.get("postName")
            ),
            # ── Contact information ───────────────────────────────────────────
            email=data.get("email"),
            phone=data.get("phone") or data.get("workPhone"),
            address=data.get("address"),
            place=data.get("place"),  # Площадка (office location)
            url=data.get("url"),
            # ── Employment status ─────────────────────────────────────────────
            is_active=data.get("active", True),
            fired=data.get("fired", False),
            # ── Acting position (ИО) ──────────────────────────────────────────
            is_acting=data.get("io", False),  # Является ИО
            has_acting=data.get("haveIo", False),  # Присутствуют ИО
            # ── Notification settings ─────────────────────────────────────────
            notify=data.get("notify", True),  # Email notifications enabled
            # ── Metadata ──────────────────────────────────────────────────────
            create_type=create_type,
            create_date=create_date,
            last_manual_avatar_upload_date=last_avatar_upload,
            current_user_leader=data.get("currentUserLeader"),
            order=data.get("order", 0),  # Sorting order
            # ── Asset paths ───────────────────────────────────────────────────
            photo_path=data.get("photoPath"),
            facsimile_path=data.get("facsimilePath"),
            # ── Avatar IDs (for internal asset management) ───────────────────
            avatar_id=data.get("avatarId"),
            small_avatar_id=data.get("smallAvatarId"),
            facsimile_id=data.get("facsimileId"),
            # ── Blocked fields (list of field names that cannot be edited) ────
            blocked_fields=data.get("blockedFields"),
        )

    @staticmethod
    def to_user_info(data: dict[str, Any] | None) -> UserInfo | None:
        """Map nested employee reference to lightweight UserInfo.

        Used when DocumentDto/TaskDto embeds partial employee data under
        keys like ``author``, ``responsibleExecutor``, ``initiator``, etc.

        Handles both UserInfoDto (from Java) and partial EmployeeDto shapes.

        Args:
            data: Partial employee/user dict or None.

        Returns:
            UserInfo value object or None when data is absent/invalid.
        """
        if not data:
            return None

        # ── Parse employee/user ID ────────────────────────────────────────────
        # UserInfoDto has "employeeId", EmployeeDto has "id"
        emp_id = data.get("employeeId") or data.get("id")
        user_id: UUID | None = None
        if emp_id:
            try:
                user_id = UUID(str(emp_id))
            except (ValueError, TypeError):
                pass

        # ── Parse department ID ───────────────────────────────────────────────
        dept_id_raw = data.get("authorDepartmentId") or data.get(  # UserInfoDto
            "departmentId"
        )  # EmployeeDto
        department_id: UUID | None = None
        if dept_id_raw:
            try:
                department_id = UUID(str(dept_id_raw))
            except (ValueError, TypeError):
                pass

        # ── Assemble full name from parts ─────────────────────────────────────
        # Priority: pre-assembled "name" > constructed from parts
        name = data.get("name") or data.get("fullName")
        if not name:
            # Construct from first/last/middle
            parts = [
                data.get("lastName"),
                data.get("firstName"),
                data.get("middleName"),
            ]
            name = " ".join(filter(None, parts)) or "Не указано"

        return UserInfo(
            id=user_id,
            name=name,
            organization_id=data.get("organizationId") or data.get("employeeOrgId"),
            # ── Department info ───────────────────────────────────────────────
            department_id=department_id,
            department_name=(
                data.get("authorDepartmentName")  # UserInfoDto
                or data.get("departmentName")  # EmployeeDto
            ),
            # ── Post/Position ─────────────────────────────────────────────────
            post_name=(
                data.get("authorPost")  # UserInfoDto
                or data.get("postName")  # EmployeeDto
            ),
            # ── Contact ───────────────────────────────────────────────────────
            email=data.get("email"),
        )

    @staticmethod
    def from_dto_list(items: list[dict[str, Any]]) -> list[Employee]:
        """Map a list of EmployeeDto dicts, skipping malformed items.

        Args:
            items: List of employee dicts from API response.

        Returns:
            List of successfully mapped domain Employee entities.
            Logs debug messages for skipped items.
        """
        result: list[Employee] = []
        for item in items or []:
            try:
                result.append(EmployeeMapper.from_dto(item))
            except (KeyError, ValueError) as exc:
                logger.debug(
                    "employee_mapper_skip",
                    extra={
                        "error": str(exc),
                        "item_id": item.get("id") if isinstance(item, dict) else None,
                    },
                )
        return result

    @staticmethod
    def _parse_datetime(raw: str | int | float | None) -> datetime | None:
        """Parse datetime from ISO string or Java timestamp (milliseconds).

        Args:
            raw: ISO 8601 string, Unix timestamp (ms), or None.

        Returns:
            Parsed datetime or None if invalid/absent.
        """
        if not raw:
            return None

        try:
            if isinstance(raw, str):
                return datetime.fromisoformat(raw.replace("Z", "+00:00"))
            elif isinstance(raw, (int, float)):
                return datetime.fromtimestamp(raw / 1000)
        except (ValueError, TypeError, OSError) as exc:
            logger.debug("datetime_parse_failed", extra={"raw": raw, "error": str(exc)})

        return None
