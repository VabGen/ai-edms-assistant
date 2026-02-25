# src/ai_edms_assistant/domain/entities/employee.py
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field, computed_field

from .base import DomainModel, MutableDomainModel


class EmployeeCreateType(StrEnum):
    MANUAL = "MANUAL"
    LDAP = "LDAP"
    EXTERNAL = "EXTERNAL"


class UserInfo(DomainModel):
    """Lightweight immutable user reference used in document relationships.

    Maps to ``UserInfoDto`` from the Java backend. Designed as a *value object*
    — instances are embedded inside ``Document``, ``Task`` and other aggregates
    and must not be mutated after construction.

    The three-part name (``last_name``, ``first_name``, ``middle_name``)
    mirrors the actual API response structure. Computed properties assemble
    display strings for LLM context injection.

    Attributes:
        id: Employee UUID. Optional because some API responses return user
            info without an explicit ID (e.g. ``whoSigned`` on historical docs).
        first_name: First name (Имя).
        last_name: Last name / surname (Фамилия).
        middle_name: Patronymic (Отчество).
        organization_id: Organization identifier string.
        department_id: UUID of the employee's department.
        department_name: Human-readable department name.
        post_name: Job title (Должность).
        email: Corporate e-mail address.
    """

    id: UUID | None = None
    first_name: str | None = Field(default=None, alias="firstName")
    last_name: str | None = Field(default=None, alias="lastName")
    middle_name: str | None = Field(default=None, alias="middleName")

    organization_id: str | None = Field(default=None, alias="organizationId")
    department_id: UUID | None = Field(default=None, alias="departmentId")
    department_name: str | None = Field(default=None, alias="departmentName")
    post_name: str | None = Field(default=None, alias="postName")

    email: str | None = None

    @computed_field
    @property
    def name(self) -> str:
        """Full display name assembled from individual name parts.

        Returns:
            'Фамилия Имя Отчество'. Falls back to empty string when no
            name parts are set.

        Example:
            >>> UserInfo(last_name="Иванов", first_name="Иван").name
            'Иванов Иван'
        """
        parts = [p for p in [self.last_name, self.first_name, self.middle_name] if p]
        return " ".join(parts)

    @computed_field
    @property
    def short_name(self) -> str:
        """Abbreviated name in 'Фамилия И.О.' format.

        Used in document headers and LLM context summaries where space is
        limited.

        Returns:
            Abbreviated name string, or full name if last_name is absent.

        Example:
            >>> UserInfo(last_name="Иванов", first_name="Иван", middle_name="Иванович").short_name
            'Иванов И.И.'
        """
        if not self.last_name:
            return self.name
        initials = "".join(f"{n[0]}." for n in [self.first_name, self.middle_name] if n)
        return f"{self.last_name} {initials}".strip()

    def __str__(self) -> str:
        return self.name or str(self.id)


class Employee(MutableDomainModel):
    """Full employee entity for organizational structure operations.

    Unlike the lightweight ``UserInfo`` value object, ``Employee`` is a
    mutable aggregate that represents a full record from the EDMS employee
    directory. Used in employee search, introduction list creation, and
    task assignment workflows.

    Attributes:
        id: Primary UUID — mandatory for a full employee record.
        organization_id: Multi-tenant organization identifier.
        first_name: First name (Имя).
        last_name: Last name / surname (Фамилия).
        middle_name: Patronymic (Отчество).
        department_id: UUID of the employee's department.
        department_name: Human-readable department name.
        department_code: Short department code from nomenclature.
        post_name: Job title (Должность).
        email: Corporate e-mail.
        phone: Contact phone number.
        address: Physical address.
        getu_id: External GETU system identifier.
        ln_address: Lotus Notes address (legacy integration).
        photo_path: Path to the employee photo in storage.
        facsimile_path: Path to the facsimile image in storage.
        is_active: Whether the employee account is active.
    """

    id: UUID
    organization_id: str = Field(alias="organizationId")

    # После строки: organization_id: str

    # Account & External IDs:
    account_id: str | None = Field(default=None, alias="uId")  # PRIMARY!
    external_id: str | None = Field(default=None, alias="externalId")
    personal_number: str | None = Field(default=None, alias="personalNumber")
    ldap_name: str | None = Field(default=None, alias="ldapName")
    sid: str | None = None

    # Post/Position:
    post_id: str | None = Field(default=None, alias="postId")
    full_post_name: str | None = Field(default=None, alias="fullPostName")

    # Additional Info:
    place: str | None = None
    url: str | None = None
    fired: bool = False

    # Acting Position (И.О.):
    is_acting: bool = Field(default=False, alias="io")
    has_acting: bool = Field(default=False, alias="haveIo")

    # Notifications:
    notify: bool = True

    # Metadata:
    create_type: EmployeeCreateType | None = Field(default=None, alias="createType")
    create_date: datetime | None = Field(default=None, alias="createDate")
    last_manual_avatar_upload_date: datetime | None = Field(
        default=None, alias="lastManualAvatarUploadDate"
    )
    current_user_leader: bool | None = Field(default=None, alias="currentUserLeader")
    order: int = 0

    # Avatar IDs:
    avatar_id: UUID | None = Field(default=None, alias="avatarId")
    small_avatar_id: UUID | None = Field(default=None, alias="smallAvatarId")
    facsimile_id: UUID | None = Field(default=None, alias="facsimileId")

    # Blocked fields:
    blocked_fields: list[str] | None = Field(default=None, alias="blockedFields")

    first_name: str | None = Field(default=None, alias="firstName")
    last_name: str | None = Field(default=None, alias="lastName")
    middle_name: str | None = Field(default=None, alias="middleName")

    department_id: UUID | None = Field(default=None, alias="departmentId")
    department_name: str | None = Field(default=None, alias="departmentName")
    department_code: str | None = Field(default=None, alias="departmentCode")
    post_name: str | None = Field(default=None, alias="postName")

    email: str | None = None
    phone: str | None = None
    address: str | None = None

    getu_id: str | None = Field(default=None, alias="getuId")
    ln_address: str | None = Field(default=None, alias="lnAddress")

    photo_path: str | None = Field(default=None, alias="photoPath")
    facsimile_path: str | None = Field(default=None, alias="facsimilePath")

    is_active: bool = Field(default=True, alias="isActive")

    @computed_field
    @property
    def full_name(self) -> str:
        """Full name in 'Фамилия Имя Отчество' format.

        Returns:
            Assembled display name. Empty string when no name parts are set.
        """
        parts = [p for p in [self.last_name, self.first_name, self.middle_name] if p]
        return " ".join(parts)

    @computed_field
    @property
    def short_name(self) -> str:
        """Abbreviated name in 'Фамилия И.О.' format.

        Returns:
            Abbreviated name, or ``full_name`` when ``last_name`` is absent.
        """
        if not self.last_name:
            return self.full_name
        initials = "".join(f"{n[0]}." for n in [self.first_name, self.middle_name] if n)
        return f"{self.last_name} {initials}".strip()

    def to_user_info(self) -> UserInfo:
        """Converts this Employee to a lightweight UserInfo reference.

        Used when constructing ``Document`` or ``Task`` entities that embed
        employees as value objects (author, executor, etc.).

        Returns:
            A frozen ``UserInfo`` instance populated from this employee's data.
        """
        return UserInfo(
            id=self.id,
            first_name=self.first_name,
            last_name=self.last_name,
            middle_name=self.middle_name,
            organization_id=self.organization_id,
            department_id=self.department_id,
            department_name=self.department_name,
            post_name=self.post_name,
            email=self.email,
        )

    def __str__(self) -> str:
        return self.short_name or str(self.id)
