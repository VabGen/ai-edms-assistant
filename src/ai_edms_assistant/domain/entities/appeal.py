# src/ai_edms_assistant/domain/entities/appeal.py
"""Domain entity for citizen / legal-entity appeals (Обращения граждан).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field

from .base import DomainModel, MutableDomainModel


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DeclarantType(StrEnum):
    """Applicant type for citizen appeals (Тип заявителя).

    Attributes:
        INDIVIDUAL: Физическое лицо (private citizen).
        ENTITY: Юридическое лицо (legal entity / organization).
    """

    INDIVIDUAL = "INDIVIDUAL"
    ENTITY = "ENTITY"


class AppealChannel(StrEnum):
    """Channel through which the appeal was received.

    Maps to ``deliveryMethod`` in the EDMS API.
    """

    MAIL = "MAIL"
    EMAIL = "EMAIL"
    PERSONAL = "PERSONAL"
    PORTAL = "PORTAL"
    PHONE = "PHONE"
    OTHER = "OTHER"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


class GeoLocation(DomainModel):
    """Immutable geographic hierarchy for an appeal applicant.

    Populated from ``DocumentAppealDto`` flat fields:
        regionId / regionName / districtId / districtName / cityId / cityName.

    Attributes:
        country_id: UUID of the country in the EDMS reference dictionary.
        country_name: Human-readable country name.
        region_id: UUID of the region (oblast / kraj).
        region_name: Human-readable region name.
        district_id: UUID of the district (rajon).
        district_name: Human-readable district name.
        city_id: UUID of the city.
        city_name: Human-readable city name.
    """

    country_id: UUID | None = Field(default=None, alias="countryId")
    country_name: str | None = Field(default=None, alias="countryName")
    region_id: UUID | None = Field(default=None, alias="regionId")
    region_name: str | None = Field(default=None, alias="regionName")
    district_id: UUID | None = Field(default=None, alias="districtId")
    district_name: str | None = Field(default=None, alias="districtName")
    city_id: UUID | None = Field(default=None, alias="cityId")
    city_name: str | None = Field(default=None, alias="cityName")

    def as_text(self) -> str:
        """Returns a comma-separated human-readable location string.

        Returns:
            String like ``'Республика Беларусь, Минская область, Дзержинский, Батурово'``.
            Empty string when no location parts are populated.
        """
        parts = [
            p
            for p in [
                self.country_name,
                self.region_name,
                self.district_name,
                self.city_name,
            ]
            if p
        ]
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# Main entity
# ---------------------------------------------------------------------------


class DocumentAppeal(MutableDomainModel):
    """Domain entity for citizen / legal-entity appeals (Обращения граждан).
    """

    id: UUID | None = None

    appeal_number: str | None = Field(default=None, alias="appealNumber")
    applicant_name: str | None = Field(default=None, alias="fioApplicant")
    declarant_type: DeclarantType | None = Field(default=None, alias="declarantType")
    collective: bool | None = None
    anonymous: bool | None = None
    signed: bool | None = None

    email: str | None = None
    phone: str | None = None
    full_address: str | None = Field(default=None, alias="fullAddress")
    index: str | None = None
    organization_name: str | None = Field(default=None, alias="organizationName")

    # ── Geographic hierarchy (built by mapper from flat JSON fields) ──────
    geo_location: GeoLocation | None = Field(default=None, alias="geoLocation")

    # ── Country of applicant ──────────────────────────────────────────────
    country_appeal_id: UUID | None = Field(default=None, alias="countryAppealId")
    # FIX: добавлено — ранее отсутствовало, LLM не знал страну заявителя
    country_appeal_name: str | None = Field(default=None, alias="countryAppealName")

    # ── Appeal type (вид обращения) ────────────────────────────────────────
    citizen_type_id: UUID | None = Field(default=None, alias="citizenTypeId")
    citizen_type_name: str | None = Field(default=None, alias="citizenTypeName")

    # ── Subject / theme ────────────────────────────────────────────────────
    subject_id: UUID | None = Field(default=None, alias="subjectId")
    subject_name: str | None = Field(default=None, alias="subjectName")
    subject_parent_name: str | None = Field(default=None, alias="subjectParentName")

    question_category: str | None = Field(default=None, alias="questionCategory")

    representative_name: str | None = Field(default=None, alias="representativeName")

    receipt_date: datetime | None = Field(default=None, alias="receiptDate")
    date_doc_correspondent_org: datetime | None = Field(
        default=None, alias="dateDocCorrespondentOrg"
    )

    correspondent_appeal_id: UUID | None = Field(
        default=None, alias="correspondentAppealId"
    )
    correspondent_appeal: str | None = Field(default=None, alias="correspondentAppeal")
    correspondent_org_number: str | None = Field(
        default=None, alias="correspondentOrgNumber"
    )
    index_date_cover_letter: str | None = Field(
        default=None, alias="indexDateCoverLetter"
    )

    review_progress: str | None = Field(default=None, alias="reviewProgress")
    solution_result_id: UUID | None = Field(default=None, alias="solutionResultId")
    nomenclature_affair_id: UUID | None = Field(
        default=None, alias="nomenclatureAffairId"
    )
    reasonably: bool | None = None
    repeat_identical_appeals: bool | None = Field(
        default=None, alias="repeatIdenticalAppeals"
    )
    description: str | None = None

    # ── Computed properties ───────────────────────────────────────────────

    @property
    def is_anonymous(self) -> bool:
        """Returns True for anonymous appeals."""
        return bool(self.anonymous)

    @property
    def is_collective(self) -> bool:
        """Returns True for collective (multi-applicant) appeals."""
        return bool(self.collective)

    @property
    def declarant_type_label(self) -> str:
        """Returns Russian label for declarant type.

        Returns:
            'Юридическое лицо', 'Физическое лицо', or 'Не указан'.
        """
        if self.declarant_type == DeclarantType.ENTITY:
            return "Юридическое лицо"
        if self.declarant_type == DeclarantType.INDIVIDUAL:
            return "Физическое лицо"
        return "Не указан"

    @property
    def location_text(self) -> str:
        """Returns formatted address for LLM context.

        Priority: geo_location.as_text() → full_address → "Не указан".

        Returns:
            Human-readable address string.
        """
        if self.geo_location:
            text = self.geo_location.as_text()
            if text:
                return text
        return self.full_address or "Не указан"

    @property
    def subject_breadcrumb(self) -> str | None:
        """Returns subject hierarchy as breadcrumb string.

        Example:
            'Государство, общество, политика → Конституционные права...'

        Returns:
            Breadcrumb string or None if no subject data.
        """
        if not self.subject_name:
            return None
        if self.subject_parent_name:
            return f"{self.subject_parent_name} → {self.subject_name}"
        return self.subject_name