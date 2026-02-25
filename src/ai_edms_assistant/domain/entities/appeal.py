# src/ai_edms_assistant/domain/entities/appeal.py
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

    Used in autofill logic to determine which API fields to populate
    and which validation rules to apply.

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
# Value objects (geographic hierarchy)
# ---------------------------------------------------------------------------


class GeoLocation(DomainModel):
    """Immutable geographic hierarchy for an appeal applicant.

    Stores the full address breakdown as a structured object rather than
    a flat string to enable comparison, filtering, and autofill against
    the EDMS reference API (regions / districts / cities dictionaries).

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

        Suitable for injection into LLM prompts as a compact address context.

        Returns:
            String like 'Россия, Московская обл., Одинцовский р-н, Одинцово'.
            Empty string when no location parts are populated.

        Example:
            >>> loc = GeoLocation(country_name="Россия", region_name="Московская обл.")
            >>> loc.as_text()
            'Россия, Московская обл.'
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

    Maps to ``DocumentAppealDto`` in the Java backend. This entity is
    embedded inside the ``Document`` aggregate as the ``appeal`` field
    and is only populated when ``document_category == APPEAL``.

    The ``geo_location`` field provides a structured alternative to the
    flat ``full_address`` string — autofill logic should prefer structured
    data for API calls and fall back to ``full_address`` for LLM context.

    Attributes:
        id: Appeal record UUID.
        applicant_name: Full applicant name (fioApplicant in Java DTO).
        declarant_type: Physical or legal entity.
        collective: Whether the appeal has multiple co-signers.
        anonymous: Whether the applicant identity is withheld.
        signed: Whether the appeal document is physically signed.
        email: Applicant's e-mail address.
        phone: Applicant's phone number.
        full_address: Free-text postal address (fallback from geo_location).
        index: Postal index (ZIP code).
        organization_name: Organization name for legal entity applicants.
        geo_location: Structured geographic hierarchy (preferred over full_address).
        citizen_type_id: UUID reference to appeal type in EDMS dictionary.
        question_category: Topic / theme of the appeal (for LLM analytics).
        subject_id: UUID reference to the subject theme in EDMS dictionary.
        representative_name: Name of the legal representative (if present).
        receipt_date: Date when the appeal was received.
        date_doc_correspondent_org: Date of the outgoing registration at correspondent org.
        correspondent_appeal_id: UUID of the organization that forwarded the appeal.
        correspondent_appeal: Name of the forwarding organization (for LLM).
        correspondent_org_number: Outgoing registration number at the correspondent.
        index_date_cover_letter: Cover letter date and index string.
        review_progress: Free-text description of review progress (for LLM).
        solution_result_id: UUID reference to the resolution result dictionary.
        nomenclature_affair_id: UUID of the nomenclature affair for archiving.
        reasonably: Whether the appeal was deemed substantiated.
        description: Free-text description of the appeal content (for LLM).
    """

    id: UUID

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

    geo_location: GeoLocation | None = Field(default=None, alias="geoLocation")

    citizen_type_id: UUID | None = Field(default=None, alias="citizenTypeId")
    question_category: str | None = Field(default=None, alias="questionCategory")
    subject_id: UUID | None = Field(default=None, alias="subjectId")

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
    description: str | None = None

    @property
    def is_anonymous(self) -> bool:
        """Returns True for anonymous appeals."""
        return bool(self.anonymous)

    @property
    def is_collective(self) -> bool:
        """Returns True for collective (multi-applicant) appeals."""
        return bool(self.collective)

    @property
    def applicant_summary(self) -> str:
        """Compact applicant description for LLM context injection.

        Assembles a single-line string combining name, declarant type label,
        and location. Used to populate the agent's system prompt context.

        Returns:
            Formatted string like:
            'Иванов Иван Иванович (физ. лицо), Россия, Московская обл., Одинцово'

        Example:
            >>> from uuid import UUID, uuid4
            >>> appeal = DocumentAppeal(
            ...     id=uuid4(),
            ...     applicant_name="Иванов И.И.",
            ...     declarant_type=DeclarantType.INDIVIDUAL,
            ... )
            >>> appeal.applicant_summary
            'Иванов И.И. (физ. лицо)'
        """
        parts: list[str] = []

        if self.applicant_name:
            type_label = {
                DeclarantType.INDIVIDUAL: "физ. лицо",
                DeclarantType.ENTITY: "юр. лицо",
            }.get(self.declarant_type, "")
            suffix = f" ({type_label})" if type_label else ""
            parts.append(f"{self.applicant_name}{suffix}")

        if self.geo_location:
            location_text = self.geo_location.as_text()
            if location_text:
                parts.append(location_text)
        elif self.full_address:
            parts.append(self.full_address)

        return ", ".join(parts)
