# edms_ai_assistant/mcp_server/models/appeal_fields.py
"""
Модели данных для автоматического заполнения карточек обращений граждан (APPEAL).
"""

import logging
import re
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class DeclarantType(StrEnum):
    """
    Тип заявителя согласно справочнику СЭД.
    """

    INDIVIDUAL = "INDIVIDUAL"
    ENTITY = "ENTITY"


class AppealFields(BaseModel):
    """
    Структура данных для извлечения информации из текста обращения с помощью LLM.
    """

    # --- Основные поля ---
    deliveryMethod: str | None = Field(None, description="Способ доставки обращения")
    shortSummary: str | None = Field(
        None,
        description="Краткое содержание (до 200 символов)",
    )
    receiptDate: datetime | None = Field(None, description="Дата поступления обращения")
    declarantType: DeclarantType | None = Field(
        None, description="Тип заявителя: INDIVIDUAL (физлицо) или ENTITY (юрлицо)"
    )
    citizenType: str | None = Field(
        None, description="Категория обращения (Жалоба, Заявление, Предложение)"
    )

    # --- Логические признаки ---
    collective: bool | None = Field(None)
    anonymous: bool | None = Field(None)
    reasonably: bool | None = Field(None)

    # --- Заявитель (Физическое лицо) ---
    fioApplicant: str | None = Field(None, description="ФИО заявителя")

    # --- Заявитель (Юридическое лицо) ---
    organizationName: str | None = Field(None)
    signed: str | None = Field(None)
    correspondentOrgNumber: str | None = Field(None)
    dateDocCorrespondentOrg: datetime | None = Field(None)

    # --- География ---
    country: str | None = Field(None)
    regionName: str | None = Field(None)
    districtName: str | None = Field(None)
    cityName: str | None = Field(None)
    index: str | None = Field(None, description="Почтовый индекс (6 цифр)")
    fullAddress: str | None = Field(None)

    # --- Контактная информация ---
    phone: str | None = Field(None)
    email: str | None = Field(None)

    # --- Дополнительные сведения ---
    correspondentAppeal: str | None = Field(None)
    indexDateCoverLetter: str | None = Field(None)
    reviewProgress: str | None = Field(None)

    @field_validator("shortSummary")
    @classmethod
    def truncate_summary(cls, v: str | None) -> str | None:
        """Truncate shortSummary to 200 chars with ellipsis."""
        if v and len(v) > 200:
            logger.debug("Краткое содержание обрезано: %d → 200 символов", len(v))
            return v[:197] + "..."
        return v

    @field_validator("index")
    @classmethod
    def validate_index(cls, v: str | None) -> str | None:
        if v:
            cleaned = re.sub(r"\D", "", str(v))
            if len(cleaned) != 6:
                return None
            return cleaned
        return v

    @model_validator(mode="after")
    def clean_placeholders(self) -> "AppealFields":
        placeholders = {
            "none",
            "null",
            "nil",
            "unknown",
            "n/a",
            "na",
            "no",
            "not specified",
            "not available",
            "неизвестно",
            "н/д",
            "нет данных",
            "отсутствует",
            "не указано",
            "нет",
            "—",
            "-",
            "–",
            "...",
            "___",
        }

        for field_name in self.model_fields:
            value = getattr(self, field_name)

            if isinstance(value, str):
                trimmed_value = value.strip()

                if trimmed_value.lower() in placeholders or not trimmed_value:
                    setattr(self, field_name, None)

        return self
