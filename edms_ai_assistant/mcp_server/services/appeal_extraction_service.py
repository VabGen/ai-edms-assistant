# mcp-server/services/appeal_extraction_service.py
"""
Appeal Extraction Service — LLM-извлечение данных из обращений.
Перенесён из edms_ai_assistant/services/appeal_extraction_service.py.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime

from ..llm import get_llm_response
from ..models.appeal_fields import AppealFields

logger = logging.getLogger(__name__)

_CITY_STOPWORDS: frozenset[str] = frozenset({
    "республики", "республика", "беларуси", "беларусь", "области", "область",
    "района", "район", "министерства", "министерство", "исполнительного",
    "комитета", "комитет", "совета", "совет", "центра", "центр",
    "лет", "годов", "года", "улицы", "улица", "проспекта", "проспект",
})

_POSTAL_PREFIX_CITY: dict[str, str] = {
    "210": "Витебск", "211": "Витебск", "212": "Могилёв", "213": "Могилёв",
    "220": "Минск", "221": "Минск", "222": "Молодечно", "223": "Борисов",
    "224": "Пинск", "225": "Брест", "230": "Гродно", "231": "Лида",
    "232": "Гродно", "236": "Барановичи", "246": "Гомель", "247": "Гомель",
    "248": "Гомель",
}

_SYSTEM_PROMPT = """Ты — эксперт-аналитик СЭД. Извлеки данные из официального обращения для регистрационной карточки.

ПРАВИЛА:
1. declarantType: INDIVIDUAL (физлицо) или ENTITY (юрлицо/организация)
2. shortSummary: суть обращения, СТРОГО не более 80 символов, законченная мысль
3. Все даты в ISO 8601 с суффиксом Z
4. Если поля нет — null (НЕ строку "None")
5. Ответ ТОЛЬКО в формате JSON без markdown

Поля: declarantType, fioApplicant, organizationName, shortSummary, citizenType,
collective, anonymous, reasonably, receiptDate, correspondentOrgNumber,
dateDocCorrespondentOrg, fullAddress, phone, email, signed,
country, regionName, districtName, cityName, index,
correspondentAppeal, reviewProgress, deliveryMethod"""


class AppealExtractionService:
    """
    Сервис LLM-извлечения структурированных данных из обращений.
    """

    MIN_TEXT_LENGTH = 30
    MAX_TEXT_LENGTH = 12000
    DEFAULT_MAX_RETRIES = 3

    async def extract_appeal_fields(self, text: str) -> AppealFields:
        """
        Извлечь поля обращения через LLM.

        Args:
            text: Текст обращения.

        Returns:
            AppealFields с извлечёнными данными.
        """
        if not text or len(text.strip()) < self.MIN_TEXT_LENGTH:
            return AppealFields()

        preprocessed = self._preprocess_text(text)
        truncated = preprocessed[:self.MAX_TEXT_LENGTH]

        try:
            prompt = f"Текст обращения для анализа:\n{'─'*60}\n{truncated}\n{'─'*60}\n\nВерни JSON с полями обращения:"
            response_text = await get_llm_response(prompt, system=_SYSTEM_PROMPT)

            clean = response_text.strip()
            if clean.startswith("```"):
                clean = re.sub(r"```(?:json)?", "", clean).strip().rstrip("`").strip()

            data = json.loads(clean)

            if isinstance(data, dict) and data.get("shortSummary"):
                ss = str(data["shortSummary"]).strip()
                if len(ss) > 80:
                    data["shortSummary"] = ss[:80]

            appeal_data = AppealFields.model_validate(data)
            appeal_data = self._post_process_fields(appeal_data, truncated)

            logger.info(
                "Appeal extracted: fio=%s org=%s type=%s city=%s",
                bool(appeal_data.fioApplicant),
                bool(appeal_data.organizationName),
                appeal_data.declarantType,
                appeal_data.cityName,
            )
            return appeal_data

        except Exception as e:
            logger.error("LLM extraction failed: %s", e, exc_info=True)
            return AppealFields()

    async def extract_with_retry(
        self, text: str, max_attempts: int | None = None
    ) -> AppealFields:
        """Извлечение с повторными попытками при неудаче."""
        max_attempts = max_attempts or self.DEFAULT_MAX_RETRIES
        for attempt in range(1, max_attempts + 1):
            try:
                result = await self.extract_appeal_fields(text)
                if self._is_valid_extraction(result):
                    return result
                logger.warning("Attempt %d/%d: insufficient data", attempt, max_attempts)
            except Exception as e:
                logger.error("Attempt %d/%d failed: %s", attempt, max_attempts, e)
            if attempt < max_attempts:
                await asyncio.sleep(2 ** attempt)

        logger.error("Extraction failed after %d attempts", max_attempts)
        return AppealFields()

    def _post_process_fields(self, fields: AppealFields, text: str) -> AppealFields:
        """Постобработка: парсинг дат, извлечение города, очистка signed."""
        if fields.declarantType == "ENTITY":
            if not fields.dateDocCorrespondentOrg and fields.correspondentOrgNumber:
                parsed_date = self._parse_date_from_number(fields.correspondentOrgNumber)
                if parsed_date:
                    fields.dateDocCorrespondentOrg = parsed_date

        if not fields.cityName and fields.fullAddress:
            extracted_city = self._extract_city_from_address(fields.fullAddress)
            if extracted_city:
                fields.cityName = extracted_city

        if fields.signed and len(fields.signed) > 20:
            fields.signed = self._extract_fio_from_signed(fields.signed)

        if fields.index and fields.fullAddress:
            if fields.index not in fields.fullAddress:
                fields.index = None

        if not fields.cityName:
            fallback_city = self._extract_city_from_full_text(
                text, phone=fields.phone, index=fields.index
            )
            if fallback_city:
                fields.cityName = fallback_city

        return fields

    def _parse_date_from_number(self, number: str) -> datetime | None:
        date_patterns = [
            r"от\s+(\d{1,2})\.(\d{1,2})\.(\d{4})",
            r"от\s+(\d{1,2})\s+([а-яА-Я]+)\s+(\d{4})",
        ]
        month_map = {
            "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
            "мая": 5, "июня": 6, "июля": 7, "августа": 8,
            "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
        }
        for pattern in date_patterns:
            match = re.search(pattern, number)
            if match:
                try:
                    day, month, year = match.groups()
                    month_num = int(month) if month.isdigit() else month_map.get(month.lower(), 1)
                    return datetime(int(year), month_num, int(day))
                except (ValueError, KeyError):
                    pass
        return None

    def _extract_city_from_address(self, address: str) -> str | None:
        m = re.search(r"\bг\.\s+([А-ЯЁ][а-яё]{3,})", address)
        if m and m.group(1).lower() not in _CITY_STOPWORDS:
            return m.group(1).strip()
        m = re.search(r"\d{6},?\s*([А-ЯЁ][а-яё]{3,})", address)
        if m and m.group(1).lower() not in _CITY_STOPWORDS:
            return m.group(1).strip()
        m = re.search(r"([А-ЯЁ][а-яё]{3,}),\s*ул\.", address)
        if m and m.group(1).lower() not in _CITY_STOPWORDS:
            return m.group(1).strip()
        return None

    def _extract_city_from_full_text(
        self, text: str, phone: str | None = None, index: str | None = None
    ) -> str | None:
        if index and len(index) >= 3:
            city = _POSTAL_PREFIX_CITY.get(index[:3])
            if city:
                return city
        m = re.search(r"\d{6},?\s*([А-ЯЁ][а-яё]{3,})", text)
        if m and m.group(1).lower() not in _CITY_STOPWORDS:
            return m.group(1).strip()
        return None

    @staticmethod
    def _extract_fio_from_signed(text: str) -> str:
        if not text or not text.strip():
            return text
        m = re.search(r"([А-ЯЁ]\.[А-ЯЁ]\.\s+[А-ЯЁ][а-яё]+)\s*$", text)
        if m:
            return m.group(1).strip()
        m = re.search(r"([А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.[А-ЯЁ]\.)\s*$", text)
        if m:
            return m.group(1).strip()
        m = re.search(r"([А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+)\s*$", text)
        if m:
            return m.group(1).strip()
        return text

    @staticmethod
    def _is_valid_extraction(result: AppealFields) -> bool:
        return any([result.fioApplicant, result.organizationName, result.shortSummary])

    @staticmethod
    def _preprocess_text(text: str) -> str:
        """Удаляет белорусскоязычные строки из двуязычных документов."""
        _BELARUSIAN_MARKERS = frozenset("ўЎіІ")
        lines = text.split("\n")
        filtered = [ln for ln in lines if not any(c in _BELARUSIAN_MARKERS for c in ln)]
        return "\n".join(filtered)