# src/ai_edms_assistant/infrastructure/nlp/extractors/base_extractor.py
"""Base NLP extractor with regex-based entity extraction.

Migrated and refactored from legacy ``edms_ai_assistant/services/nlp_service.py``
(EntityExtractor class) into the infrastructure layer.

Architecture:
    Infrastructure Layer → implements AbstractNLPExtractor port
    No external ML deps — pure regex + Python datetime stdlib.
    Production extractors (AppealExtractor) inherit from this base.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any

from ....application.ports.nlp_port import AbstractNLPExtractor
from ....domain.value_objects.nlp_entities import Entity, EntityType

logger = logging.getLogger(__name__)


class BaseExtractor(AbstractNLPExtractor):
    """Regex-based named entity extractor.

    Handles dates, persons (ФИО), numbers, money amounts, and document UUIDs.
    All patterns are tuned for Russian-language EDMS document text.

    Override ``extract_all`` in subclasses to add domain-specific extraction
    (e.g., appeal-specific fields in AppealExtractor).
    """

    # ── Russian month name → month number ────────────────────────────────────
    MONTH_NAMES: dict[str, int] = {
        "января": 1,
        "февраля": 2,
        "марта": 3,
        "апреля": 4,
        "мая": 5,
        "июня": 6,
        "июля": 7,
        "августа": 8,
        "сентября": 9,
        "октября": 10,
        "ноября": 11,
        "декабря": 12,
    }

    # ── Date patterns: (regex, handler_key | callable) ───────────────────────
    DATE_PATTERNS: list[tuple[str, Any]] = [
        # DD.MM.YYYY
        (
            r"(\d{1,2})\.(\d{1,2})\.(\d{4})",
            lambda m: datetime(int(m.group(3)), int(m.group(2)), int(m.group(1))),
        ),
        # DD месяц [YYYY]
        (
            r"(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа"
            r"|сентября|октября|ноября|декабря)(?:\s+(\d{4}))?",
            "month_name",
        ),
        # сегодня / завтра / вчера / послезавтра
        (r"\b(сегодня|завтра|вчера|послезавтра)\b", "relative_day"),
        # через N дней/недель/месяцев
        (
            r"через\s+(\d+)\s+(день|дня|дней|неделю|недели|недель|месяц|месяца|месяцев)",
            "duration",
        ),
        # до DD.MM
        (r"до\s+(\d{1,2})\.(\d{1,2})", "deadline"),
    ]

    # ── Currency patterns: (regex, currency_code) ────────────────────────────
    CURRENCY_PATTERNS: list[tuple[str, str]] = [
        (r"(\d+(?:[.,]\d+)?)\s*(бел\.?\s*руб|BYN|BYR)", "BYN"),
        (r"(\d+(?:[.,]\d+)?)\s*(\$|USD|долл\.?)", "USD"),
        (r"(\d+(?:[.,]\d+)?)\s*(€|EUR|евро)", "EUR"),
        (r"(\d+(?:[.,]\d+)?)\s*(руб\.?|RUB|₽)", "RUB"),
    ]

    # ── Stopwords for person extraction ──────────────────────────────────────
    _PERSON_STOPWORDS: frozenset[str] = frozenset(
        {
            "через",
            "после",
            "перед",
            "около",
            "между",
            "среди",
        }
    )

    # ── UUID pattern ──────────────────────────────────────────────────────────
    _UUID_RE = re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        re.IGNORECASE,
    )

    # ── Cyrillic person pattern (Фамилия Имя [Отчество]) ─────────────────────
    _PERSON_RE = re.compile(
        r"\b([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+)(?:\s+([А-ЯЁ][а-яё]+))?\b"
    )

    # ------------------------------------------------------------------
    # AbstractNLPExtractor interface
    # ------------------------------------------------------------------

    def extract_all(
        self,
        text: str,
        base_date: datetime | None = None,
    ) -> dict[str, list[Entity]]:
        """Extract all entity types from text.

        Args:
            text: Russian-language source text.
            base_date: Base datetime for relative date resolution (default: now).

        Returns:
            Dict mapping entity type names to Entity lists.
            Empty categories are omitted from the result.
        """
        if base_date is None:
            base_date = datetime.now()

        result: dict[str, list[Entity]] = {}

        dates = self.extract_dates(text, base_date)
        if dates:
            result["dates"] = dates

        persons = self.extract_persons(text)
        if persons:
            result["persons"] = persons

        numbers = self.extract_numbers(text)
        if numbers:
            result["numbers"] = numbers

        money = self.extract_money(text)
        if money:
            result["money"] = money

        doc_ids = self.extract_document_ids(text)
        if doc_ids:
            result["document_ids"] = doc_ids

        return result

    def suggest_summarize_format(self, text: str) -> dict[str, Any]:
        """Recommend summarization format based on text structure.

        Heuristics:
            - Long text (>5000 chars) or numeric-heavy → "thesis"
            - Short text (<5 lines) → "abstractive"
            - Default → "extractive"

        Args:
            text: Text to analyse.

        Returns:
            Dict with ``recommended``, ``reason``, ``stats`` keys.
        """
        if not text:
            return {
                "recommended": "abstractive",
                "reason": "Текст пуст",
                "stats": {"chars": 0, "lines": 0},
            }

        length = len(text)
        lines = text.count("\n")
        has_many_digits = len(re.findall(r"\d+", text)) > 20

        if length > 5000 or has_many_digits:
            return {
                "recommended": "thesis",
                "reason": "Объёмный текст с большим количеством данных — тезисный план удобнее.",
                "stats": {"chars": length, "lines": lines},
            }
        elif lines < 5:
            return {
                "recommended": "abstractive",
                "reason": "Компактный текст — краткий пересказ сути.",
                "stats": {"chars": length, "lines": lines},
            }
        else:
            return {
                "recommended": "extractive",
                "reason": "Конкретный текст — выделим ключевые факты.",
                "stats": {"chars": length, "lines": lines},
            }

    # ------------------------------------------------------------------
    # Extraction methods (can be called individually)
    # ------------------------------------------------------------------

    def extract_dates(
        self,
        text: str,
        base_date: datetime | None = None,
    ) -> list[Entity]:
        """Extract and normalise dates from Russian text.

        Args:
            text: Source text.
            base_date: Base date for relative expressions (default: now).

        Returns:
            List of DATE Entity objects with ``normalized_value`` as ISO string.
        """
        if base_date is None:
            base_date = datetime.now()

        dates: list[Entity] = []

        for pattern, handler in self.DATE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                raw = match.group(0)
                normalized: datetime | None = None

                try:
                    if callable(handler):
                        normalized = handler(match)

                    elif handler == "month_name":
                        day = int(match.group(1))
                        month = self.MONTH_NAMES[match.group(2).lower()]
                        year = int(match.group(3)) if match.group(3) else base_date.year
                        normalized = datetime(year, month, day)

                    elif handler == "relative_day":
                        delta_map = {
                            "сегодня": 0,
                            "завтра": 1,
                            "послезавтра": 2,
                            "вчера": -1,
                        }
                        normalized = base_date + timedelta(
                            days=delta_map[match.group(1).lower()]
                        )

                    elif handler == "duration":
                        count = int(match.group(1))
                        unit = match.group(2).lower()
                        if "день" in unit or "дня" in unit or "дней" in unit:
                            normalized = base_date + timedelta(days=count)
                        elif "недел" in unit:
                            normalized = base_date + timedelta(weeks=count)
                        elif "месяц" in unit:
                            normalized = base_date + timedelta(days=count * 30)

                    elif handler == "deadline":
                        day, month_num = int(match.group(1)), int(match.group(2))
                        if not (1 <= month_num <= 12 and 1 <= day <= 31):
                            continue
                        normalized = datetime(base_date.year, month_num, day)
                        if normalized < base_date:
                            normalized = datetime(base_date.year + 1, month_num, day)

                    if normalized:
                        dates.append(
                            Entity(
                                type=EntityType.DATE,
                                value=normalized,
                                raw_text=raw,
                                normalized_value=normalized.isoformat(),
                            )
                        )

                except (ValueError, KeyError, IndexError) as exc:
                    logger.debug(
                        "date_parse_failed",
                        extra={"raw": raw, "error": str(exc)},
                    )

        return dates

    def extract_persons(self, text: str) -> list[Entity]:
        """Extract ФИО (person names) from Russian text.

        Args:
            text: Source text.

        Returns:
            List of PERSON Entity objects with ``normalized_value`` as
            ``{"lastName": str, "firstName": str, "middleName": str|None}``.
        """
        persons: list[Entity] = []

        for match in self._PERSON_RE.finditer(text):
            last_name = match.group(1)
            first_name = match.group(2)
            middle_name = match.group(3)

            if last_name.lower() in self._PERSON_STOPWORDS:
                continue

            persons.append(
                Entity(
                    type=EntityType.PERSON,
                    value=match.group(0),
                    raw_text=match.group(0),
                    confidence=0.8 if middle_name else 0.6,
                    normalized_value={
                        "lastName": last_name,
                        "firstName": first_name,
                        "middleName": middle_name,
                    },
                )
            )

        return persons

    def extract_numbers(self, text: str) -> list[Entity]:
        """Extract numeric values from text.

        Args:
            text: Source text.

        Returns:
            List of NUMBER Entity objects.
        """
        numbers: list[Entity] = []

        for match in re.finditer(r"\b(\d+(?:[.,]\d+)?)\b", text):
            raw = match.group(0)
            value = float(raw.replace(",", "."))
            numbers.append(
                Entity(
                    type=EntityType.NUMBER,
                    value=value,
                    raw_text=raw,
                    normalized_value=value,
                )
            )

        return numbers

    def extract_money(self, text: str) -> list[Entity]:
        """Extract monetary amounts with currency from text.

        Args:
            text: Source text.

        Returns:
            List of MONEY Entity objects with ``normalized_value`` as
            ``{"amount": float, "currency": str}``.
        """
        money: list[Entity] = []

        for pattern, currency in self.CURRENCY_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                raw = match.group(0)
                amount = float(match.group(1).replace(",", "."))
                money.append(
                    Entity(
                        type=EntityType.MONEY,
                        value=amount,
                        raw_text=raw,
                        normalized_value={"amount": amount, "currency": currency},
                    )
                )

        return money

    def extract_document_ids(self, text: str) -> list[Entity]:
        """Extract document UUIDs from text.

        Args:
            text: Source text.

        Returns:
            List of DOCUMENT_ID Entity objects.
        """
        doc_ids: list[Entity] = []

        for match in self._UUID_RE.finditer(text):
            uuid_str = match.group(0).lower()
            doc_ids.append(
                Entity(
                    type=EntityType.DOCUMENT_ID,
                    value=uuid_str,
                    raw_text=match.group(0),
                    normalized_value=uuid_str,
                )
            )

        return doc_ids
