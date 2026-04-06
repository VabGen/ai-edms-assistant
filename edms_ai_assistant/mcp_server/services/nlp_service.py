from __future__ import annotations

import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from edms_ai_assistant.config import settings
from edms_ai_assistant.utils.regex_utils import UUID_RE

logger = logging.getLogger(__name__)

_RESERVED_LOG_KEYS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "exc_info",
    "exc_text",
    "stack_info",
}


def safe_extra(**kwargs: Any) -> dict[str, Any]:
    """Prefix reserved LogRecord keys to avoid log-record attribute conflicts.

    Args:
        **kwargs: Arbitrary key-value pairs for structured logging.

    Returns:
        Dict safe to pass as ``extra`` to any logger call.
    """
    return {f"ctx_{k}" if k in _RESERVED_LOG_KEYS else k: v for k, v in kwargs.items()}


class UserIntent(Enum):
    """Detected user intention categories."""

    SUMMARIZE = "summarize"
    COMPARE = "compare"
    SEARCH = "search"
    CREATE_INTRODUCTION = "create_introduction"
    CREATE_TASK = "create_task"
    QUESTION = "question"
    ANALYZE = "analyze"
    EXTRACT = "extract"
    UPDATE = "update"
    DELETE = "delete"
    COMPOSITE = "composite"
    UNKNOWN = "unknown"
    FILE_ANALYSIS = "file_analysis"
    CREATE_DOCUMENT = "create_document"
    NOTIFICATION = "notification"


class QueryComplexity(Enum):
    """Estimated complexity level of a user query."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class EntityType(Enum):
    """Types of named entities that can be extracted from text."""

    DATE = "date"
    DATETIME = "datetime"
    PERSON = "person"
    NUMBER = "number"
    MONEY = "money"
    DOCUMENT_ID = "document_id"
    DEPARTMENT = "department"
    DURATION = "duration"


@dataclass
class Entity:
    """A single extracted named entity.

    Attributes:
        type: Entity category.
        value: Raw or parsed entity value.
        raw_text: Original text span.
        confidence: Extraction confidence in [0, 1].
        normalized_value: Canonical representation.
    """

    type: EntityType
    value: Any
    raw_text: str
    confidence: float = 1.0
    normalized_value: Any | None = None


@dataclass
class UserQuery:
    """Enriched representation of a user query.

    Attributes:
        original: Unmodified user input.
        refined: Post-processed query text.
        intent: Primary detected intent.
        secondary_intents: Additional detected intents.
        complexity: Estimated query complexity.
        entities: Extracted entities grouped by type name.
        keywords: Significant content words.
        confidence: Intent classification confidence.
    """

    original: str
    refined: str
    intent: UserIntent
    secondary_intents: list[UserIntent] = field(default_factory=list)
    complexity: QueryComplexity = QueryComplexity.SIMPLE
    entities: dict[str, list[Entity]] = field(default_factory=dict)
    keywords: set[str] = field(default_factory=set)
    confidence: float = 1.0


@dataclass
class SemanticContext:
    """Full semantic context produced for one user turn.

    Attributes:
        query: Enriched query object.
        document: Active EDMS document (if available).
        metadata: Computed metadata about the query/document.
        suggestions: Actionable suggestions for the user.
        warnings: Potential issues to surface to the user.
    """

    query: UserQuery
    document: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class EntityExtractor:
    """Rule-based named entity extractor for EDMS domain text."""

    DATE_PATTERNS: list[tuple[str, Any]] = [
        (
            r"(\d{1,2})\.(\d{1,2})\.(\d{4})",
            lambda m: f"{m[2]}-{int(m[1]):02d}-{int(m[0]):02d}",
        ),
        (
            r"(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)(?:\s+(\d{4}))?",
            "month_name",
        ),
        (r"\b(сегодня|завтра|вчера|послезавтра)\b", "relative_day"),
        (
            r"через\s+(\d+)\s+(день|дня|дней|неделю|недели|недель|месяц|месяца|месяцев)",
            "duration",
        ),
        (r"до\s+(\d{1,2})\.(\d{1,2})", "deadline"),
    ]

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

    def extract_dates(
        self,
        text: str,
        base_date: datetime | None = None,
    ) -> list[Entity]:
        """Extract and normalise date expressions from text.

        Args:
            text: Source text.
            base_date: Reference date for relative expressions.

        Returns:
            List of date entities with ISO-formatted normalised values.
        """
        if base_date is None:
            base_date = datetime.now()

        dates: list[Entity] = []

        for pattern, handler in self.DATE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                raw = match.group(0)
                try:
                    normalized: datetime
                    if handler == "month_name":
                        day = int(match.group(1))
                        month = self.MONTH_NAMES[match.group(2)]
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
                            days=delta_map[match.group(1)]
                        )

                    elif handler == "duration":
                        count = int(match.group(1))
                        unit = match.group(2)
                        if "день" in unit or "дня" in unit or "дней" in unit:
                            delta = timedelta(days=count)
                        elif "недел" in unit:
                            delta = timedelta(weeks=count)
                        elif "месяц" in unit:
                            delta = timedelta(days=count * 30)
                        else:
                            continue
                        normalized = base_date + delta

                    elif handler == "deadline":
                        day = int(match.group(1))
                        month = int(match.group(2))
                        if not (1 <= month <= 12 and 1 <= day <= 31):
                            continue
                        normalized = datetime(base_date.year, month, day)
                        if normalized < base_date:
                            normalized = datetime(base_date.year + 1, month, day)

                    elif callable(handler):
                        normalized = datetime.fromisoformat(handler(match.groups()))

                    else:
                        continue

                    dates.append(
                        Entity(
                            type=EntityType.DATE,
                            value=normalized,
                            raw_text=raw,
                            normalized_value=normalized.isoformat(),
                        )
                    )
                except (ValueError, KeyError) as exc:
                    logger.debug("Failed to parse date '%s': %s", raw, exc)

        return dates

    def extract_persons(self, text: str) -> list[Entity]:
        """Extract person names (ФИО) from text.

        Args:
            text: Source text in Russian.

        Returns:
            List of person entities with normalised name dicts.
        """
        persons: list[Entity] = []
        pattern = r"\b([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+)(?:\s+([А-ЯЁ][а-яё]+))?\b"

        for match in re.finditer(pattern, text):
            last_name = match.group(1)
            first_name = match.group(2)
            middle_name = match.group(3)

            if last_name.lower() in {"через", "после", "перед", "около"}:
                continue

            persons.append(
                Entity(
                    type=EntityType.PERSON,
                    value=match.group(0),
                    raw_text=match.group(0),
                    normalized_value={
                        "lastName": last_name,
                        "firstName": first_name,
                        "middleName": middle_name,
                    },
                    confidence=0.8 if middle_name else 0.6,
                )
            )

        return persons

    def extract_numbers(self, text: str) -> list[Entity]:
        """Extract numeric values from text.

        Args:
            text: Source text.

        Returns:
            List of number entities.
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
        """Extract monetary amounts with currency codes.

        Args:
            text: Source text.

        Returns:
            List of money entities with amount and currency dicts.
        """
        money: list[Entity] = []
        currency_patterns = [
            (r"(\d+(?:[.,]\d+)?)\s*(руб|₽|rub|бел\.руб)", "BYN"),
            (r"(\d+(?:[.,]\d+)?)\s*(\$|usd|долл)", "USD"),
            (r"(\d+(?:[.,]\d+)?)\s*(€|eur|евро)", "EUR"),
            (r"(\d+(?:[.,]\d+)?)\s*(руб|rub)", "RUB"),
        ]
        for pattern, currency in currency_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                amount = float(match.group(1).replace(",", "."))
                money.append(
                    Entity(
                        type=EntityType.MONEY,
                        value=amount,
                        raw_text=match.group(0),
                        normalized_value={"amount": amount, "currency": currency},
                    )
                )
        return money

    def extract_document_ids(self, text: str) -> list[Entity]:
        """Extract UUID document identifiers from text.

        Args:
            text: Source text.

        Returns:
            List of document ID entities.
        """
        doc_ids: list[Entity] = []
        uuid_re = r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
        for match in re.finditer(uuid_re, text, re.IGNORECASE):
            raw = match.group(0)
            doc_ids.append(
                Entity(
                    type=EntityType.DOCUMENT_ID,
                    value=raw,
                    raw_text=raw,
                    normalized_value=raw.lower(),
                )
            )
        return doc_ids

    def extract_all(
        self,
        text: str,
        base_date: datetime | None = None,
    ) -> dict[str, list[Entity]]:
        """Run all extractors and return a grouped entity dict.

        Args:
            text: Source text.
            base_date: Reference date for relative expressions.

        Returns:
            Dict mapping entity type names to lists of entities.
        """
        entities: dict[str, list[Entity]] = {}

        for key, extractor in [
            ("dates", lambda: self.extract_dates(text, base_date)),
            ("persons", lambda: self.extract_persons(text)),
            ("numbers", lambda: self.extract_numbers(text)),
            ("money", lambda: self.extract_money(text)),
            ("document_ids", lambda: self.extract_document_ids(text)),
        ]:
            result = extractor()
            if result:
                entities[key] = result

        return entities


class QueryRefiner:
    """Normalises and enriches raw user queries before LLM dispatch."""

    ABBREVIATIONS: dict[str, str] = {
        # Общие сокращения
        "doc": "документ",
        "dok": "документ",
        "доки": "документы",
        "сэд": "СЭД",
        # EDMS-специфичные
        "ознак": "ознакомление",
        "пор": "поручение",
        "исп": "исполнитель",
        "отв": "ответственный",
        "сов": "совещание",
        "дог": "договор",
        "рег": "регистрационный",
        "нотиф": "уведомление",
        "резол": "резолюция",
    }

    ACTION_SYNONYMS: dict[str, str] = {
        "покажи": "найди",
        "выведи": "найди",
        "дай": "найди",
        "скажи": "опиши",
        "расскажи": "опиши",
        "объясни": "опиши",
    }

    EDMS_DOMAIN_SYNONYMS: dict[str, str] = {
        # Версионирование / история (базовые формы + падежи)
        "история документа": "сравни версии документа",
        "историю документа": "сравни версии документа",
        "истории документа": "сравни версии документа",
        "история изменений": "сравни версии документа",
        "историю изменений": "сравни версии документа",
        "истории изменений": "сравни версии документа",
        "что менялось": "сравни версии документа",
        "что изменилось": "сравни версии документа",
        "покажи изменения": "сравни версии документа",
        "изменения в документе": "сравни версии документа",
        "посмотреть версии": "сравни версии документа",
        "какие версии": "сравни версии документа",
        # Резолюции
        "наложить резолюцию": "добавь резолюцию",
        "поставь визу": "добавь резолюцию",
        "написать резолюцию": "добавь резолюцию",
        "наложи резолюцию": "добавь резолюцию",
        "поставить резолюцию": "добавь резолюцию",
        # Поручения / задачи
        "дай задачу": "создай поручение",
        "поставь задачу": "создай поручение",
        "поставить задачу": "создай поручение",
        "назначь исполнителя": "создай поручение",
        "назначить исполнителя": "создай поручение",
        "поручи выполнить": "создай поручение",
        # Ознакомление
        "виза": "ознакомление",
        "завизировать": "добавить в ознакомление",
        "завизируй": "добавить в ознакомление",
        "отправь на визирование": "добавить в ознакомление",
        "отправить на ознакомление": "добавить в ознакомление",
        "отправь на ознакомление": "добавить в ознакомление",
        # Уведомления
        "напомни о документе": "отправь уведомление",
        "напомни о сроке": "отправь уведомление",
        "предупреди исполнителя": "отправь уведомление",
        "уведомить": "отправь уведомление",
        # Суммаризация
        "вкратце": "кратко опиши",
        "в двух словах": "кратко опиши",
        "суть документа": "кратко опиши документ",
        "о чём документ": "кратко опиши документ",
        "что в документе": "кратко опиши документ",
        # Поиск
        "найти документ": "найди документ",
        "поиск документов": "найди документы",
        "реестр документов": "найди документы",
        # Информация о документе
        "инфо": "информация о документе",
        "данные документа": "информация о документе",
        "реквизиты": "информация о документе",
        "статус документа": "информация о документе",
        "кто автор": "информация об авторе документа",
        "создай документ": "создай документ из файла",
        "создай обращение": "создай документ из файла",
        "создай входящий": "создай документ из файла",
        "создай исходящий": "создай документ из файла",
        "оформи": "создай документ из файла",
        "зарегистрируй": "создай документ из файла",
        "на основе файла": "создай документ из файла",
        "на основе этого": "создай документ из файла",
    }

    def normalize_domain_synonyms(self, text: str) -> str:
        """Replace EDMS domain jargon with canonical phrases before intent detection.

        Применяет замены за ОДИН ПРОХОД слева направо, от длинных фраз к коротким.
        Уже замененные участки текста помечаются как "использованные" — это предотвращает
        каскадные двойные замены (например: "история" → "версии" → "версии" снова).

        Args:
            text: Input user query text.

        Returns:
            Text with domain jargon replaced by canonical EDMS phrases.
        """
        text_lower = text.lower()
        sorted_pairs = sorted(
            self.EDMS_DOMAIN_SYNONYMS.items(), key=lambda x: -len(x[0])
        )
        if not sorted_pairs:
            return text_lower

        pattern = re.compile(
            "|".join(re.escape(jargon) for jargon, _ in sorted_pairs),
            flags=re.IGNORECASE,
        )
        canonical_map = {
            jargon.lower(): canonical for jargon, canonical in sorted_pairs
        }

        def _replace(m: re.Match) -> str:
            return canonical_map.get(m.group(0).lower(), m.group(0))

        return pattern.sub(_replace, text_lower)

    def expand_abbreviations(self, text: str) -> str:
        """Replace known abbreviations with full words.

        Args:
            text: Input text.

        Returns:
            Text with abbreviations expanded.
        """
        words = text.split()
        return " ".join(self.ABBREVIATIONS.get(w.lower(), w) for w in words)

    def normalize_actions(self, text: str) -> str:
        """Canonicalise action verbs.

        Args:
            text: Input text.

        Returns:
            Text with synonym verbs replaced by canonical forms.
        """
        text_lower = text.lower()
        for synonym, canonical in self.ACTION_SYNONYMS.items():
            text_lower = re.sub(r"\b" + synonym + r"\b", canonical, text_lower)
        return text_lower

    def add_context(
        self,
        text: str,
        intent: UserIntent,
        entities: dict[str, list[Entity]],
    ) -> str:
        """Augment the query with intent-specific structured context hints.

        Формирует структурированный суффикс к запросу — он попадает в HumanMessage
        и помогает модели сразу выбрать правильный инструмент и параметры.
        Суффикс пишется в формате [ключ: значение] — короткий и машиночитаемый.

        Args:
            text: Current query text (already normalized).
            intent: Detected primary intent.
            entities: Extracted entities grouped by type.

        Returns:
            Augmented query string with structured context suffix.
        """
        hints: list[str] = []

        if intent == UserIntent.SEARCH:
            if "persons" in entities:
                hints.append(f"исполнитель: {entities['persons'][0].value}")
            if "dates" in entities:
                hints.append(
                    f"дата: {entities['dates'][0].normalized_value or entities['dates'][0].raw_text}"
                )

        elif intent == UserIntent.CREATE_TASK:
            if "dates" not in entities:
                hints.append("срок: +7 дней (не указан)")
            else:
                hints.append(
                    f"срок: {entities['dates'][0].normalized_value or entities['dates'][0].raw_text}"
                )
            if "persons" in entities:
                hints.append(f"исполнитель: {entities['persons'][0].value}")

        elif intent == UserIntent.COMPARE:
            ids = entities.get("document_ids", [])
            if len(ids) >= 2:
                hints.append(f"версия_1: {ids[0].value}")
                hints.append(f"версия_2: {ids[1].value}")
            else:
                hints.append("версии: авто (первая↔последняя)")

        elif intent == UserIntent.NOTIFICATION:
            if "persons" in entities:
                hints.append(f"получатель: {entities['persons'][0].value}")
            if "dates" in entities:
                hints.append(f"дедлайн: {entities['dates'][0].raw_text}")

        elif intent == UserIntent.SUMMARIZE:
            if "numbers" in entities:
                hints.append(f"объём: {entities['numbers'][0].value} слов")

        elif intent == UserIntent.COMPOSITE:
            hints.append("тип: составной запрос (выполни последовательно)")

        if hints:
            suffix = " [" + "; ".join(hints) + "]"
            return text + suffix

        return text

    def refine(
        self,
        text: str,
        intent: UserIntent,
        entities: dict[str, list[Entity]],
    ) -> str:
        """Run the full refinement pipeline on a query.

        Pipeline order (each step feeds into the next):
        1. normalize_domain_synonyms — EDMS жаргон → канонические фразы
        2. expand_abbreviations — сокращения → полные слова
        3. normalize_actions — синонимы действий → канонические глаголы
        4. add_context — дополняем подсказками из entities/intent

        Args:
            text: Raw user query.
            intent: Detected primary intent.
            entities: Extracted entities.

        Returns:
            Refined query string ready for LLM prompt injection.
        """
        refined = text.strip()
        refined = self.normalize_domain_synonyms(refined)  # EDMS-жаргон первым
        refined = self.expand_abbreviations(refined)
        refined = self.normalize_actions(refined)
        refined = self.add_context(refined, intent, entities)
        return re.sub(r"\s+", " ", refined).strip()


class SemanticDispatcher:
    """Central NLP dispatcher: classifies intent, extracts entities, and builds SemanticContext.

    Args: None (stateless helpers are instantiated internally).
    """

    INTENT_KEYWORDS: dict[UserIntent, dict[str, list[str]]] = {
        UserIntent.CREATE_INTRODUCTION: {
            "primary": [
                "ознакомление",
                "ознакомь",
                "список ознакомления",
                "добавь в ознакомление",
                "добавить в ознакомление",
                "отправь на ознакомление",
            ],
            "secondary": ["виза", "визирование", "согласование", "подпись"],
            "negative": [],
        },
        UserIntent.CREATE_TASK: {
            "primary": [
                "поручение",
                "создай задачу",
                "создай поручение",
                "задание",
                "поручи",
                "поставь задачу",
                "назначь исполнителя",
            ],
            "secondary": ["исполнитель", "срок исполнения", "контроль", "выполнить"],
            "negative": [],
        },
        UserIntent.SUMMARIZE: {
            "primary": [
                "суммаризуй",
                "кратко",
                "резюме",
                "опиши",
                "сводка",
                "кратко опиши",
                "в двух словах",
                "суть документа",
                "о чём документ",
            ],
            "secondary": ["анализ", "выжимка", "основное", "тезисы", "пересказ"],
            "negative": [],
        },
        UserIntent.COMPARE: {
            "primary": [
                "сравни",
                "сравни версии",
                "отличия",
                "разница",
                "версии документа",
                "история документа",
                "что изменилось",
                "что менялось",
                "изменения в документе",
            ],
            "secondary": ["изменения", "версия", "обновление", "правки"],
            "negative": [],
        },
        UserIntent.SEARCH: {
            "primary": [
                "найди",
                "поиск",
                "найти документ",
                "поиск документов",
                "реестр документов",
            ],
            "secondary": ["выведи", "список", "реестр", "покажи", "где"],
            "negative": [],
        },
        UserIntent.ANALYZE: {
            "primary": [
                "проанализируй",
                "подробно",
                "детали",
                "детальный анализ",
            ],
            "secondary": ["разбор", "структура", "состав", "содержание"],
            "negative": [],
        },
        UserIntent.QUESTION: {
            "primary": [
                "какая",
                "какой",
                "сколько",
                "когда",
                "почему",
                "зачем",
                "кто",
                "чей",
            ],
            "secondary": ["расскажи", "объясни", "что", "как"],
            "negative": [],
        },
        UserIntent.FILE_ANALYSIS: {
            "primary": [
                "загруженный файл",
                "локальный файл",
                "файл на компьютере",
                "проанализируй файл",
                "что в файле",
                "о чем файл",
                "содержимое файла",
                "прочитай файл",
                "открой файл",
            ],
            "secondary": [
                "вложение",
                "содержимое",
                "текст файла",
                "данные из файла",
                "кратко",
                "пересказ",
                "тезисы",
            ],
            "negative": [],
        },
        UserIntent.CREATE_DOCUMENT: {
            "primary": [
                "создай документ из файла",
                "создай обращение",
                "создай входящий",
                "создай исходящий",
                "создай договор",
                "оформи обращение",
                "зарегистрируй обращение",
                "на основе файла",
                "на основе этого файла",
                "создать новый документ",
            ],
            "secondary": [
                "создай",
                "оформи",
                "зарегистрируй",
                "сделай входящий",
                "новый документ",
            ],
            "negative": [],
        },
        UserIntent.NOTIFICATION: {
            "primary": [
                "уведоми",
                "напомни",
                "отправь напоминание",
                "уведомление",
                "напоминание",
                "предупреди",
                "сообщи",
                "отправь уведомление",
            ],
            "secondary": ["дедлайн", "срок", "исполнитель", "отправь"],
            "negative": [],
        },
    }

    COMPOSITE_CONNECTORS: tuple[str, ...] = (
        " и ",
        " а также ",
        " плюс ",
        " потом ",
        " после этого ",
        ", а ",
        " затем ",
    )

    def __init__(self) -> None:
        self.entity_extractor = EntityExtractor()
        self.query_refiner = QueryRefiner()
        logger.info(
            "SemanticDispatcher initialised",
            extra=safe_extra(
                component="nlp_service",
                version="3.0.0",
                features=[
                    "domain_synonym_normalization",
                    "composite_intent_detection",
                    "structured_query_refinement",
                    "confidence_scoring",
                    "entity_extraction",
                    "file_analysis",
                ],
            ),
        )

    def detect_intent(
        self,
        message: str,
        file_path: str | None = None,
    ) -> tuple[UserIntent, list[UserIntent], float]:
        """Classify primary and secondary intents with confidence scoring.

        Scoring rules:
        - Local file present (non-UUID path): FILE_ANALYSIS +3, SUMMARIZE +2
        - primary keyword hit: +2 per keyword
        - secondary keyword hit: +1 per keyword
        - negative keyword hit: -1 per keyword
        - Question mark at end: QUESTION +1
        - COMPOSITE detected via connector heuristic: returns COMPOSITE primary

        Composite detection: если сообщение содержит разделитель между двумя
        разными интент-маркерами — запрос помечается COMPOSITE, оба интента
        передаются в secondary_intents для последовательной обработки агентом.

        Args:
            message: User message text.
            file_path: Optional file path or attachment UUID to boost FILE_ANALYSIS.

        Returns:
            Tuple of (primary_intent, secondary_intents, confidence).
        """
        if not message or not message.strip():
            return UserIntent.UNKNOWN, [], 0.0

        message_lower = self.query_refiner.normalize_domain_synonyms(message)

        scores: Counter[UserIntent] = Counter()

        if file_path and not UUID_RE.match(file_path):
            scores[UserIntent.FILE_ANALYSIS] += 3
            scores[UserIntent.SUMMARIZE] += 2
            scores[UserIntent.CREATE_DOCUMENT] += 2

        for intent, keywords in self.INTENT_KEYWORDS.items():
            primary_hits = sum(
                1 for kw in keywords.get("primary", []) if kw in message_lower
            )
            secondary_hits = sum(
                1 for kw in keywords.get("secondary", []) if kw in message_lower
            )
            negative_hits = sum(
                1 for kw in keywords.get("negative", []) if kw in message_lower
            )
            raw_score = primary_hits * 2 + secondary_hits - negative_hits
            if raw_score > 0:
                scores[intent] = scores.get(intent, 0) + raw_score

        if message.strip().endswith("?"):
            scores[UserIntent.QUESTION] += 1

        if not scores:
            return UserIntent.UNKNOWN, [], 0.0

        sorted_intents = scores.most_common()
        primary_intent = sorted_intents[0][0]
        primary_score = sorted_intents[0][1]

        intent_kw = self.INTENT_KEYWORDS.get(primary_intent, {})
        max_possible = len(intent_kw.get("primary", [])) * 2 + len(
            intent_kw.get("secondary", [])
        )
        confidence = min(primary_score / max(max_possible, 1), 1.0)

        threshold = primary_score * 0.5
        secondary_intents = [
            intent
            for intent, score in sorted_intents[1:]
            if score >= threshold and score > 0
        ]

        if secondary_intents:
            has_connector = any(
                connector in message_lower for connector in self.COMPOSITE_CONNECTORS
            )
            if has_connector and len(secondary_intents) >= 1:
                # Убеждаемся что у нас реально два разных намерения с ненулевыми scores
                composite_intents = [primary_intent] + secondary_intents[:2]
                logger.debug(
                    "Composite intent detected",
                    extra=safe_extra(
                        intents=[i.value for i in composite_intents],
                        connector_found=True,
                    ),
                )
                return UserIntent.COMPOSITE, composite_intents, confidence

        return primary_intent, secondary_intents, confidence

    def estimate_complexity(
        self,
        message: str,
        document: Any | None = None,
    ) -> QueryComplexity:
        """Estimate the processing complexity of a query.

        Args:
            message: User message text.
            document: Active EDMS document (if available).

        Returns:
            QueryComplexity level.
        """
        word_count = len(message.split())
        has_conditions = any(
            w in message.lower() for w in ("если", "когда", "где", "как", "при")
        )
        has_multiple_entities = (
            len(re.findall(r"\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\b", message)) > 2
        )

        score = 0
        if word_count > 20:
            score += 3
        elif word_count > 10:
            score += 2
        elif word_count > 5:
            score += 1

        if has_conditions:
            score += 2
        if has_multiple_entities:
            score += 1

        if document:
            attachments = getattr(document, "attachmentDocument", None) or []
            tasks = getattr(document, "taskList", None) or []
            if len(attachments) > 3:
                score += 1
            if len(tasks) > 5:
                score += 1
            if not getattr(getattr(document, "process", None), "completed", True):
                score += 1
            if getattr(document, "docCategoryConstant", None) == "CONTRACT":
                score += 1

        if score >= 8:
            return QueryComplexity.VERY_COMPLEX
        if score >= 5:
            return QueryComplexity.COMPLEX
        if score >= 2:
            return QueryComplexity.MEDIUM
        return QueryComplexity.SIMPLE

    def extract_keywords(self, message: str) -> set[str]:
        """Extract content keywords by removing Russian stop words.

        Args:
            message: User message text.

        Returns:
            Set of significant word stems.
        """
        stop_words = {
            "а",
            "в",
            "и",
            "на",
            "с",
            "о",
            "к",
            "по",
            "для",
            "из",
            "у",
            "от",
            "до",
            "это",
            "как",
            "что",
            "где",
            "когда",
            "мне",
            "меня",
            "мой",
            "моя",
            "мое",
            "можно",
            "нужно",
            "надо",
            "пожалуйста",
        }
        words = re.findall(r"\b[а-яёА-ЯЁ]{3,}\b", message.lower())
        return {w for w in words if w not in stop_words}

    def _validate_file_path(self, file_path: str) -> tuple[bool, str | None]:
        """Check that *file_path* exists and is within the configured upload directory.

        Args:
            file_path: Path string to validate.

        Returns:
            Tuple of (is_valid, error_message_or_none).
        """
        if not file_path:
            return False, "Путь не указан"
        try:
            path = Path(file_path).resolve()
            upload_dir = getattr(settings, "UPLOAD_DIR", None)
            if upload_dir:
                upload_path = Path(upload_dir).resolve()
                if upload_path not in path.parents and path.parent != upload_path:
                    return False, "Путь вне разрешённой директории"
            if not path.exists():
                return False, "Файл не найден"
            if not path.is_file():
                return False, "Указанный путь не является файлом"
            return True, None
        except Exception as exc:
            return False, f"Ошибка валидации: {exc}"

    def build_context(
        self,
        message: str,
        document: Any | None = None,
        file_path: str | None = None,
    ) -> SemanticContext:
        """Build a complete SemanticContext for one user turn.

        Args:
            message: Raw user message.
            document: Active EDMS document (if available).
            file_path: Local file path or EDMS attachment UUID.

        Returns:
            Fully populated SemanticContext.
        """
        primary_intent, secondary_intents, confidence = self.detect_intent(
            message, file_path
        )
        entities = self.entity_extractor.extract_all(message)
        complexity = self.estimate_complexity(message, document)
        keywords = self.extract_keywords(message)
        refined = self.query_refiner.refine(message, primary_intent, entities)

        query = UserQuery(
            original=message,
            refined=refined,
            intent=primary_intent,
            secondary_intents=secondary_intents,
            complexity=complexity,
            entities=entities,
            keywords=keywords,
            confidence=confidence,
        )

        metadata: dict[str, Any] = {
            "word_count": len(message.split()),
            "char_count": len(message),
            "has_question_mark": message.strip().endswith("?"),
        }

        if document:
            attachments = getattr(document, "attachmentDocument", None) or []
            tasks = getattr(document, "taskList", None) or []
            metadata.update(
                {
                    "has_document": True,
                    "document_id": str(getattr(document, "id", "")),
                    "document_category": str(
                        getattr(document, "docCategoryConstant", "")
                    ),
                    "document_status": str(getattr(document, "status", "")),
                    "attachments_count": len(attachments),
                    "tasks_count": len(tasks),
                }
            )

        if file_path:
            metadata["is_local_file"] = True
            metadata["ctx_file_path"] = file_path
            if not UUID_RE.match(file_path):
                metadata["file_type"] = "local"
                is_valid, error = self._validate_file_path(file_path)
                if not is_valid:
                    logger.warning(
                        "File validation failed",
                        extra=safe_extra(file_path=file_path, error=error),
                    )

        suggestions: list[str] = []
        warnings: list[str] = []

        if primary_intent == UserIntent.CREATE_TASK:
            if "dates" not in entities:
                suggestions.append("Рекомендуется указать срок выполнения поручения")
            if "persons" not in entities:
                warnings.append(
                    "Не указан исполнитель. Будет использован ответственный по умолчанию"
                )

        if primary_intent == UserIntent.COMPARE:
            if (
                "document_ids" not in entities
                or len(entities.get("document_ids", [])) < 2
            ):
                warnings.append(
                    "Для сравнения требуется указать два документа или версии"
                )

        if complexity == QueryComplexity.VERY_COMPLEX and confidence < 0.7:
            suggestions.append(
                "Запрос очень сложный. Рекомендуется разбить на несколько более простых"
            )

        return SemanticContext(
            query=query,
            document=document,
            metadata=metadata,
            suggestions=suggestions,
            warnings=warnings,
        )


class EDMSNaturalLanguageService:
    """High-level service for semantic analysis of EDMS domain objects."""

    @staticmethod
    def format_user(user: Any) -> str | None:
        """Format a UserInfoDto or EmployeeDto into a display name string.

        Args:
            user: Any object with lastName, firstName, middleName attributes.

        Returns:
            Formatted name string or None.
        """
        if not user:
            return None
        try:
            ln = getattr(user, "lastName", "") or ""
            fn = getattr(user, "firstName", "") or ""
            mn = getattr(user, "middleName", "") or ""
            post = getattr(user, "authorPost", "") or getattr(
                getattr(user, "post", None), "name", ""
            )
            name = f"{ln} {fn} {mn}".strip()
            return f"{name} ({post})" if post else name or None
        except Exception as exc:
            logger.debug("Error formatting user: %s", exc)
            return None

    @staticmethod
    def format_date(instant: Any) -> str | None:
        """Format an Instant-like value to DD.MM.YYYY.

        Args:
            instant: datetime object or ISO string.

        Returns:
            Formatted date string or None.
        """
        if not instant:
            return None
        try:
            if hasattr(instant, "strftime"):
                return instant.strftime("%d.%m.%Y")
            s = str(instant)
            if len(s) >= 10:
                parts = s[:10].split("-")
                if len(parts) == 3:
                    return f"{parts[2]}.{parts[1]}.{parts[0]}"
            return s[:10] if len(s) >= 10 else s
        except Exception as exc:
            logger.debug("Error formatting date: %s", exc)
            return None

    @staticmethod
    def format_datetime(instant: Any) -> str | None:
        """Format an Instant-like value to DD.MM.YYYY HH:MM.

        Args:
            instant: datetime object or ISO string.

        Returns:
            Formatted datetime string or None.
        """
        if not instant:
            return None
        try:
            if hasattr(instant, "strftime"):
                return instant.strftime("%d.%m.%Y %H:%M")
            s = str(instant)
            if len(s) >= 16:
                parts = s[:10].split("-")
                time_part = s[11:16]
                if len(parts) == 3:
                    return f"{parts[2]}.{parts[1]}.{parts[0]} {time_part}"
            return s[:16] if len(s) >= 16 else s
        except Exception as exc:
            logger.debug("Error formatting datetime: %s", exc)
            return None

    def get_safe(self, obj: Any, path: str, default: Any = None) -> Any:
        """Safely traverse a dot-separated attribute path.

        Args:
            obj: Root object.
            path: Dot-separated attribute path (e.g. ``"process.completed"``).
            default: Value returned when any segment is missing.

        Returns:
            Resolved value or *default*.
        """
        if obj is None:
            return default
        val = obj
        try:
            for part in path.split("."):
                if val is None:
                    return default
                val = (
                    val.get(part, default)
                    if isinstance(val, dict)
                    else getattr(val, part, default)
                )
                if hasattr(val, "value"):
                    return val.value
            return val if val is not None else default
        except (AttributeError, KeyError, TypeError) as exc:
            logger.debug("Error accessing path '%s': %s", path, exc)
            return default

    def process_document(self, doc: Any) -> dict[str, Any]:
        """Produce a full structured analysis of a DocumentDto.

        Covers ALL nested entities defined in the Java EDMS DTO schema:
        tasks with executors, attachments, recipients, process route,
        introduction list, nomenclature affairs, and category-specific
        sections (APPEAL, MEETING, MEETING_QUESTION, CONTRACT, QUESTION).

        Args:
            doc: DocumentDto instance (generated Pydantic model).

        Returns:
            Nested dict with all document sections, cleaned of None/empty values.
        """
        if not doc:
            logger.warning("Attempted to process None document")
            return {}

        try:
            # ── Категория (enum-safe) ──────────────────────────────────────────
            category = self.get_safe(doc, "docCategoryConstant")
            category_value: str = (
                category.value if hasattr(category, "value") else str(category or "")
            )

            # ── 1. Базовая информация ─────────────────────────────────────────
            base_info = {
                "id": str(doc.id) if getattr(doc, "id", None) else None,
                "категория": category_value,
                "краткое_содержание": getattr(doc, "shortSummary", None),
                "полный_текст": getattr(doc, "summary", None),
                "примечание": getattr(doc, "note", None),
                "профиль": getattr(doc, "profileName", None),
                "гриф_ДСП": getattr(doc, "dspFlag", None),
                "вид_документа": self.get_safe(doc, "documentType.typeName"),
                "способ_создания": self.get_safe(doc, "createType"),
            }

            # ── 2. Регистрация ────────────────────────────────────────────────
            registration = {
                "рег_номер": getattr(doc, "regNumber", None)
                or getattr(doc, "reservedRegNumber", None),
                "дата_регистрации": self.format_date(getattr(doc, "regDate", None)),
                "дата_создания": self.format_datetime(getattr(doc, "createDate", None)),
                "исходящий_номер": getattr(doc, "outRegNumber", None),
                "исходящая_дата": self.format_date(getattr(doc, "outRegDate", None)),
                "журнал_регистрации": self.get_safe(
                    doc, "registrationJournal.journalName"
                ),
                "версия": self.get_safe(doc, "version.version"),
                "признак_версионности": getattr(doc, "versionFlag", None),
                "страниц": getattr(doc, "pages", None),
                "кол-во_экземпляров": getattr(doc, "exemplarCount", None),
            }

            # ── 3. Участники ──────────────────────────────────────────────────
            responsible_executors = [
                self.format_user(getattr(r, "executor", None))
                for r in (getattr(doc, "responsibleExecutors", None) or [])
                if getattr(r, "executor", None)
            ]
            # Контрагенты договора — хранятся в recipientList
            # (DocumentRecipientDtoModel: name, unp, contractNumber и др.)
            _recipient_list_raw = getattr(doc, "recipientList", None) or []
            _contractors = [
                {
                    "название": getattr(r, "name", None),
                    "УНП": getattr(r, "unp", None),
                    "номер_договора_контрагента": getattr(r, "contractNumber", None),
                    "дата_договора_контрагента": self.format_date(
                        getattr(r, "contractDate", None)
                    ),
                }
                for r in _recipient_list_raw
                if getattr(r, "name", None)
            ] or None

            # Ответственные по договору — приходят из contractResponsible (enricher).
            # contractResponsible — list[dict], каждый элемент: {user: UserInfoDto, createDate}
            _contract_responsible_raw = getattr(doc, "contractResponsible", None) or []
            _contract_responsible_users: list[str] | None = [
                self.format_user(
                    r.get("user") if isinstance(r, dict) else getattr(r, "user", None)
                )
                for r in _contract_responsible_raw
                if (r.get("user") if isinstance(r, dict) else getattr(r, "user", None))
            ] or None

            participants = {
                "автор": self.format_user(getattr(doc, "author", None)),
                "инициатор": self.format_user(getattr(doc, "initiator", None)),
                "ответственный_исполнитель": self.format_user(
                    getattr(doc, "responsibleExecutor", None)
                ),
                "корреспондент": getattr(doc, "correspondentName", None),
                "контрагенты": _contractors,
                "ответственные_по_договору": _contract_responsible_users,
                "кем_подписан": self.format_user(getattr(doc, "whoSigned", None)),
                "председатель": self.format_user(getattr(doc, "chairperson", None)),
                "секретарь": self.format_user(getattr(doc, "secretary", None)),
                "ответственные_за_подготовку": responsible_executors or None,
            }

            # ── 4. Жизненный цикл ─────────────────────────────────────────────
            process_obj = getattr(doc, "process", None)
            process_detail: dict[str, Any] | None = None
            if process_obj:
                items_raw = getattr(process_obj, "items", None) or []
                process_items = [
                    {
                        "название": getattr(item, "name", None),
                        "тип": self.get_safe(item, "type"),
                        "статус": (
                            "завершён"
                            if getattr(item, "completed", False)
                            else (
                                "начат"
                                if getattr(item, "started", False)
                                else "ожидает"
                            )
                        ),
                        "дата_начала": self.format_datetime(
                            getattr(item, "start", None)
                        ),
                        "дата_конца": self.format_datetime(getattr(item, "end", None)),
                        "дней": getattr(item, "days", None),
                        "исполнители": [
                            {
                                "имя": self.format_user(getattr(ex, "executor", None)),
                                "результат": getattr(ex, "result", None),
                                "комментарий": getattr(ex, "comment", None),
                                "дата_исполнения": self.format_datetime(
                                    getattr(ex, "executionEnd", None)
                                ),
                            }
                            for ex in (getattr(item, "executors", None) or [])
                        ]
                        or None,
                    }
                    for item in items_raw
                ]
                process_detail = {
                    "завершен": self.get_safe(doc, "process.completed"),
                    "запущен": self.get_safe(doc, "process.started"),
                    "текущий_этап": self.get_safe(doc, "process.current.name"),
                    "следующий_этап": self.get_safe(doc, "process.next.name"),
                    "этапы": process_items or None,
                }

            lifecycle = {
                "текущий_статус": self.get_safe(doc, "status"),
                "предыдущий_статус": self.get_safe(doc, "prevStatus"),
                "текущий_этап_БП": getattr(doc, "currentBpmnTaskName", None),
                "процесс": process_detail,
            }

            # ── 5. Контроль ───────────────────────────────────────────────────
            control_obj = getattr(doc, "control", None)
            control_info: dict[str, Any] = {
                "на_контроле": getattr(doc, "controlFlag", None),
                "снять_с_контроля": getattr(doc, "removeControl", None),
                "дней_на_исполнение": getattr(doc, "daysExecution", None),
            }
            if control_obj:
                control_info.update(
                    {
                        "тип_контроля": self.get_safe(doc, "control.controlType.name"),
                        "дата_начала_контроля": self.format_date(
                            self.get_safe(doc, "control.controlDateStart")
                        ),
                        "плановая_дата_снятия": self.format_date(
                            self.get_safe(doc, "control.controlPlanDateEnd")
                        ),
                        "контролёр": self.format_user(
                            self.get_safe(doc, "control.controlEmployee")
                        ),
                    }
                )

            # ── 6. Поручения ──────────────────────────────────────────────────
            task_list = getattr(doc, "taskList", None) or []
            tasks_info: dict[str, Any] = {
                "общее_количество": getattr(doc, "countTask", None) or len(task_list),
                "выполнено": getattr(doc, "completedTaskCount", None),
                "список": [
                    {
                        "номер": getattr(t, "taskNumber", None),
                        "тип": self.get_safe(t, "type"),
                        "текст": getattr(t, "taskText", None),
                        "статус": self.get_safe(t, "taskStatus"),
                        "автор": self.format_user(getattr(t, "author", None)),
                        "срок": self.format_date(getattr(t, "planedDateEnd", None)),
                        "на_контроле": getattr(t, "onControl", None),
                        "бессрочное": getattr(t, "endless", None),
                        "периодическое": getattr(t, "periodTask", None),
                        "интервал_периода": self.get_safe(t, "period"),
                        "исполнители": [
                            {
                                "имя": self.format_user(getattr(ex, "executor", None)),
                                "ответственный": getattr(ex, "responsible", None),
                                "дата_исполнения": self.format_datetime(
                                    getattr(ex, "executedDate", None)
                                ),
                                "текст_отметки": getattr(ex, "stampText", None),
                            }
                            for ex in (getattr(t, "taskExecutors", None) or [])
                        ]
                        or None,
                    }
                    for t in task_list
                ]
                or None,
            }

            # ── 7. Вложения ───────────────────────────────────────────────────
            attachments_list = getattr(doc, "attachmentDocument", None) or []
            relations: dict[str, Any] = {
                "вложения": [
                    {
                        "название": getattr(a, "name", None),
                        "id": str(a.id) if getattr(a, "id", None) else None,
                        "тип_вложения": self.get_safe(a, "attachmentDocumentType"),
                        "вид": self.get_safe(a, "type"),
                        "размер_байт": getattr(a, "size", None),
                        "кол-во_подписей": len(getattr(a, "signs", None) or []) or None,
                        "дата_загрузки": self.format_datetime(
                            getattr(a, "uploadDate", None)
                        ),
                    }
                    for a in attachments_list
                ]
                or None,
            }

            # ── 8. Адресаты и корреспондент ───────────────────────────────────
            recipient_list = getattr(doc, "recipientList", None) or []
            if recipient_list:
                relations["адресаты"] = [
                    {
                        "название": getattr(r, "name", None),
                        "статус": self.get_safe(r, "status"),
                        "способ_доставки": self.get_safe(
                            r, "deliveryMethod.deliveryName"
                        ),
                        "отправлено": self.format_datetime(
                            getattr(r, "dateSend", None)
                        ),
                        "доставлено": getattr(r, "delivered", None),
                        "контрагент_УНП": getattr(r, "unp", None),
                        "контрагент_номер_договора": getattr(r, "contractNumber", None),
                    }
                    for r in recipient_list
                ]

            correspondent_obj = getattr(doc, "correspondent", None)
            if correspondent_obj:
                relations["корреспондент_орг"] = {
                    "название": self.get_safe(doc, "correspondent.name"),
                    "способ_доставки": self.get_safe(
                        doc, "correspondent.deliveryMethod.deliveryName"
                    ),
                }

            # ── 9. Ознакомления ───────────────────────────────────────────────
            intro_list = getattr(doc, "introduction", None) or []
            if intro_list:
                relations["ознакомления"] = {
                    "всего": getattr(doc, "introductionCount", None) or len(intro_list),
                    "выполнено": getattr(doc, "introductionCompleteCount", None),
                    "список": [
                        {
                            "сотрудник": self.format_user(getattr(i, "author", None)),
                            "дата_ознакомления": self.format_datetime(
                                getattr(i, "introductionDate", None)
                            ),
                            "комментарий": getattr(i, "comment", None),
                        }
                        for i in intro_list[
                            :10
                        ]  # не более 10, чтобы не раздувать промпт
                    ]
                    or None,
                }

            # ── 10. Предварительные номенклатурные дела ───────────────────────
            pre_noms = getattr(doc, "preNomenclatureAffairs", None) or []
            if pre_noms:
                relations["предвыбранные_дела"] = [
                    {
                        "дело": self.get_safe(pn, "nomenclatureAffair.name"),
                        "индекс": self.get_safe(pn, "nomenclatureAffair.index"),
                        "списан": getattr(pn, "writeOff", None),
                    }
                    for pn in pre_noms
                ]

            # ── 11. Специализированные секции по категории ───────────────────
            specialized: dict[str, Any] = {}

            # --- CONTRACT ---------------------------------------------------
            if getattr(doc, "contractNumber", None) or category_value == "CONTRACT":
                currency_code = self.get_safe(doc, "currency.code") or "BYN"
                # Контрагенты договора берутся из recipientList
                _contract_contractors = [
                    {
                        "название": getattr(r, "name", None),
                        "УНП": getattr(r, "unp", None),
                        "номер_договора": getattr(r, "contractNumber", None),
                        "дата_подписания": self.format_date(
                            getattr(r, "signDate", None)
                        ),
                    }
                    for r in (_recipient_list_raw or [])
                    if getattr(r, "name", None)
                ] or None
                specialized["договор"] = {
                    "номер": getattr(doc, "contractNumber", None),
                    "дата": self.format_date(getattr(doc, "contractDate", None)),
                    "сумма": (
                        f"{doc.contractSum} {currency_code}"
                        if getattr(doc, "contractSum", None) is not None
                        else None
                    ),
                    "валюта": self.get_safe(doc, "currency.name"),
                    "дата_подписания": self.format_date(
                        getattr(doc, "contractSigningDate", None)
                    ),
                    "срок_начала": self.format_date(
                        getattr(doc, "contractDurationStart", None)
                    ),
                    "срок_окончания": self.format_date(
                        getattr(doc, "contractDurationEnd", None)
                    ),
                    "автопролонгация": getattr(doc, "contractAutoProlongation", None),
                    "типовой": getattr(doc, "contractTypical", None),
                    "согласован_с_заказчиком": getattr(doc, "contractAgreement", None),
                    "контрагенты": _contract_contractors,
                    "ответственные_по_договору": _contract_responsible_users,
                }

            # --- APPEAL -----------------------------------------------------
            appeal_obj = getattr(doc, "documentAppeal", None)
            if appeal_obj or category_value == "APPEAL":
                if appeal_obj:
                    repeat_list = (
                        getattr(appeal_obj, "repeatIdenticalAppeals", None) or []
                    )
                    specialized["обращение"] = {
                        "заявитель": getattr(appeal_obj, "fioApplicant", None),
                        "тип_заявителя": self.get_safe(appeal_obj, "declarantType"),
                        "организация_заявителя": getattr(
                            appeal_obj, "organizationName", None
                        ),
                        "вид_обращения": self.get_safe(appeal_obj, "citizenType.name"),
                        "коллективное": getattr(appeal_obj, "collective", None),
                        "анонимное": getattr(appeal_obj, "anonymous", None),
                        "дата_поступления": self.format_datetime(
                            getattr(appeal_obj, "receiptDate", None)
                        ),
                        "страна": getattr(appeal_obj, "countryAppealName", None),
                        "регион": getattr(appeal_obj, "regionName", None),
                        "район": getattr(appeal_obj, "districtName", None),
                        "город": getattr(appeal_obj, "cityName", None),
                        "полный_адрес": getattr(appeal_obj, "fullAddress", None),
                        "телефон": getattr(appeal_obj, "phone", None),
                        "email": getattr(appeal_obj, "email", None),
                        "тематика": self.get_safe(appeal_obj, "subject.name"),
                        "рассмотрен_обоснованно": getattr(
                            appeal_obj, "reasonably", None
                        ),
                        "результат_решения": self.get_safe(
                            appeal_obj, "solutionResult.name"
                        ),
                        "ход_рассмотрения": getattr(appeal_obj, "reviewProgress", None),
                        "повторных_обращений": len(repeat_list) or None,
                        "исходящий_индекс_корреспондента": getattr(
                            appeal_obj, "correspondentOrgNumber", None
                        ),
                    }

            # --- MEETING ----------------------------------------------------
            if category_value == "MEETING" or getattr(doc, "dateMeeting", None):
                questions = getattr(doc, "documentQuestions", None) or []
                specialized["совещание"] = {
                    "дата": self.format_date(getattr(doc, "dateMeeting", None)),
                    "время_начала": self.format_datetime(
                        getattr(doc, "startMeeting", None)
                    ),
                    "время_конца": self.format_datetime(
                        getattr(doc, "endMeeting", None)
                    ),
                    "место": getattr(doc, "placeMeeting", None),
                    "председатель": self.format_user(getattr(doc, "chairperson", None)),
                    "секретарь": self.format_user(getattr(doc, "secretary", None)),
                    "внешние_приглашённые": getattr(doc, "externalInvitees", None),
                    "кол-во_приглашённых": getattr(doc, "inviteesCount", None),
                    "дополнение_к_повестке": getattr(doc, "addition", None),
                    "вопросы": [
                        {
                            "номер": getattr(q, "questionNumber", None),
                            "формулировка": getattr(q, "question", None),
                            "докладчики": [
                                {
                                    "имя": self.format_user(
                                        getattr(s, "employee", None)
                                    ),
                                    "тип": self.get_safe(s, "type"),
                                }
                                for s in (getattr(q, "speakers", None) or [])
                            ]
                            or None,
                        }
                        for q in questions
                    ]
                    or None,
                }

            # --- MEETING_QUESTION (Повестка заседания) ----------------------
            if category_value == "MEETING_QUESTION" or getattr(
                doc, "dateMeetingQuestion", None
            ):
                specialized["повестка_заседания"] = {
                    "дата_заседания": self.format_date(
                        getattr(doc, "dateMeetingQuestion", None)
                    ),
                    "форма_проведения": self.get_safe(doc, "formMeetingType"),
                    "номер_вопроса": getattr(doc, "numberQuestion", None),
                    "дата_вопроса": self.format_date(
                        getattr(doc, "dateQuestion", None)
                    ),
                    "комментарий_руководителю": getattr(doc, "commentQuestion", None),
                    "есть_вопросы": getattr(doc, "hasQuestion", None),
                }

            # ── 12. Сборка итогового словаря ──────────────────────────────────
            result: dict[str, Any] = {
                "базовая_информация": self._clean_dict(base_info),
                "регистрация": self._clean_dict(registration),
                "участники": self._clean_dict(participants),
                "жизненный_цикл": self._clean_dict(lifecycle),
                "контроль": self._clean_dict(control_info),
                "задачи": self._clean_dict(tasks_info),
                "связи_и_вложения": self._clean_dict(relations),
            }

            if specialized:
                result["специализированная_информация"] = self._clean_dict(specialized)

            return result

        except Exception as exc:
            logger.error("Error processing document: %s", exc, exc_info=True)
            return {"error": "Ошибка обработки документа", "details": str(exc)}

    def _clean_dict(self, d: Any) -> Any:
        """Recursively remove None, empty lists, and empty dicts from *d*.

        Args:
            d: Input data structure (dict, list, or scalar).

        Returns:
            Cleaned data structure, or None if empty after cleaning.
        """
        if isinstance(d, dict):
            cleaned = {k: self._clean_dict(v) for k, v in d.items()}
            cleaned = {k: v for k, v in cleaned.items() if v not in (None, [], {}, "")}
            return cleaned or None
        if isinstance(d, list):
            cleaned_list = [self._clean_dict(i) for i in d]
            cleaned_list = [i for i in cleaned_list if i not in (None, [], {}, "")]
            return cleaned_list or None
        return d

    def analyze_local_file(self, file_path: str) -> dict[str, Any]:
        """Return basic metadata for a local file before reading its content.

        Args:
            file_path: Absolute path to the file.

        Returns:
            Dict with filename, extension, size and content type.
        """
        try:
            if not os.path.exists(file_path):
                return {"error": "Файл не найден"}
            stats = os.stat(file_path)
            ext = os.path.splitext(file_path)[1].lower()
            return {
                "имя_файла": os.path.basename(file_path),
                "расширение": ext,
                "размер_мб": round(stats.st_size / (1024 * 1024), 2),
                "путь": file_path,
                "тип_контента": (
                    "Документ" if ext in {".pdf", ".docx", ".doc"} else "Текст"
                ),
            }
        except Exception as exc:
            logger.error(
                "Error analysing file %s: %s",
                file_path,
                exc,
                extra=safe_extra(file_path=file_path),
            )
            return {"error": str(exc)}

    def suggest_summarize_format(self, text: str) -> dict[str, Any]:
        """Recommend a summarisation format based on text characteristics.

        Args:
            text: Source text to analyse.

        Returns:
            Dict with ``recommended``, ``reason`` and ``stats`` keys.
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
                "reason": "Текст объемный или содержит много данных.",
                "stats": {"chars": length, "lines": lines},
            }
        if lines < 5:
            return {
                "recommended": "abstractive",
                "reason": "Компактный текст — подойдёт краткий пересказ.",
                "stats": {"chars": length, "lines": lines},
            }
        return {
            "recommended": "extractive",
            "reason": "Много конкретики — выделим ключевые факты.",
            "stats": {"chars": length, "lines": lines},
        }

    def process_employee_info(self, emp: Any) -> dict[str, Any]:
        """Build an analytical employee card from an EmployeeDto.

        Args:
            emp: EmployeeDto-like object.

        Returns:
            Nested dict with основное, контакты, структура sections.
        """
        if not emp:
            return {}
        try:
            return {
                "основное": {
                    "фио": self.format_user(emp),
                    "должность": self.get_safe(emp, "post.postName"),
                    "департамент": self.get_safe(emp, "department.name"),
                    "статус": "Уволен" if getattr(emp, "fired", False) else "Активен",
                },
                "контакты": {
                    "email": getattr(emp, "email", None),
                    "телефон": getattr(emp, "phone", None),
                },
                "структура": {
                    "код_департамента": self.get_safe(emp, "department.departmentCode"),
                },
            }
        except Exception as exc:
            logger.error("Error processing employee info: %s", exc, exc_info=True)
            return {"error": str(exc)}
