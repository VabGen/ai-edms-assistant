"""
nlp_preprocessor.py — NLU/NLP предобработка запросов EDMS AI Ассистента.

Выполняет до вызова LLM:
1. Нормализацию текста запроса
2. Извлечение именованных сущностей (UUID, даты, статусы, категории)
3. Определение намерения (intent) с оценкой уверенности
4. Принятие решения о fast-path (без LLM для очевидных запросов)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger("nlp")

# ── Intent definitions ────────────────────────────────────────────────────────

class Intent(str, Enum):
    GET_DOCUMENT      = "get_document"
    SEARCH_DOCUMENTS  = "search_documents"
    GET_HISTORY       = "get_history"
    GET_ATTACHMENTS   = "get_attachments"
    GET_VERSIONS      = "get_versions"
    CREATE_TASK       = "create_task"
    CREATE_INTRO      = "create_introduction"
    UPDATE_STATUS     = "update_status"
    SET_CONTROL       = "set_control"
    REMOVE_CONTROL    = "remove_control"
    START_DOCUMENT    = "start_document"
    SEARCH_EMPLOYEES  = "search_employees"
    GET_STATS         = "get_stats"
    UPDATE_FIELD      = "update_field"
    GET_USER_INFO     = "get_user_info"
    GENERAL_QUESTION  = "general_question"
    UNKNOWN           = "unknown"


@dataclass
class Entity:
    """Извлечённая именованная сущность."""
    type: str       # uuid, date, person, doc_number, status, category, amount
    value: Any
    raw: str        # исходный текст
    confidence: float = 1.0


@dataclass
class NLUResult:
    """Результат NLU-предобработки."""
    original_query: str
    normalized_query: str
    intent: Intent
    secondary_intents: list[Intent] = field(default_factory=list)
    confidence: float = 0.0
    entities: dict[str, list[Entity]] = field(default_factory=dict)
    can_skip_llm: bool = False   # True = прямой вызов инструмента без LLM
    fast_path_tool: str | None = None
    fast_path_args: dict[str, Any] = field(default_factory=dict)

    def get_entity(self, entity_type: str, default: Any = None) -> Any:
        """Получить первое значение сущности по типу."""
        items = self.entities.get(entity_type, [])
        return items[0].value if items else default

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "confidence": round(self.confidence, 2),
            "can_skip_llm": self.can_skip_llm,
            "entities": {k: [e.raw for e in v] for k, v in self.entities.items()},
        }


# ── Intent keyword maps ───────────────────────────────────────────────────────

# Format: (intent, base_confidence, [keywords/phrases])
_INTENT_RULES: list[tuple[Intent, float, list[str]]] = [
    (Intent.GET_DOCUMENT, 0.85, [
        "покажи документ", "открой документ", "документ по id", "документ с id",
        "получи документ", "найди документ по", "покажи карточку",
        "что за документ", "информация о документе",
    ]),
    (Intent.SEARCH_DOCUMENTS, 0.80, [
        "найди документы", "поиск документов", "покажи документы",
        "все документы", "список документов", "документы за", "документы со статусом",
        "входящие", "исходящие", "внутренние", "договоры", "обращения",
        "найди входящ", "найди исходящ",
    ]),
    (Intent.GET_HISTORY, 0.85, [
        "история документа", "история движения", "кто согласовал",
        "кто подписал", "этапы согласования", "движение документа",
        "журнал документа", "кто и когда",
    ]),
    (Intent.GET_ATTACHMENTS, 0.85, [
        "вложения", "файлы", "приложения к документу", "прикреплённые файлы",
        "что приложено", "список файлов",
    ]),
    (Intent.GET_VERSIONS, 0.85, [
        "версии документа", "история версий", "изменения в документе",
        "что менялось", "предыдущие версии",
    ]),
    (Intent.CREATE_TASK, 0.85, [
        "создай поручение", "создать поручение", "назначь поручение",
        "поставь задачу", "создай задачу", "назначь задачу",
        "добавь поручение", "поручи",
    ]),
    (Intent.CREATE_INTRO, 0.85, [
        "добавь в ознакомление", "ознакомление", "ознакомить",
        "список ознакомления", "добавить в ознакомление", "ознакомь",
    ]),
    (Intent.UPDATE_STATUS, 0.80, [
        "измени статус", "поменяй статус", "обнови статус",
        "изменить статус", "статус на", "перевести в статус",
    ]),
    (Intent.SET_CONTROL, 0.85, [
        "поставь на контроль", "взять на контроль", "поставить на контроль",
        "контрольный срок", "добавь контроль",
    ]),
    (Intent.REMOVE_CONTROL, 0.85, [
        "снять с контроля", "убрать с контроля", "снять контроль",
        "убери контроль",
    ]),
    (Intent.START_DOCUMENT, 0.85, [
        "запусти документ", "запустить документ", "начать согласование",
        "отправить на согласование", "начать маршрут",
    ]),
    (Intent.SEARCH_EMPLOYEES, 0.85, [
        "найди сотрудника", "найди сотрудников", "поиск сотрудника",
        "кто такой", "найди работника", "найди человека",
    ]),
    (Intent.GET_STATS, 0.80, [
        "статистика", "сколько документов", "мои документы", "мои задачи",
        "обзор", "сводка", "нагрузка",
    ]),
    (Intent.UPDATE_FIELD, 0.80, [
        "измени заголовок", "поменяй название", "обнови заголовок",
        "измени примечание", "поменяй содержание",
    ]),
    (Intent.GET_USER_INFO, 0.85, [
        "кто я", "мой профиль", "информация обо мне", "мои данные",
    ]),
]

# ── Patterns ──────────────────────────────────────────────────────────────────

_UUID_PATTERN = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)

_DOC_NUMBER_PATTERN = re.compile(
    r"\b(ВХ|ИСХ|ВН|ОБ|ДОГ)-\d{4}-\d+\b|\b\d{2,6}/\d{2,6}\b",
    re.IGNORECASE,
)

_DATE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b"), "dd.mm.yyyy"),
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), "iso"),
]

_STATUS_MAP: dict[str, str] = {
    "зарегистрирован": "REGISTERED",
    "в работе": "IN_PROGRESS",
    "выполнен": "COMPLETED",
    "завершён": "COMPLETED",
    "отменён": "CANCELLED",
    "аннулирован": "CANCELLED",
    "в архиве": "ARCHIVE",
}

_CATEGORY_MAP: dict[str, str] = {
    "входящ": "INCOMING",
    "исходящ": "OUTGOING",
    "внутренн": "INTERN",
    "обращени": "APPEAL",
    "договор": "CONTRACT",
    "совещани": "MEETING",
}

_PERSON_PATTERN = re.compile(
    r"\b([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+)(?:\s+([А-ЯЁ][а-яё]+))?\b"
)

# ── NLPPreprocessor ───────────────────────────────────────────────────────────

class NLPPreprocessor:
    """
    NLU-препроцессор запросов.

    Алгоритм:
    1. Нормализация текста
    2. Извлечение сущностей
    3. Классификация намерения
    4. Оценка возможности fast-path (пропуск LLM)
    """

    def __init__(self) -> None:
        self._now = datetime.now

    def process(self, query: str, context: dict[str, Any] | None = None) -> NLUResult:
        """
        Обработать запрос пользователя.

        Args:
            query: Исходный текст запроса
            context: Контекст сессии (active_document_id и т.п.)

        Returns:
            NLUResult с intent, entities, fast-path флагами
        """
        normalized = self._normalize(query)
        entities = self._extract_entities(normalized)

        # Если в контексте есть активный документ — добавляем его UUID
        if context and context.get("active_document_id"):
            if "uuid" not in entities:
                entities["uuid"] = [Entity(
                    type="uuid",
                    value=context["active_document_id"],
                    raw=context["active_document_id"],
                    confidence=0.9,
                )]

        intent, confidence, secondary = self._classify_intent(normalized, entities)

        can_skip, tool, args = self._check_fast_path(
            intent, confidence, entities, context
        )

        result = NLUResult(
            original_query=query,
            normalized_query=normalized,
            intent=intent,
            secondary_intents=secondary,
            confidence=confidence,
            entities=entities,
            can_skip_llm=can_skip,
            fast_path_tool=tool,
            fast_path_args=args,
        )

        logger.info(
            "NLU: intent=%s conf=%.2f skip_llm=%s entities=%s",
            intent.value, confidence, can_skip,
            {k: len(v) for k, v in entities.items()},
        )
        return result

    def _normalize(self, text: str) -> str:
        """Нормализовать текст: нижний регистр, пробелы, очистка."""
        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[«»\"']", "", text)
        return text

    def _extract_entities(self, text: str) -> dict[str, list[Entity]]:
        """Извлечь все именованные сущности."""
        entities: dict[str, list[Entity]] = {}

        # UUID
        for m in _UUID_PATTERN.finditer(text):
            entities.setdefault("uuid", []).append(
                Entity(type="uuid", value=m.group(), raw=m.group())
            )

        # Регистрационные номера
        for m in _DOC_NUMBER_PATTERN.finditer(text):
            entities.setdefault("doc_number", []).append(
                Entity(type="doc_number", value=m.group().upper(), raw=m.group())
            )

        # Даты
        for pat, fmt in _DATE_PATTERNS:
            for m in pat.finditer(text):
                try:
                    if fmt == "dd.mm.yyyy":
                        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    else:
                        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    parsed = datetime(year, month, day)
                    entities.setdefault("date", []).append(
                        Entity(type="date", value=parsed.isoformat()[:10], raw=m.group())
                    )
                except ValueError:
                    pass

        # Относительные даты
        rel_dates = self._extract_relative_dates(text)
        if rel_dates:
            entities.setdefault("date", []).extend(rel_dates)

        # Статусы
        for kw, val in _STATUS_MAP.items():
            if kw in text:
                entities.setdefault("status", []).append(
                    Entity(type="status", value=val, raw=kw)
                )

        # Категории документов
        for kw, val in _CATEGORY_MAP.items():
            if kw in text:
                entities.setdefault("category", []).append(
                    Entity(type="category", value=val, raw=kw)
                )

        # ФИО (упрощённо)
        for m in _PERSON_PATTERN.finditer(text):
            entities.setdefault("person", []).append(
                Entity(
                    type="person",
                    value={"last_name": m.group(1), "first_name": m.group(2)},
                    raw=m.group(),
                    confidence=0.7,
                )
            )

        return entities

    def _extract_relative_dates(self, text: str) -> list[Entity]:
        """Извлечь относительные даты: сегодня, завтра, через N дней."""
        results: list[Entity] = []
        now = self._now()

        if "сегодня" in text:
            results.append(Entity(type="date", value=now.isoformat()[:10], raw="сегодня"))
        if "завтра" in text:
            d = (now + timedelta(days=1)).isoformat()[:10]
            results.append(Entity(type="date", value=d, raw="завтра"))
        if "вчера" in text:
            d = (now - timedelta(days=1)).isoformat()[:10]
            results.append(Entity(type="date", value=d, raw="вчера"))

        # "через N дней/недель"
        for m in re.finditer(r"через\s+(\d+)\s+(день|дня|дней|неделю|недели|недель|месяц|месяца|месяцев)", text):
            n = int(m.group(1))
            unit = m.group(2)
            if "нед" in unit:
                delta = timedelta(weeks=n)
            elif "мес" in unit:
                delta = timedelta(days=30 * n)
            else:
                delta = timedelta(days=n)
            d = (now + delta).isoformat()[:10]
            results.append(Entity(type="date", value=d, raw=m.group()))

        return results

    def _classify_intent(
        self,
        text: str,
        entities: dict[str, list[Entity]],
    ) -> tuple[Intent, float, list[Intent]]:
        """Классифицировать намерение по ключевым словам."""
        scores: dict[Intent, float] = {}

        for intent, base_conf, keywords in _INTENT_RULES:
            for kw in keywords:
                if kw in text:
                    scores[intent] = max(scores.get(intent, 0.0), base_conf)

        # Буст для намерений подкреплённых сущностями
        if "uuid" in entities:
            if Intent.GET_DOCUMENT in scores:
                scores[Intent.GET_DOCUMENT] = min(1.0, scores[Intent.GET_DOCUMENT] + 0.1)
        if "doc_number" in entities and Intent.SEARCH_DOCUMENTS not in scores:
            scores[Intent.SEARCH_DOCUMENTS] = max(scores.get(Intent.SEARCH_DOCUMENTS, 0.0), 0.65)

        if not scores:
            return Intent.UNKNOWN, 0.0, []

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_intent, best_conf = sorted_scores[0]
        secondary = [i for i, c in sorted_scores[1:3] if c >= 0.5]

        return best_intent, best_conf, secondary

    def _check_fast_path(
        self,
        intent: Intent,
        confidence: float,
        entities: dict[str, list[Entity]],
        context: dict[str, Any] | None,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        """
        Определить возможность fast-path (вызов инструмента без LLM).

        Критерии fast-path:
        - Высокая уверенность в намерении (>= 0.85)
        - Все необходимые сущности присутствуют
        - Запрос простой (один шаг, без условий)
        """
        if confidence < 0.85:
            return False, None, {}

        token_placeholder = "{token}"  # заменяется оркестратором

        if intent == Intent.GET_DOCUMENT and "uuid" in entities:
            return True, "get_document", {
                "document_id": entities["uuid"][0].value,
                "token": token_placeholder,
            }

        if intent == Intent.GET_HISTORY and "uuid" in entities:
            return True, "get_document_history", {
                "document_id": entities["uuid"][0].value,
                "token": token_placeholder,
            }

        if intent == Intent.GET_ATTACHMENTS and "uuid" in entities:
            return True, "get_document_attachments", {
                "document_id": entities["uuid"][0].value,
                "token": token_placeholder,
            }

        if intent == Intent.GET_VERSIONS and "uuid" in entities:
            return True, "get_document_versions", {
                "document_id": entities["uuid"][0].value,
                "token": token_placeholder,
            }

        if intent == Intent.REMOVE_CONTROL and "uuid" in entities:
            return True, "remove_document_control", {
                "document_id": entities["uuid"][0].value,
                "token": token_placeholder,
            }

        if intent == Intent.GET_STATS:
            return True, "get_user_document_stats", {
                "token": token_placeholder,
            }

        if intent == Intent.GET_USER_INFO:
            return True, "get_current_user", {
                "token": token_placeholder,
            }

        return False, None, {}
