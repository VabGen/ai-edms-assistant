"""
NLP Preprocessor — Модуль предобработки естественно-языковых запросов.

Выполняет:
1. Извлечение именованных сущностей (UUID, даты, ФИО, номера документов)
2. Определение намерения пользователя (intent) с оценкой уверенности
3. Нормализацию запроса перед передачей в LLM
4. Fast-path: для простых запросов с высокой уверенностью — прямой вызов инструмента без LLM
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


# ── Intent definitions ────────────────────────────────────────────────────────


class Intent(str, Enum):
    """Типы намерений пользователя."""
    GET_DOCUMENT = "get_document"
    SEARCH_DOCUMENTS = "search_documents"
    DOCUMENT_HISTORY = "document_history"
    DOCUMENT_VERSIONS = "document_versions"
    STATISTICS = "statistics"
    SEARCH_EMPLOYEES = "search_employees"
    CURRENT_USER = "current_user"
    CREATE_TASK = "create_task"
    CREATE_INTRODUCTION = "create_introduction"
    AGREE_DOCUMENT = "agree_document"
    SIGN_DOCUMENT = "sign_document"
    REJECT_DOCUMENT = "reject_document"
    START_ROUTING = "start_routing"
    SET_CONTROL = "set_control"
    SEND_NOTIFICATION = "send_notification"
    GET_REFERENCES = "get_references"
    GENERAL_QUESTION = "general_question"
    SUMMARIZE = "summarize"
    COMPARE = "compare"
    ANALYZE = "analyze"
    UNKNOWN = "unknown"


@dataclass
class Entity:
    """Извлечённая именованная сущность."""
    type: str       # uuid, date, person, doc_number, status, category, amount
    value: Any
    raw_text: str
    start: int = 0
    end: int = 0
    normalized: Any = None


@dataclass
class NLUResult:
    """Результат NLU-анализа запроса."""
    original: str
    normalized: str
    intent: Intent
    confidence: float           # 0.0 — 1.0
    secondary_intents: list[Intent] = field(default_factory=list)
    entities: dict[str, list[Entity]] = field(default_factory=dict)
    keywords: list[str] = field(default_factory=list)
    can_skip_llm: bool = False  # True = прямой вызов инструмента без LLM
    suggested_tool: str | None = None
    suggested_args: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Pattern definitions ───────────────────────────────────────────────────────

UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
DATE_DMY = re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{2,4})\b")
DATE_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
DATE_RELATIVE = re.compile(
    r"\b(сегодня|вчера|завтра|послезавтра|"
    r"на прошлой неделе|на этой неделе|в этом месяце|в прошлом месяце|"
    r"через\s+\d+\s+(?:день|дня|дней|недел\w+|месяц\w*))\b",
    re.IGNORECASE,
)
PERSON_RE = re.compile(r"\b([А-ЯЁ][а-яё]{1,20})\s+([А-ЯЁ][а-яё]{1,15})\s+([А-ЯЁ][а-яё]{1,20})\b")
PERSON_LAST_FIRST = re.compile(r"\b([А-ЯЁ][а-яё]{1,20})\s+([А-ЯЁ])\.\s*([А-ЯЁ])\.\s*")
DOC_NUMBER_RE = re.compile(r"\b(?:ВХ|ИСХ|ВН|УТ|ДОГ)[-/]\d{4}[-/]\d+\b", re.IGNORECASE)
AMOUNT_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(руб|₽|BYN|USD|EUR|\$|€)\b", re.IGNORECASE)

MONTH_NAMES = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
    "мая": 5, "июня": 6, "июля": 7, "августа": 8,
    "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}
DATE_NAMED = re.compile(
    r"\b(\d{1,2})\s+(" + "|".join(MONTH_NAMES.keys()) + r")(?:\s+(\d{4}))?\b",
    re.IGNORECASE,
)

STOP_WORDS = {
    "а", "в", "и", "к", "на", "по", "с", "о", "от", "до", "за", "при",
    "из", "у", "же", "не", "но", "или", "что", "как", "это", "так",
    "пожалуйста", "можно", "нужно", "надо", "хочу", "хотел", "бы",
    "мне", "меня", "мой", "моя", "моё", "наш",
}

# ── Intent keyword maps ───────────────────────────────────────────────────────

INTENT_PATTERNS: list[tuple[Intent, float, list[str]]] = [
    # Format: (intent, base_confidence, [keywords/phrases])
    (Intent.GET_DOCUMENT, 0.85, [
        "покажи документ", "открой документ", "что за документ",
        "информация о документе", "данные документа", "документ по id",
        "найди документ по номеру", "реквизиты документа",
    ]),
    (Intent.SEARCH_DOCUMENTS, 0.8, [
        "найди документы", "поиск документов", "список документов",
        "покажи входящие", "покажи исходящие", "входящие документы",
        "договоры", "обращения граждан", "внутренние документы",
        "документы за", "документы от", "документы по",
    ]),
    (Intent.DOCUMENT_HISTORY, 0.9, [
        "история документа", "кто и что делал", "движение документа",
        "что происходило", "журнал", "аудит", "история движения",
        "кто согласовал", "кто подписал",
    ]),
    (Intent.DOCUMENT_VERSIONS, 0.9, [
        "версии документа", "изменения в документе", "что изменилось",
        "сравни версии", "история изменений", "какие были правки",
    ]),
    (Intent.STATISTICS, 0.85, [
        "статистика", "сколько документов", "сводка", "дашборд",
        "мои документы", "что у меня", "на исполнении", "на контроле",
        "общая статистика", "сколько на",
    ]),
    (Intent.SEARCH_EMPLOYEES, 0.85, [
        "найди сотрудника", "кто такой", "найди коллегу",
        "сотрудники отдела", "сотрудник по фамилии",
        "контакты", "телефон сотрудника",
    ]),
    (Intent.CURRENT_USER, 0.95, [
        "кто я", "мой профиль", "мои данные", "мои реквизиты",
        "текущий пользователь", "я сейчас",
    ]),
    (Intent.CREATE_TASK, 0.9, [
        "создай поручение", "поставь задачу", "назначь исполнителя",
        "создай задание", "поручи", "задача для", "дай поручение",
    ]),
    (Intent.CREATE_INTRODUCTION, 0.9, [
        "ознакомь", "добавь в ознакомление", "отправь на ознакомление",
        "список ознакомления", "для ознакомления",
    ]),
    (Intent.AGREE_DOCUMENT, 0.9, [
        "согласуй", "согласовать", "поставь визу", "одобри",
    ]),
    (Intent.SIGN_DOCUMENT, 0.9, [
        "подпиши", "подписать", "поставь подпись",
    ]),
    (Intent.REJECT_DOCUMENT, 0.9, [
        "отклони", "отклонить", "не согласовывать", "возврат на доработку",
    ]),
    (Intent.START_ROUTING, 0.9, [
        "запусти документ", "отправь на согласование", "запусти маршрут",
        "пустить по маршруту", "начать процесс",
    ]),
    (Intent.SET_CONTROL, 0.9, [
        "поставь на контроль", "установи контроль", "взять на контроль",
        "поставить на контроль",
    ]),
    (Intent.SEND_NOTIFICATION, 0.85, [
        "напомни", "уведоми", "отправь напоминание", "предупреди",
        "сообщи сотруднику", "напоминание о сроке",
    ]),
    (Intent.GET_REFERENCES, 0.8, [
        "виды документов", "типы контроля", "способы доставки",
        "справочник", "список типов", "возможные статусы",
    ]),
    (Intent.SUMMARIZE, 0.85, [
        "краткое содержание", "суммаризуй", "о чём документ",
        "что в документе", "перескажи", "кратко", "суть документа",
    ]),
    (Intent.COMPARE, 0.85, [
        "сравни", "отличия между", "разница между", "сравнение версий",
    ]),
    (Intent.ANALYZE, 0.8, [
        "проанализируй", "детальный анализ", "подробный разбор",
        "изучи документ", "аналитика по",
    ]),
    (Intent.GENERAL_QUESTION, 0.5, [
        "как", "что такое", "объясни", "расскажи о",
        "помоги разобраться", "не понимаю",
    ]),
]


# ── Main preprocessor class ───────────────────────────────────────────────────


class NLPPreprocessor:
    """
    Модуль предобработки запросов пользователей.

    Применяет многоуровневый анализ:
    1. Токенизация и нормализация
    2. Извлечение именованных сущностей
    3. Классификация намерения
    4. Принятие решения о fast-path (без LLM)
    """

    # Синонимы для нормализации
    SYNONYMS: dict[str, str] = {
        "покажи": "найди",
        "открой": "получи",
        "выведи": "найди",
        "достань": "получи",
        "виза": "согласование",
        "завизируй": "согласуй",
        "поручи": "создай поручение",
        "поставь задачу": "создай поручение",
        "отправь на вису": "согласуй",
        "отправь на подпись": "подпиши",
        "на контроль": "поставь на контроль",
        "сводка": "статистика",
        "дашборд": "статистика",
        "что в файле": "содержание документа",
        "что за бумага": "информация о документе",
    }

    def __init__(self) -> None:
        self._now = datetime.now

    def preprocess(self, text: str, context: dict[str, Any] | None = None) -> NLUResult:
        """
        Главный метод предобработки.

        Args:
            text: Исходный текст запроса пользователя
            context: Контекст сессии (текущий документ, пользователь и т.д.)

        Returns:
            NLUResult с intent, сущностями, нормализованным текстом
        """
        context = context or {}
        normalized = self._normalize(text)
        entities = self._extract_entities(text)
        intent, confidence, secondary = self._classify_intent(normalized, entities)
        keywords = self._extract_keywords(normalized)
        can_skip, tool, args = self._check_fast_path(
            intent, confidence, entities, context
        )

        return NLUResult(
            original=text,
            normalized=normalized,
            intent=intent,
            confidence=confidence,
            secondary_intents=secondary,
            entities=entities,
            keywords=keywords,
            can_skip_llm=can_skip,
            suggested_tool=tool,
            suggested_args=args,
            metadata={
                "has_document_id": bool(entities.get("uuid")),
                "has_date": bool(entities.get("date")),
                "has_person": bool(entities.get("person")),
                "context_doc_id": context.get("document_id"),
                "word_count": len(text.split()),
            },
        )

    def _normalize(self, text: str) -> str:
        """Нормализация текста: нижний регистр, замена синонимов, очистка."""
        result = text.lower().strip()
        result = re.sub(r"\s+", " ", result)
        for src, dst in self.SYNONYMS.items():
            result = result.replace(src, dst)
        return result

    def _extract_entities(self, text: str) -> dict[str, list[Entity]]:
        """Извлечение всех именованных сущностей из текста."""
        entities: dict[str, list[Entity]] = {}

        # UUID
        uuids = []
        for m in UUID_RE.finditer(text):
            uuids.append(Entity(
                type="uuid", value=m.group(0).lower(),
                raw_text=m.group(0), start=m.start(), end=m.end(),
                normalized=m.group(0).lower(),
            ))
        if uuids:
            entities["uuid"] = uuids

        # Dates: DD.MM.YYYY
        dates = []
        for m in DATE_DMY.finditer(text):
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if y < 100:
                y += 2000
            try:
                dt = datetime(y, mo, d)
                dates.append(Entity(
                    type="date", value=dt, raw_text=m.group(0),
                    start=m.start(), end=m.end(),
                    normalized=dt.strftime("%Y-%m-%d"),
                ))
            except ValueError:
                pass

        # Dates: YYYY-MM-DD
        for m in DATE_ISO.finditer(text):
            try:
                dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                dates.append(Entity(
                    type="date", value=dt, raw_text=m.group(0),
                    start=m.start(), end=m.end(),
                    normalized=dt.strftime("%Y-%m-%d"),
                ))
            except ValueError:
                pass

        # Dates: "15 января 2024"
        for m in DATE_NAMED.finditer(text):
            d = int(m.group(1))
            mo = MONTH_NAMES.get(m.group(2).lower(), 0)
            y = int(m.group(3)) if m.group(3) else self._now().year
            if mo:
                try:
                    dt = datetime(y, mo, d)
                    dates.append(Entity(
                        type="date", value=dt, raw_text=m.group(0),
                        start=m.start(), end=m.end(),
                        normalized=dt.strftime("%Y-%m-%d"),
                    ))
                except ValueError:
                    pass

        # Relative dates
        for m in DATE_RELATIVE.finditer(text):
            raw = m.group(0).lower()
            now = self._now()
            normalized_dt: datetime | None = None
            if "сегодня" in raw:
                normalized_dt = now
            elif "вчера" in raw:
                normalized_dt = now - timedelta(days=1)
            elif "завтра" in raw:
                normalized_dt = now + timedelta(days=1)
            elif "послезавтра" in raw:
                normalized_dt = now + timedelta(days=2)
            elif "прошлой неделе" in raw:
                normalized_dt = now - timedelta(weeks=1)
            elif "этой неделе" in raw:
                normalized_dt = now
            elif "этом месяце" in raw:
                normalized_dt = now.replace(day=1)
            elif "прошлом месяце" in raw:
                normalized_dt = (now.replace(day=1) - timedelta(days=1)).replace(day=1)

            if normalized_dt:
                dates.append(Entity(
                    type="date_relative", value=normalized_dt,
                    raw_text=raw, start=m.start(), end=m.end(),
                    normalized=normalized_dt.strftime("%Y-%m-%d"),
                ))

        if dates:
            entities["date"] = dates

        # Persons (ФИО)
        persons = []
        for m in PERSON_RE.finditer(text):
            persons.append(Entity(
                type="person", value={
                    "lastName": m.group(1),
                    "firstName": m.group(2),
                    "middleName": m.group(3),
                },
                raw_text=m.group(0), start=m.start(), end=m.end(),
                normalized=f"{m.group(1)} {m.group(2)[0]}. {m.group(3)[0]}.",
            ))
        for m in PERSON_LAST_FIRST.finditer(text):
            persons.append(Entity(
                type="person_initials",
                value={"lastName": m.group(1), "firstName": m.group(2), "middleName": m.group(3)},
                raw_text=m.group(0), start=m.start(), end=m.end(),
                normalized=f"{m.group(1)} {m.group(2)}. {m.group(3)}.",
            ))
        if persons:
            entities["person"] = persons

        # Document numbers
        doc_nums = []
        for m in DOC_NUMBER_RE.finditer(text):
            doc_nums.append(Entity(
                type="doc_number", value=m.group(0).upper(),
                raw_text=m.group(0), start=m.start(), end=m.end(),
                normalized=m.group(0).upper(),
            ))
        if doc_nums:
            entities["doc_number"] = doc_nums

        # Amounts
        amounts = []
        for m in AMOUNT_RE.finditer(text):
            amount = float(m.group(1).replace(",", "."))
            amounts.append(Entity(
                type="amount", value=amount,
                raw_text=m.group(0), start=m.start(), end=m.end(),
                normalized={"amount": amount, "currency": m.group(2).upper()},
            ))
        if amounts:
            entities["amount"] = amounts

        # Last names (single word, capitalized)
        last_names_re = re.compile(r"\b([А-ЯЁ][а-яё]{3,20})(а|у|е|ым|им|ов|ев|ева|ова)?\b")
        last_names = []
        person_positions = {e.start for e in persons}
        for m in last_names_re.finditer(text):
            if m.start() not in person_positions:
                word = m.group(1)
                if word not in {"Москва", "России", "EDMS", "UUID"}:
                    last_names.append(Entity(
                        type="last_name", value=word,
                        raw_text=m.group(0), start=m.start(), end=m.end(),
                        normalized=word,
                    ))
        if last_names:
            entities["last_name"] = last_names[:3]  # берём не более 3

        return entities

    def _classify_intent(
        self,
        normalized_text: str,
        entities: dict[str, list[Entity]],
    ) -> tuple[Intent, float, list[Intent]]:
        """Классификация намерения пользователя с уверенностью."""
        scores: dict[Intent, float] = {}

        for intent, base_conf, phrases in INTENT_PATTERNS:
            score = 0.0
            for phrase in phrases:
                if phrase in normalized_text:
                    # Длинные фразы дают больший вес
                    weight = min(len(phrase.split()) / 3.0, 2.0)
                    score = max(score, base_conf * (0.5 + weight * 0.3))
            if score > 0:
                scores[intent] = score

        # Бусты от сущностей
        if entities.get("uuid"):
            scores[Intent.GET_DOCUMENT] = max(scores.get(Intent.GET_DOCUMENT, 0), 0.6)
        if entities.get("last_name") or entities.get("person"):
            scores[Intent.SEARCH_EMPLOYEES] = max(scores.get(Intent.SEARCH_EMPLOYEES, 0) + 0.1, 0)
        if entities.get("date"):
            scores[Intent.SEARCH_DOCUMENTS] = max(scores.get(Intent.SEARCH_DOCUMENTS, 0) + 0.1, 0)

        if not scores:
            return Intent.UNKNOWN, 0.2, []

        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_intents[0]
        secondary = [i for i, s in sorted_intents[1:] if s > 0.4][:2]

        return primary[0], min(primary[1], 1.0), secondary

    def _extract_keywords(self, normalized: str) -> list[str]:
        """Извлечение значимых ключевых слов."""
        words = re.findall(r"\b[а-яёa-z]{3,}\b", normalized)
        return [w for w in words if w not in STOP_WORDS][:10]

    def _check_fast_path(
        self,
        intent: Intent,
        confidence: float,
        entities: dict[str, list[Entity]],
        context: dict[str, Any],
    ) -> tuple[bool, str | None, dict[str, Any]]:
        """
        Определяет, можно ли обработать запрос без LLM (fast-path).

        Fast-path применяется при confidence > 0.85 и наличии всех нужных данных.
        """
        if confidence < 0.85:
            return False, None, {}

        doc_id = (
            entities["uuid"][0].value if entities.get("uuid")
            else context.get("document_id")
        )
        token = context.get("token", "")

        # get_document: есть UUID + явный запрос документа
        if intent == Intent.GET_DOCUMENT and doc_id:
            return True, "get_document", {"document_id": doc_id, "token": token}

        # document_history: есть UUID
        if intent == Intent.DOCUMENT_HISTORY and doc_id:
            return True, "get_document_history", {"document_id": doc_id, "token": token}

        # statistics: без параметров
        if intent == Intent.STATISTICS:
            return True, "get_document_statistics", {"token": token}

        # current_user
        if intent == Intent.CURRENT_USER:
            return True, "get_current_user", {"token": token}

        # search_employees: есть фамилия
        if intent == Intent.SEARCH_EMPLOYEES:
            last_names = entities.get("last_name") or entities.get("person")
            if last_names:
                last_name = (
                    last_names[0].value if isinstance(last_names[0].value, str)
                    else last_names[0].value.get("lastName", "")
                )
                return True, "search_employees", {"last_name": last_name, "token": token}

        return False, None, {}

    def build_enriched_prompt_context(self, result: NLUResult) -> str:
        """
        Формирует строку контекста для добавления в LLM промпт.

        Включает: намерение, сущности, нормализованный запрос.
        """
        parts = [
            f"[NLU] Намерение: {result.intent.value} (уверенность: {result.confidence:.2f})",
        ]
        if result.secondary_intents:
            parts.append(f"[NLU] Вторичные намерения: {', '.join(i.value for i in result.secondary_intents)}")

        if result.entities.get("uuid"):
            uuids = [e.value for e in result.entities["uuid"]]
            parts.append(f"[NLU] UUID документов: {', '.join(uuids)}")

        if result.entities.get("date"):
            dates = [e.normalized for e in result.entities["date"]]
            parts.append(f"[NLU] Даты: {', '.join(dates)}")

        if result.entities.get("person") or result.entities.get("last_name"):
            persons_raw = result.entities.get("person", []) + result.entities.get("last_name", [])
            names = [
                (e.normalized if e.normalized else e.raw_text) for e in persons_raw
            ]
            parts.append(f"[NLU] Персоны: {', '.join(names[:3])}")

        if result.keywords:
            parts.append(f"[NLU] Ключевые слова: {', '.join(result.keywords[:5])}")

        return "\n".join(parts)
