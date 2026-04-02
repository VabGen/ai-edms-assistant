"""
orchestrator/nlp_preprocessor.py — NLU-препроцессор запросов на русском языке.

Выполняет:
- Распознавание намерения (intent classification) с уверенностью 0..1
- Извлечение именованных сущностей (документы, даты, статусы, типы)
- Нормализацию запроса (плейсхолдеры вместо конкретных значений)
- Fast-path: при confidence > 0.92 и наличии всех сущностей — bypass LLM

Не имеет внешних зависимостей: только стандартная библиотека Python (re, dataclasses).
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Датаклассы результатов NLU
# ---------------------------------------------------------------------------

@dataclass
class ExtractedEntities:
    """
    Извлечённые именованные сущности из текста запроса.

    Поля:
        document_ids: найденные ID документов (DOC-\\d+, UUID v4, #\\d{4,})
        date_range: диапазон дат (start, end) или None
        statuses: нормализованные статусы документов
        document_types: типы документов (договор, приказ и т.д.)
        user_names: упомянутые имена пользователей/сотрудников
        departments: упомянутые отделы/подразделения
        page_number: номер страницы для пагинации
        limit: лимит результатов
    """
    document_ids: list[str] = field(default_factory=list)
    date_range: tuple[datetime, datetime] | None = None
    statuses: list[str] = field(default_factory=list)
    document_types: list[str] = field(default_factory=list)
    user_names: list[str] = field(default_factory=list)
    departments: list[str] = field(default_factory=list)
    page_number: int | None = None
    limit: int | None = None


@dataclass
class NLUResult:
    """
    Результат NLU-анализа запроса пользователя.

    Поля:
        intent: распознанное намерение
        confidence: уверенность классификации (0.0 – 1.0)
        entities: извлечённые сущности
        normalized_query: запрос с плейсхолдерами вместо конкретных значений
        bypass_llm: True если можно обойти LLM и вызвать MCP напрямую
        required_tool: имя MCP-инструмента для bypass_llm
        tool_args: готовые аргументы для вызова инструмента
    """
    intent: str
    confidence: float
    entities: ExtractedEntities
    normalized_query: str
    bypass_llm: bool = False
    required_tool: str | None = None
    tool_args: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Константы: паттерны и словари
# ---------------------------------------------------------------------------

# UUID v4 паттерн
_UUID_PATTERN = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)

# Номер документа вида DOC-12345 или #1234
_DOC_NUMBER_PATTERN = re.compile(
    r"\b(?:DOC-\d{1,10}|#\d{4,10}|№\s*\d{4,10})\b",
    re.IGNORECASE,
)

# Дата в формате ДД.ММ.ГГГГ или ДД/ММ/ГГГГ или ГГГГ-ММ-ДД
_DATE_EXPLICIT = re.compile(
    r"\b(\d{2})[./](\d{2})[./](\d{4})\b"
    r"|\b(\d{4})-(\d{2})-(\d{2})\b"
)

# Номер страницы
_PAGE_PATTERN = re.compile(r"\bстраниц[аую]?\s+(\d+)\b|\bpage\s+(\d+)\b", re.IGNORECASE)

# Лимит результатов
_LIMIT_PATTERN = re.compile(
    r"\b(?:первые|покажи|выведи|top)\s+(\d+)\b"
    r"|\b(\d+)\s+(?:документов|результатов|записей)\b",
    re.IGNORECASE,
)

# Статусы: синонимы → нормализованное значение
_STATUS_SYNONYMS: dict[str, str] = {
    "черновик": "draft",
    "черновике": "draft",
    "в работе": "draft",
    "на согласовании": "review",
    "согласовании": "review",
    "на проверке": "review",
    "ожидает согласования": "review",
    "одобрен": "approved",
    "согласован": "approved",
    "подтверждён": "approved",
    "утверждён": "approved",
    "отклонён": "rejected",
    "отказан": "rejected",
    "возвращён": "rejected",
    "не принят": "rejected",
    "подписан": "signed",
    "подписанный": "signed",
    "с подписью": "signed",
    "в архиве": "archived",
    "архивный": "archived",
    "архивирован": "archived",
    "закрыт": "archived",
}

# Типы документов
_DOC_TYPES: list[str] = ["договор", "приказ", "акт", "счёт", "протокол", "спецификация"]
_DOC_TYPE_VARIANTS: dict[str, str] = {
    "договоры": "договор",
    "договора": "договор",
    "договором": "договор",
    "приказы": "приказ",
    "приказа": "приказ",
    "акты": "акт",
    "актов": "акт",
    "актах": "акт",
    "счета": "счёт",
    "счетов": "счёт",
    "счётов": "счёт",
    "протоколы": "протокол",
    "протоколов": "протокол",
    "спецификации": "спецификация",
    "спецификаций": "спецификация",
}

# Относительные даты: паттерн → смещение в днях
_RELATIVE_DATES: list[tuple[re.Pattern[str], int, int]] = [
    # (pattern, days_start_offset, days_end_offset)
    # Отрицательный offset = в прошлом
    (re.compile(r"\bсегодня\b", re.IGNORECASE), 0, 0),
    (re.compile(r"\bвчера\b", re.IGNORECASE), -1, -1),
    (re.compile(r"\bзавтра\b", re.IGNORECASE), 1, 1),
    (re.compile(r"\bза\s+(?:последнюю|прошлую)\s+неделю\b", re.IGNORECASE), -7, 0),
    (re.compile(r"\bза\s+(?:эту|текущую)\s+неделю\b", re.IGNORECASE), -7, 0),
    (re.compile(r"\bза\s+последние\s+7\s+дней\b", re.IGNORECASE), -7, 0),
    (re.compile(r"\bза\s+последние\s+30\s+дней\b", re.IGNORECASE), -30, 0),
    (re.compile(r"\bза\s+последний\s+месяц\b", re.IGNORECASE), -30, 0),
    (re.compile(r"\bза\s+прошлый\s+месяц\b", re.IGNORECASE), -60, -30),
    (re.compile(r"\bв\s+(?:этом|текущем)\s+месяце\b", re.IGNORECASE), -30, 0),
    (re.compile(r"\bза\s+(?:последний|прошлый)\s+квартал\b", re.IGNORECASE), -90, 0),
    (re.compile(r"\bза\s+(?:этот|текущий)\s+год\b", re.IGNORECASE), -365, 0),
    (re.compile(r"\bза\s+прошлый\s+год\b", re.IGNORECASE), -730, -365),
]

# Известные отделы/подразделения (паттерны)
_DEPARTMENT_PATTERN = re.compile(
    r"\b(?:отдел|департамент|управление|служба|подразделение)\s+([А-Яа-яёЁ\s]+?)(?:\s|$|\.|,)",
    re.IGNORECASE,
)

# Паттерны имён (Фамилия Имя или Фамилия И.О.)
_NAME_PATTERN = re.compile(
    r"\b([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?)\b"
    r"|\b([А-ЯЁ][а-яё]+)\s+([А-ЯЁ])\.\s*([А-ЯЁ])\.\b"
)

# ---------------------------------------------------------------------------
# Intent: ключевые слова и паттерны
# ---------------------------------------------------------------------------

@dataclass
class _IntentPattern:
    """Паттерн для распознавания намерения."""
    intent: str
    keywords: list[str]
    phrases: list[str]
    weight: float  # базовый вес


_INTENT_PATTERNS: list[_IntentPattern] = [
    _IntentPattern(
        intent="get_document",
        keywords=["покажи", "открой", "найди документ", "что за документ", "посмотри"],
        phrases=[
            "покажи документ", "открой документ", "найди документ",
            "что за документ", "информация о документе", "детали документа",
            "содержание документа",
        ],
        weight=0.75,
    ),
    _IntentPattern(
        intent="search_documents",
        keywords=["найди", "поищи", "список", "покажи все", "все документы"],
        phrases=[
            "найди все", "поиск документов", "список документов",
            "покажи все договоры", "покажи все документы",
            "какие документы", "сколько документов",
        ],
        weight=0.70,
    ),
    _IntentPattern(
        intent="create_document",
        keywords=["создай", "заведи", "добавь", "оформи", "новый документ"],
        phrases=[
            "создай документ", "создай договор", "создай приказ",
            "заведи документ", "оформи договор", "новый документ",
            "добавь документ",
        ],
        weight=0.85,
    ),
    _IntentPattern(
        intent="update_status",
        keywords=["согласуй", "подпиши", "утверди", "отклони", "архивируй", "измени статус"],
        phrases=[
            "отправь на согласование", "согласовать документ",
            "подписать документ", "утвердить документ",
            "отклонить документ", "архивировать документ",
            "изменить статус", "сменить статус", "перевести в статус",
        ],
        weight=0.88,
    ),
    _IntentPattern(
        intent="get_history",
        keywords=["история", "журнал", "кто изменял", "аудит", "что происходило"],
        phrases=[
            "история документа", "журнал изменений", "кто изменял",
            "кто смотрел", "кто подписал", "что делали с документом",
            "аудит документа", "история изменений",
        ],
        weight=0.82,
    ),
    _IntentPattern(
        intent="assign_document",
        keywords=["назначь", "передай", "добавь", "поставь на согласование"],
        phrases=[
            "назначь ответственного", "передай документ",
            "добавь в согласующие", "поставь на согласование",
            "назначить рецензента", "добавить подписанта",
            "назначить наблюдателя",
        ],
        weight=0.82,
    ),
    _IntentPattern(
        intent="get_analytics",
        keywords=["статистика", "отчёт", "аналитика", "метрики", "дашборд", "сколько"],
        phrases=[
            "покажи статистику", "отчёт по документам",
            "сколько документов на согласовании", "аналитика",
            "нагрузка на отдел", "просроченные документы",
            "коэффициент одобрения", "дашборд",
        ],
        weight=0.78,
    ),
    _IntentPattern(
        intent="get_workflow_status",
        keywords=["статус", "процесс", "где застрял", "кто не согласовал", "прогресс"],
        phrases=[
            "где застрял документ", "кто ещё не согласовал",
            "статус согласования", "прогресс документа",
            "кто должен подписать", "workflow статус",
            "ожидает действия", "очередь согласования",
        ],
        weight=0.80,
    ),
]

# ---------------------------------------------------------------------------
# Классификатор намерений
# ---------------------------------------------------------------------------

class _IntentClassifier:
    """
    Правило-базированный классификатор намерений.
    Использует взвешенное совпадение ключевых слов и фраз.
    """

    def __init__(self, patterns: list[_IntentPattern]) -> None:
        self._patterns = patterns

    def classify(self, text: str) -> tuple[str, float]:
        """
        Классифицировать намерение.

        Параметры:
            text: нормализованный текст запроса

        Возвращает:
            (intent, confidence) — лучшее совпадение
        """
        text_lower = text.lower()
        scores: dict[str, float] = {}

        for pattern in self._patterns:
            score = 0.0

            # Точные фразы дают больший вес
            for phrase in pattern.phrases:
                if phrase in text_lower:
                    score += 0.35

            # Ключевые слова дают меньший вес
            for kw in pattern.keywords:
                if kw in text_lower:
                    score += 0.15

            if score > 0:
                scores[pattern.intent] = min(
                    score * pattern.weight, 0.98
                )

        if not scores:
            return "unknown", 0.3

        best_intent = max(scores, key=lambda k: scores[k])
        confidence = scores[best_intent]

        # Нормализуем уверенность
        confidence = min(confidence, 0.98)

        return best_intent, round(confidence, 3)


# ---------------------------------------------------------------------------
# Основной класс препроцессора
# ---------------------------------------------------------------------------

class NLPPreprocessor:
    """
    NLU-препроцессор для русскоязычных запросов EDMS.

    Не использует внешних зависимостей — только стандартная библиотека Python.
    Подходит для запросов вида:
    - «Покажи договор DOC-123»
    - «Найди все документы на согласовании за прошлый месяц»
    - «Отправь на подписание договор с Ивановым»
    """

    def __init__(self) -> None:
        self._classifier = _IntentClassifier(_INTENT_PATTERNS)

    def preprocess(self, text: str) -> NLUResult:
        """
        Выполнить полный NLU-анализ запроса.

        Параметры:
            text: исходный текст запроса пользователя

        Возвращает:
            NLUResult со всеми извлечёнными данными
        """
        # Нормализуем пробелы
        clean_text = " ".join(text.split())

        entities = self._extract_entities(clean_text)
        intent, confidence = self._classifier.classify(clean_text)
        normalized = self._normalize_query(clean_text, entities)

        # Проверяем bypass_llm
        bypass, tool, args = self._check_bypass(intent, confidence, entities)

        return NLUResult(
            intent=intent,
            confidence=confidence,
            entities=entities,
            normalized_query=normalized,
            bypass_llm=bypass,
            required_tool=tool,
            tool_args=args,
        )

    # ------------------------------------------------------------------
    # Извлечение сущностей
    # ------------------------------------------------------------------

    def _extract_entities(self, text: str) -> ExtractedEntities:
        """Извлечь все именованные сущности из текста."""
        return ExtractedEntities(
            document_ids=self._extract_document_ids(text),
            date_range=self._extract_date_range(text),
            statuses=self._extract_statuses(text),
            document_types=self._extract_document_types(text),
            user_names=self._extract_user_names(text),
            departments=self._extract_departments(text),
            page_number=self._extract_page_number(text),
            limit=self._extract_limit(text),
        )

    def _extract_document_ids(self, text: str) -> list[str]:
        """Извлечь UUID и номера документов."""
        ids: list[str] = []

        # UUID v4
        for match in _UUID_PATTERN.finditer(text):
            ids.append(match.group(0).lower())

        # DOC-NNNN, #NNNN
        for match in _DOC_NUMBER_PATTERN.finditer(text):
            raw = match.group(0).strip()
            # Нормализуем: #1234 → DOC-1234, № 1234 → DOC-1234
            normalized = re.sub(r"^[#№\s]+", "DOC-", raw).upper()
            ids.append(normalized)

        return ids

    def _extract_date_range(self, text: str) -> tuple[datetime, datetime] | None:
        """Извлечь диапазон дат из текста (абсолютные и относительные)."""
        now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

        # Сначала пробуем относительные даты
        for pattern, start_offset, end_offset in _RELATIVE_DATES:
            if pattern.search(text):
                start = now + timedelta(days=start_offset)
                end = now + timedelta(days=end_offset)
                if end < start:
                    start, end = end, start
                # end = конец дня
                end = end.replace(hour=23, minute=59, second=59)
                return (start, end)

        # Явные даты ДД.ММ.ГГГГ или ГГГГ-ММ-ДД
        found_dates: list[datetime] = []
        for match in _DATE_EXPLICIT.finditer(text):
            try:
                if match.group(1):  # ДД.ММ.ГГГГ
                    day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
                else:  # ГГГГ-ММ-ДД
                    year, month, day = int(match.group(4)), int(match.group(5)), int(match.group(6))
                dt = datetime(year, month, day, tzinfo=timezone.utc)
                found_dates.append(dt)
            except ValueError:
                continue

        if len(found_dates) == 1:
            d = found_dates[0]
            return (d, d.replace(hour=23, minute=59, second=59))
        if len(found_dates) >= 2:
            found_dates.sort()
            return (found_dates[0], found_dates[-1].replace(hour=23, minute=59, second=59))

        return None

    def _extract_statuses(self, text: str) -> list[str]:
        """Извлечь и нормализовать статусы документов."""
        text_lower = text.lower()
        found: list[str] = []

        for synonym, normalized_status in _STATUS_SYNONYMS.items():
            if synonym in text_lower and normalized_status not in found:
                found.append(normalized_status)

        return found

    def _extract_document_types(self, text: str) -> list[str]:
        """Извлечь типы документов."""
        text_lower = text.lower()
        found: list[str] = []

        # Прямые совпадения
        for doc_type in _DOC_TYPES:
            if doc_type in text_lower and doc_type not in found:
                found.append(doc_type)

        # Морфологические варианты
        for variant, canonical in _DOC_TYPE_VARIANTS.items():
            if variant in text_lower and canonical not in found:
                found.append(canonical)

        return found

    def _extract_user_names(self, text: str) -> list[str]:
        """Извлечь упомянутые имена пользователей."""
        names: list[str] = []
        for match in _NAME_PATTERN.finditer(text):
            # Берём первую группу которая дала совпадение
            if match.group(1) and match.group(2):
                full_name = f"{match.group(1)} {match.group(2)}"
                names.append(full_name)
            elif match.group(3) and match.group(4) and match.group(5):
                full_name = f"{match.group(3)} {match.group(4)}.{match.group(5)}."
                names.append(full_name)
        return names

    def _extract_departments(self, text: str) -> list[str]:
        """Извлечь названия отделов/подразделений."""
        departments: list[str] = []
        for match in _DEPARTMENT_PATTERN.finditer(text):
            dept = match.group(1).strip()
            if dept and len(dept) > 2:
                departments.append(dept)
        return departments

    def _extract_page_number(self, text: str) -> int | None:
        """Извлечь номер страницы."""
        match = _PAGE_PATTERN.search(text)
        if match:
            return int(match.group(1) or match.group(2))
        return None

    def _extract_limit(self, text: str) -> int | None:
        """Извлечь лимит результатов."""
        match = _LIMIT_PATTERN.search(text)
        if match:
            return int(match.group(1) or match.group(2))
        return None

    # ------------------------------------------------------------------
    # Нормализация запроса
    # ------------------------------------------------------------------

    def _normalize_query(self, text: str, entities: ExtractedEntities) -> str:
        """
        Нормализовать запрос: заменить конкретные значения плейсхолдерами.

        Это улучшает кэшируемость и позволяет сравнивать похожие запросы.
        """
        normalized = text

        # UUID → {document_uuid}
        normalized = _UUID_PATTERN.sub("{document_uuid}", normalized)

        # Номера документов → {document_id}
        normalized = _DOC_NUMBER_PATTERN.sub("{document_id}", normalized)

        # Явные даты → {date}
        normalized = _DATE_EXPLICIT.sub("{date}", normalized)

        # Числа в контексте лимита → {limit}
        normalized = _LIMIT_PATTERN.sub(r"\g<0>".replace(r"\g<0>", "{limit}"), normalized)

        # Имена → {person_name}
        for name in entities.user_names:
            normalized = normalized.replace(name, "{person_name}")

        return normalized.strip()

    # ------------------------------------------------------------------
    # Fast-path: bypass LLM
    # ------------------------------------------------------------------

    def _check_bypass(
        self,
        intent: str,
        confidence: float,
        entities: ExtractedEntities,
    ) -> tuple[bool, str | None, dict[str, Any] | None]:
        """
        Определить возможность обхода LLM для прямого вызова инструмента.

        Условие bypass: confidence > 0.92 И все необходимые сущности присутствуют.

        Возвращает:
            (bypass_llm, tool_name, tool_args)
        """
        if confidence <= 0.92:
            return False, None, None

        # get_document: нужен ID документа
        if intent == "get_document" and entities.document_ids:
            return True, "get_document", {
                "document_id": entities.document_ids[0],
                "include_history": False,
                "include_attachments": True,
            }

        # get_history: нужен ID документа
        if intent == "get_history" and entities.document_ids:
            return True, "get_document_history", {
                "document_id": entities.document_ids[0],
                "limit": entities.limit or 50,
            }

        # get_workflow_status: нужен ID документа
        if intent == "get_workflow_status" and entities.document_ids:
            return True, "get_workflow_status", {
                "document_id": entities.document_ids[0],
                "include_completed": False,
            }

        # search_documents: есть хотя бы один фильтр
        if intent == "search_documents":
            has_filters = any([
                entities.statuses,
                entities.document_types,
                entities.date_range,
                entities.user_names,
                entities.departments,
            ])
            if has_filters:
                args: dict[str, Any] = {
                    "page": entities.page_number or 1,
                    "page_size": entities.limit or 20,
                }
                if entities.statuses:
                    args["status"] = entities.statuses
                if entities.document_types:
                    args["document_type"] = entities.document_types
                if entities.date_range:
                    args["date_from"] = entities.date_range[0].isoformat()
                    args["date_to"] = entities.date_range[1].isoformat()
                if entities.user_names:
                    # Используем первое имя как автора
                    pass  # потребует resolve user_id — не можем bypass
                return True, "search_documents", args

        # get_analytics: достаточно знать тип метрики
        if intent == "get_analytics":
            args = {"metric_type": "status_distribution"}  # default
            if entities.date_range:
                args["date_from"] = entities.date_range[0].isoformat()
                args["date_to"] = entities.date_range[1].isoformat()
            return True, "get_analytics", args

        return False, None, None


# ---------------------------------------------------------------------------
# Фабричная функция для удобного использования
# ---------------------------------------------------------------------------

_preprocessor_singleton: NLPPreprocessor | None = None


def get_preprocessor() -> NLPPreprocessor:
    """Получить singleton экземпляр NLPPreprocessor."""
    global _preprocessor_singleton
    if _preprocessor_singleton is None:
        _preprocessor_singleton = NLPPreprocessor()
    return _preprocessor_singleton
