# edms_ai_assistant\orchestrator/nlp_preprocessor.py
"""
NLU-препроцессор для русскоязычных запросов EDMS.

Без внешних зависимостей — только стандартная библиотека Python (re, dataclasses).

Экспортирует:
    ExtractedEntities — извлечённые сущности
    NLUResult         — результат NLU-анализа
    NLPPreprocessor   — основной класс
    get_preprocessor() — синглтон
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any


# ── Датаклассы ────────────────────────────────────────────────────────────


@dataclass
class ExtractedEntities:
    """Именованные сущности, извлечённые из текста запроса."""

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
    """Результат NLU-анализа запроса пользователя."""

    intent: str
    confidence: float
    entities: ExtractedEntities
    normalized_query: str
    bypass_llm: bool = False
    required_tool: str | None = None
    tool_args: dict[str, Any] | None = None


# ── Паттерны ──────────────────────────────────────────────────────────────

_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_DOC_NUMBER_RE = re.compile(r"\b(?:DOC-\d{1,10}|#\d{4,10}|№\s*\d{4,10})\b", re.IGNORECASE)
_DATE_RE = re.compile(
    r"\b(\d{2})[./](\d{2})[./](\d{4})\b"
    r"|\b(\d{4})-(\d{2})-(\d{2})\b"
)
_PAGE_RE = re.compile(r"\bстраниц[аую]?\s+(\d+)\b|\bpage\s+(\d+)\b", re.IGNORECASE)
_LIMIT_RE = re.compile(
    r"\b(?:первые|покажи|top)\s+(\d+)\b|\b(\d+)\s+(?:документов|результатов|записей)\b",
    re.IGNORECASE,
)
_NAME_RE = re.compile(
    r"\b([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?)\b"
)
_DEPT_RE = re.compile(
    r"\b(?:отдел|департамент|управление|служба)\s+([А-Яа-яёЁ\s]+?)(?:\s|$|\.|,)",
    re.IGNORECASE,
)

_STATUS_MAP: dict[str, str] = {
    "черновик": "draft", "черновике": "draft", "в работе": "draft",
    "на согласовании": "review", "на проверке": "review", "ожидает согласования": "review",
    "одобрен": "approved", "согласован": "approved", "утверждён": "approved",
    "отклонён": "rejected", "отказан": "rejected", "возвращён": "rejected",
    "подписан": "signed", "подписанный": "signed",
    "в архиве": "archived", "архивный": "archived", "архивирован": "archived",
}

_DOC_TYPES: list[str] = ["договор", "приказ", "акт", "счёт", "протокол", "спецификация"]
_DOC_TYPE_VARIANTS: dict[str, str] = {
    "договоры": "договор", "договора": "договор", "договором": "договор",
    "приказы": "приказ", "приказа": "приказ",
    "акты": "акт", "актов": "акт",
    "счета": "счёт", "счетов": "счёт",
    "протоколы": "протокол", "протоколов": "протокол",
    "спецификации": "спецификация", "спецификаций": "спецификация",
}

_RELATIVE_DATES: list[tuple[re.Pattern, int, int]] = [
    (re.compile(r"\bсегодня\b", re.IGNORECASE), 0, 0),
    (re.compile(r"\bвчера\b", re.IGNORECASE), -1, -1),
    (re.compile(r"\bзавтра\b", re.IGNORECASE), 1, 1),
    (re.compile(r"\bза\s+(?:последнюю|прошлую)\s+неделю\b", re.IGNORECASE), -7, 0),
    (re.compile(r"\bза\s+последние\s+7\s+дней\b", re.IGNORECASE), -7, 0),
    (re.compile(r"\bза\s+последние\s+30\s+дней\b", re.IGNORECASE), -30, 0),
    (re.compile(r"\bза\s+последний\s+месяц\b", re.IGNORECASE), -30, 0),
    (re.compile(r"\bза\s+прошлый\s+месяц\b", re.IGNORECASE), -60, -30),
    (re.compile(r"\bза\s+(?:этот|текущий)\s+год\b", re.IGNORECASE), -365, 0),
]

_INTENT_KEYWORDS: dict[str, dict[str, list[str]]] = {
    "get_document": {
        "primary": ["покажи документ", "открой документ", "найди документ", "что за документ", "детали документа"],
        "secondary": ["покажи", "открой", "посмотри", "документ"],
    },
    "search_documents": {
        "primary": ["найди все", "поиск документов", "список документов", "покажи все договоры", "все документы"],
        "secondary": ["найди", "поищи", "список", "реестр", "сколько документов"],
    },
    "create_document": {
        "primary": ["создай документ", "создай договор", "создай приказ", "новый документ", "заведи документ"],
        "secondary": ["создай", "оформи", "зарегистрируй"],
    },
    "update_status": {
        "primary": ["измени статус", "сменить статус", "согласуй", "подпиши", "отклони", "архивируй", "отправь на согласование"],
        "secondary": ["утвердить", "перевести в статус"],
    },
    "get_history": {
        "primary": ["история документа", "журнал изменений", "кто изменял", "аудит документа", "что делали с документом"],
        "secondary": ["история", "журнал", "аудит"],
    },
    "assign_document": {
        "primary": ["назначь ответственного", "передай документ", "добавь в согласующие", "поставь на согласование", "назначить рецензента"],
        "secondary": ["назначь", "передай", "добавь исполнителя"],
    },
    "get_analytics": {
        "primary": ["статистика документов", "отчёт по документам", "аналитика", "нагрузка на отдел", "просроченные документы"],
        "secondary": ["статистика", "отчёт", "метрики", "дашборд"],
    },
    "get_workflow_status": {
        "primary": ["где застрял документ", "кто не согласовал", "статус согласования", "прогресс документа", "кто должен подписать"],
        "secondary": ["статус", "прогресс", "ожидает", "процесс"],
    },
}


# ── NLPPreprocessor ───────────────────────────────────────────────────────


class NLPPreprocessor:
    """
    NLU-препроцессор для русскоязычных запросов EDMS.

    Методы:
        preprocess(text) → NLUResult  — полный NLU-анализ запроса
    """

    def preprocess(self, text: str) -> NLUResult:
        """
        Выполняет NLU-анализ: intent, entities, normalized_query, bypass_llm.

        Args:
            text: Исходный текст запроса пользователя.

        Returns:
            NLUResult со всеми извлечёнными данными.
        """
        clean = " ".join(text.split())
        entities = self._extract_entities(clean)
        intent, confidence = self._classify_intent(clean)
        normalized = self._normalize(clean, entities)
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

    # ── Извлечение сущностей ──────────────────────────────────────────────

    def _extract_entities(self, text: str) -> ExtractedEntities:
        return ExtractedEntities(
            document_ids=self._extract_doc_ids(text),
            date_range=self._extract_date_range(text),
            statuses=self._extract_statuses(text),
            document_types=self._extract_doc_types(text),
            user_names=self._extract_names(text),
            departments=self._extract_departments(text),
            page_number=self._extract_page(text),
            limit=self._extract_limit(text),
        )

    def _extract_doc_ids(self, text: str) -> list[str]:
        ids: list[str] = []
        for m in _UUID_RE.finditer(text):
            ids.append(m.group(0).lower())
        for m in _DOC_NUMBER_RE.finditer(text):
            normalized = re.sub(r"^[#№\s]+", "DOC-", m.group(0).strip()).upper()
            ids.append(normalized)
        return ids

    def _extract_date_range(self, text: str) -> tuple[datetime, datetime] | None:
        now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        for pattern, start_off, end_off in _RELATIVE_DATES:
            if pattern.search(text):
                start = now + timedelta(days=start_off)
                end = now + timedelta(days=end_off)
                if end < start:
                    start, end = end, start
                return start, end.replace(hour=23, minute=59, second=59)

        dates: list[datetime] = []
        for m in _DATE_RE.finditer(text):
            try:
                if m.group(1):
                    day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
                else:
                    year, month, day = int(m.group(4)), int(m.group(5)), int(m.group(6))
                dates.append(datetime(year, month, day, tzinfo=timezone.utc))
            except ValueError:
                continue

        if len(dates) == 1:
            d = dates[0]
            return d, d.replace(hour=23, minute=59, second=59)
        if len(dates) >= 2:
            dates.sort()
            return dates[0], dates[-1].replace(hour=23, minute=59, second=59)
        return None

    def _extract_statuses(self, text: str) -> list[str]:
        text_lower = text.lower()
        found: list[str] = []
        for synonym, normalized in _STATUS_MAP.items():
            if synonym in text_lower and normalized not in found:
                found.append(normalized)
        return found

    def _extract_doc_types(self, text: str) -> list[str]:
        text_lower = text.lower()
        found: list[str] = []
        for doc_type in _DOC_TYPES:
            if doc_type in text_lower and doc_type not in found:
                found.append(doc_type)
        for variant, canonical in _DOC_TYPE_VARIANTS.items():
            if variant in text_lower and canonical not in found:
                found.append(canonical)
        return found

    def _extract_names(self, text: str) -> list[str]:
        names: list[str] = []
        for m in _NAME_RE.finditer(text):
            names.append(m.group(0))
        return names

    def _extract_departments(self, text: str) -> list[str]:
        depts: list[str] = []
        for m in _DEPT_RE.finditer(text):
            dept = m.group(1).strip()
            if dept and len(dept) > 2:
                depts.append(dept)
        return depts

    def _extract_page(self, text: str) -> int | None:
        m = _PAGE_RE.search(text)
        return int(m.group(1) or m.group(2)) if m else None

    def _extract_limit(self, text: str) -> int | None:
        m = _LIMIT_RE.search(text)
        return int(m.group(1) or m.group(2)) if m else None

    # ── Классификация намерения ───────────────────────────────────────────

    def _classify_intent(self, text: str) -> tuple[str, float]:
        """Определяет намерение через взвешенное совпадение ключевых слов."""
        text_lower = text.lower()
        scores: dict[str, float] = {}

        for intent, kws in _INTENT_KEYWORDS.items():
            score = 0.0
            for phrase in kws.get("primary", []):
                if phrase in text_lower:
                    score += 0.35
            for kw in kws.get("secondary", []):
                if kw in text_lower:
                    score += 0.15
            if score > 0:
                scores[intent] = min(score, 0.98)

        if text.strip().endswith("?"):
            scores["get_document"] = scores.get("get_document", 0) + 0.1

        if not scores:
            return "unknown", 0.3

        best = max(scores, key=lambda k: scores[k])
        return best, round(scores[best], 3)

    # ── Нормализация ──────────────────────────────────────────────────────

    def _normalize(self, text: str, entities: ExtractedEntities) -> str:
        """Заменяет конкретные значения плейсхолдерами для кэширования."""
        normalized = _UUID_RE.sub("{document_uuid}", text)
        normalized = _DOC_NUMBER_RE.sub("{document_id}", normalized)
        normalized = _DATE_RE.sub("{date}", normalized)
        for name in entities.user_names:
            normalized = normalized.replace(name, "{person_name}")
        return normalized.strip()

    # ── Bypass LLM ────────────────────────────────────────────────────────

    def _check_bypass(
        self,
        intent: str,
        confidence: float,
        entities: ExtractedEntities,
    ) -> tuple[bool, str | None, dict[str, Any] | None]:
        """
        Определяет возможность обхода LLM при высокой уверенности.

        Bypass при: confidence > 0.92 + все нужные сущности присутствуют.
        """
        if confidence <= 0.92:
            return False, None, None

        if intent == "get_document" and entities.document_ids:
            return True, "get_document", {
                "document_id": entities.document_ids[0],
                "include_history": False,
                "include_attachments": True,
            }

        if intent == "get_history" and entities.document_ids:
            return True, "get_document_history", {
                "document_id": entities.document_ids[0],
                "limit": entities.limit or 50,
            }

        if intent == "get_workflow_status" and entities.document_ids:
            return True, "get_workflow_status", {
                "document_id": entities.document_ids[0],
                "include_completed": False,
            }

        if intent == "search_documents":
            has_filters = any([
                entities.statuses, entities.document_types, entities.date_range,
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
                return True, "search_documents", args

        return False, None, None


# ── Синглтон ──────────────────────────────────────────────────────────────

_preprocessor: NLPPreprocessor | None = None


def get_preprocessor() -> NLPPreprocessor:
    """Возвращает синглтон NLPPreprocessor."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = NLPPreprocessor()
    return _preprocessor
