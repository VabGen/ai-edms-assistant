# edms_ai_assistant/tools/doc_search.py
"""
EDMS AI Assistant — Document Search Tool.

Поиск документов в EDMS по широкому набору параметров фильтрации.

Маппинг параметров инструмента → поля DocumentFilter (resources_openapi.py):
    short_summary                   → DocumentFilter.shortSummary
    reg_number                      → DocumentFilter.regNumber
    out_reg_number                  → DocumentFilter.outRegNumber
    doc_category                    → DocumentFilter.categoryConstants      (list[DocCategory])
    status                          → DocumentFilter.status                 (list[Status2])
    date_from                       → DocumentFilter.dateRegStart           (datetime ISO 8601)
    date_to                         → DocumentFilter.dateRegEnd             (datetime ISO 8601)
    date_control_start              → DocumentFilter.dateControlStart       (datetime ISO 8601)
    date_control_end                → DocumentFilter.dateControlEnd         (datetime ISO 8601)
    author_last_name                → DocumentFilter.authorLastName
    correspondent_name              → DocumentFilter.correspondentName
    recipient_name                  → DocumentFilter.recipientName
    task_executor_last_name         → DocumentFilter.taskExecutorLastName
    author_current_user             → DocumentFilter.authorCurrentUser      (bool)
    process_executor_current_user   → DocumentFilter.processExecutorCurrentUser (bool)
    task_executor_current_user      → DocumentFilter.taskExecutorCurrentUser (bool)
    control_user_current_user       → DocumentFilter.controlUserCurrentUser (bool)
    introduction_current_user       → DocumentFilter.introductionCurrentUser (bool)
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator, model_validator

from edms_ai_assistant.clients.document_client import DocumentClient

logger = logging.getLogger(__name__)

# Максимальное количество документов в одном ответе агенту
_MAX_RESULTS: int = 10

# Допустимые значения DocCategory из resources_openapi.py
_VALID_CATEGORIES: frozenset[str] = frozenset(
    {
        "INTERN",
        "INCOMING",
        "OUTGOING",
        "MEETING",
        "QUESTION",
        "MEETING_QUESTION",
        "APPEAL",
        "CONTRACT",
        "CUSTOM",
    }
)

# Допустимые значения Status2 из resources_openapi.py
_VALID_STATUSES: frozenset[str] = frozenset(
    {
        "FORMING",
        "REGISTRATION",
        "IN_PROGRESS",
        "COMPLETED",
        "CANCELLED",
        "ARCHIVE",
        "ON_SIGNING",
        "ON_AGREEMENT",
        "ON_REVIEW",
        "ON_STATEMENT",
    }
)

# Шаблон ISO-даты для валидации полей date_*
_DATE_PATTERN: str = r"^\d{4}-\d{2}-\d{2}$|^$"


class DocSearchInput(BaseModel):
    """Validated input schema for the document search tool.

    Покрывает наиболее востребованные агентом поля DocumentFilter.
    Поля сгруппированы по смысловым блокам:
      - Идентификация документа (номера, категория, статус)
      - Даты регистрации и контроля
      - Участники (автор, корреспондент, адресат, исполнитель поручения)
      - Флаги текущего пользователя
    """

    token: str = Field(..., description="JWT токен авторизации пользователя")

    # ── Идентификация документа ───────────────────────────────────────────────

    short_summary: str | None = Field(
        None,
        max_length=500,
        description=(
            "Поиск по краткому содержанию документа. "
            "Пример: 'договор с ООО Альфа', 'обращение по вопросу аренды'."
        ),
    )
    reg_number: str | None = Field(
        None,
        max_length=100,
        description="Регистрационный номер (входящий). Пример: 'ВХ-2026-001'.",
    )
    out_reg_number: str | None = Field(
        None,
        max_length=100,
        description="Исходящий регистрационный номер. Пример: 'ИСХ-2025-042'.",
    )
    doc_category: str | None = Field(
        None,
        description=(
            "Категория документа. Допустимые значения: "
            "INCOMING (входящие), OUTGOING (исходящие), INTERN (внутренние), "
            "APPEAL (обращения граждан), CONTRACT (договоры), "
            "MEETING (совещания), QUESTION (вопросы повестки), "
            "MEETING_QUESTION (вопросы заседания), CUSTOM (произвольные)."
        ),
    )
    status: list[str] | None = Field(
        None,
        description=(
            "Статусы документов для фильтрации (список). Допустимые значения: "
            "FORMING, REGISTRATION, IN_PROGRESS, COMPLETED, CANCELLED, ARCHIVE, "
            "ON_SIGNING, ON_AGREEMENT, ON_REVIEW, ON_STATEMENT."
        ),
    )

    # ── Даты регистрации ──────────────────────────────────────────────────────

    date_from: str | None = Field(
        None,
        description="Начало диапазона дат регистрации (включительно). Формат: YYYY-MM-DD.",
        pattern=_DATE_PATTERN,
    )
    date_to: str | None = Field(
        None,
        description="Конец диапазона дат регистрации (включительно). Формат: YYYY-MM-DD.",
        pattern=_DATE_PATTERN,
    )

    # ── Даты контроля ─────────────────────────────────────────────────────────

    date_control_start: str | None = Field(
        None,
        description=(
            "Начало диапазона дат постановки на контроль. Формат: YYYY-MM-DD. "
            "Используй когда пользователь спрашивает о документах на контроле за период."
        ),
        pattern=_DATE_PATTERN,
    )
    date_control_end: str | None = Field(
        None,
        description="Конец диапазона дат контроля. Формат: YYYY-MM-DD.",
        pattern=_DATE_PATTERN,
    )

    # ── Участники документа ───────────────────────────────────────────────────

    author_last_name: str | None = Field(
        None,
        max_length=150,
        description=(
            "Фамилия автора документа. "
            "Пример: 'Иванов'. Используй когда ищут документы конкретного сотрудника."
        ),
    )
    correspondent_name: str | None = Field(
        None,
        max_length=300,
        description=(
            "Наименование организации-корреспондента (отправитель/получатель). "
            "Пример: 'ООО Альфа', 'Министерство финансов'."
        ),
    )
    recipient_name: str | None = Field(
        None,
        max_length=300,
        description=(
            "Наименование адресата документа. " "Пример: 'Акимат города', 'ТОО Бета'."
        ),
    )
    task_executor_last_name: str | None = Field(
        None,
        max_length=150,
        description=(
            "Фамилия исполнителя поручения по документу. "
            "Используй когда ищут документы с поручением на конкретного сотрудника."
        ),
    )

    # ── Флаги текущего пользователя ──────────────────────────────────────────

    author_current_user: bool | None = Field(
        None,
        description=(
            "True — только документы, где автор = текущий пользователь. "
            "Используй для: 'мои документы', 'документы которые я создал'."
        ),
    )
    process_executor_current_user: bool | None = Field(
        None,
        description=(
            "True — документы, где текущий пользователь участник активного процесса. "
            "Используй для: 'документы на моём рассмотрении', 'что мне нужно обработать'."
        ),
    )
    task_executor_current_user: bool | None = Field(
        None,
        description=(
            "True — документы, по которым текущий пользователь является исполнителем поручения. "
            "Используй для: 'мои поручения', 'документы по которым я исполнитель'."
        ),
    )
    control_user_current_user: bool | None = Field(
        None,
        description=(
            "True — документы, где текущий пользователь является контролёром. "
            "Используй для: 'документы на моём контроле'."
        ),
    )
    introduction_current_user: bool | None = Field(
        None,
        description=(
            "True — документы на ознакомлении у текущего пользователя. "
            "Используй для: 'документы для ознакомления', 'что мне нужно ознакомиться'."
        ),
    )

    # ── Валидаторы ────────────────────────────────────────────────────────────

    @field_validator(
        "short_summary",
        "reg_number",
        "out_reg_number",
        "author_last_name",
        "correspondent_name",
        "recipient_name",
        "task_executor_last_name",
        mode="before",
    )
    @classmethod
    def strip_and_none(cls, v: str | None) -> str | None:
        """Strips surrounding whitespace; converts empty string to None."""
        if v is None:
            return None
        stripped = v.strip()
        return stripped if stripped else None

    @field_validator("doc_category", mode="before")
    @classmethod
    def validate_category(cls, v: str | None) -> str | None:
        """Uppercases and validates category against DocCategory enum.

        Raises:
            ValueError: If the value is not a known DocCategory constant.
        """
        if v is None:
            return None
        upper = v.strip().upper()
        if not upper:
            return None
        if upper not in _VALID_CATEGORIES:
            raise ValueError(
                f"Неизвестная категория документа: '{upper}'. "
                f"Допустимые значения: {sorted(_VALID_CATEGORIES)}"
            )
        return upper

    @field_validator("status", mode="before")
    @classmethod
    def validate_statuses(cls, v: list[str] | None) -> list[str] | None:
        """Uppercases and validates each status value against Status2 enum.

        Raises:
            ValueError: If any value is not a known Status2 constant.
        """
        if not v:
            return None
        result: list[str] = []
        for item in v:
            upper = item.strip().upper()
            if upper not in _VALID_STATUSES:
                raise ValueError(
                    f"Неизвестный статус документа: '{upper}'. "
                    f"Допустимые значения: {sorted(_VALID_STATUSES)}"
                )
            result.append(upper)
        return result if result else None

    @model_validator(mode="after")
    def at_least_one_search_param(self) -> DocSearchInput:
        """Ensures that at least one meaningful filter parameter is provided.

        token не считается фильтром — проверяем все остальные поля.

        Raises:
            ValueError: If all filter fields are None.
        """
        filter_fields = [
            self.short_summary,
            self.reg_number,
            self.out_reg_number,
            self.doc_category,
            self.status,
            self.date_from,
            self.date_to,
            self.date_control_start,
            self.date_control_end,
            self.author_last_name,
            self.correspondent_name,
            self.recipient_name,
            self.task_executor_last_name,
            self.author_current_user,
            self.process_executor_current_user,
            self.task_executor_current_user,
            self.control_user_current_user,
            self.introduction_current_user,
        ]
        if not any(v is not None for v in filter_fields):
            raise ValueError(
                "Укажите хотя бы один параметр поиска: краткое содержание, "
                "номер документа, категорию, статус, диапазон дат, "
                "участника документа или флаг текущего пользователя."
            )
        return self


@tool("doc_search_tool", args_schema=DocSearchInput)
async def doc_search_tool(
    token: str,
    short_summary: str | None = None,
    reg_number: str | None = None,
    out_reg_number: str | None = None,
    doc_category: str | None = None,
    status: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    date_control_start: str | None = None,
    date_control_end: str | None = None,
    author_last_name: str | None = None,
    correspondent_name: str | None = None,
    recipient_name: str | None = None,
    task_executor_last_name: str | None = None,
    author_current_user: bool | None = None,
    process_executor_current_user: bool | None = None,
    task_executor_current_user: bool | None = None,
    control_user_current_user: bool | None = None,
    introduction_current_user: bool | None = None,
) -> dict[str, Any]:
    """
    Searches documents in EDMS by a wide range of filter criteria.

    Use when the user asks:
    - «Найди договоры с ООО Альфа»                → correspondent_name + doc_category=CONTRACT
    - «Покажи входящие документы за март 2026»    → doc_category=INCOMING + date_from/date_to
    - «Есть ли документ ВХ-2026-001?»             → reg_number
    - «Обращения граждан за последний месяц»      → doc_category=APPEAL + date_from/date_to
    - «Мои документы на согласовании»             → process_executor_current_user=True
    - «Документы, которые я создал в этом году»  → author_current_user=True + date_from
    - «Документы на моём контроле»                → control_user_current_user=True
    - «Поручения исполнителя Иванова»             → task_executor_last_name='Иванов'
    - «Документы на ознакомлении у меня»         → introduction_current_user=True
    - «Входящие со статусом В работе за январь»  → doc_category + status + date range

    Returns up to 10 documents with: id, registration number, date,
    short summary, category, author, and status.
    """
    # Формируем DocumentFilter согласно полной модели из resources_openapi.py
    doc_filter: dict[str, Any] = {}

    # ── Идентификация документа ───────────────────────────────────────────────
    if short_summary:
        doc_filter["shortSummary"] = short_summary
    if reg_number:
        doc_filter["regNumber"] = reg_number
    if out_reg_number:
        doc_filter["outRegNumber"] = out_reg_number
    if doc_category:
        # categoryConstants принимает list[DocCategory]
        doc_filter["categoryConstants"] = [doc_category]
    if status:
        # status принимает list[Status2]
        doc_filter["status"] = status

    # ── Даты регистрации ──────────────────────────────────────────────────────
    if date_from:
        doc_filter["dateRegStart"] = _to_iso_start(date_from)
    if date_to:
        doc_filter["dateRegEnd"] = _to_iso_end(date_to)

    # ── Даты контроля ─────────────────────────────────────────────────────────
    if date_control_start:
        doc_filter["dateControlStart"] = _to_iso_start(date_control_start)
    if date_control_end:
        doc_filter["dateControlEnd"] = _to_iso_end(date_control_end)

    # ── Участники документа ───────────────────────────────────────────────────
    if author_last_name:
        doc_filter["authorLastName"] = author_last_name
    if correspondent_name:
        doc_filter["correspondentName"] = correspondent_name
    if recipient_name:
        doc_filter["recipientName"] = recipient_name
    if task_executor_last_name:
        doc_filter["taskExecutorLastName"] = task_executor_last_name

    # ── Флаги текущего пользователя ──────────────────────────────────────────
    # Передаём только True: False не сужает выборку в этой API-семантике
    if author_current_user is True:
        doc_filter["authorCurrentUser"] = True
    if process_executor_current_user is True:
        doc_filter["processExecutorCurrentUser"] = True
    if task_executor_current_user is True:
        doc_filter["taskExecutorCurrentUser"] = True
    if control_user_current_user is True:
        doc_filter["controlUserCurrentUser"] = True
    if introduction_current_user is True:
        doc_filter["introductionCurrentUser"] = True

    pageable: dict[str, Any] = {"page": 0, "size": _MAX_RESULTS}

    logger.info(
        "Document search requested",
        extra={"filter_keys": list(doc_filter.keys())},
    )

    try:
        async with DocumentClient() as client:
            raw_docs = await client.search_documents(
                token=token,
                doc_filter=doc_filter,
                pageable=pageable,
            )

        if not raw_docs:
            return {
                "status": "success",
                "message": "По вашему запросу документы не найдены.",
                "documents": [],
                "total": 0,
            }

        documents: list[dict[str, Any]] = [
            _serialize_document(d) for d in raw_docs[:_MAX_RESULTS]
        ]

        logger.info(
            "Document search completed",
            extra={"found": len(documents), "filter_keys": list(doc_filter.keys())},
        )

        return {
            "status": "success",
            "total": len(documents),
            "documents": documents,
            "message": f"Найдено {len(documents)} документ(ов).",
        }

    except Exception as exc:
        logger.error(
            "Document search failed",
            exc_info=True,
            extra={"filter_keys": list(doc_filter.keys())},
        )
        return {
            "status": "error",
            "message": f"Ошибка поиска документов: {exc}",
        }


# ── Вспомогательные функции ───────────────────────────────────────────────────


from datetime import datetime, time, timezone

def _to_iso_start(date_str: str) -> str:
    dt = datetime.fromisoformat(date_str)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%S.000Z")

def _to_iso_end(date_str: str) -> str:
    dt = datetime.fromisoformat(date_str)
    return dt.replace(hour=23, minute=59, second=59, microsecond=999).strftime("%Y-%m-%dT%H:%M:%S.999Z")

# def _to_iso_start(date_str: str) -> str:
#     """Converts a YYYY-MM-DD string to start-of-day ISO 8601 datetime string.
#
#     Args:
#         date_str: Date string in YYYY-MM-DD format.
#
#     Returns:
#         ISO 8601 datetime string, e.g. '2026-03-01T00:00:00'.
#     """
#     return f"{date_str}T00:00:00"
#
#
# def _to_iso_end(date_str: str) -> str:
#     """Converts a YYYY-MM-DD string to end-of-day ISO 8601 datetime string.
#
#     Args:
#         date_str: Date string in YYYY-MM-DD format.
#
#     Returns:
#         ISO 8601 datetime string, e.g. '2026-03-31T23:59:59'.
#     """
#     return f"{date_str}T23:59:59"


def _serialize_document(d: dict[str, Any]) -> dict[str, Any]:
    """Converts a raw DocumentDto dict into a compact agent-friendly representation.

    Keeps only fields relevant for the user-facing response.
    Truncates shortSummary to 200 chars to preserve context window.

    Args:
        d: Raw DocumentDto dict from EDMS API.

    Returns:
        Compact dict with: id, reg_number, reg_date, category,
        short_summary, author, status.
    """
    return {
        "id": str(d.get("id", "")),
        "reg_number": d.get("regNumber") or d.get("reservedRegNumber") or "—",
        "reg_date": _extract_date(d.get("regDate")),
        "category": str(d.get("docCategoryConstant", "—")),
        "short_summary": (d.get("shortSummary") or "")[:200],
        "author": _format_author(d.get("author")),
        "status": str(d.get("status", "—")),
    }


def _extract_date(raw: Any) -> str:
    """Extracts the date part (YYYY-MM-DD) from an ISO datetime value.

    Args:
        raw: Raw date value from API response (str, datetime, or None).

    Returns:
        'YYYY-MM-DD' string, or '—' if absent or unparseable.
    """
    if not raw:
        return "—"
    return str(raw)[:10]


def _format_author(author: dict[str, Any] | None) -> str:
    """Formats an author dict into a human-readable full name string.

    Args:
        author: Dict with optional keys: lastName, firstName, middleName.

    Returns:
        Space-joined full name string, or '—' if author is absent.
    """
    if not author:
        return "—"
    parts = [
        author.get("lastName", ""),
        author.get("firstName", ""),
        author.get("middleName", ""),
    ]
    return " ".join(p for p in parts if p).strip() or "—"
