# orchestrator/utils/format_utils.py
"""Утилиты форматирования дат, чисел, текста для ответов агента."""
from __future__ import annotations

from datetime import datetime, timezone


def format_datetime_ru(dt: datetime | str | None) -> str:
    """Форматирует datetime в русском формате: 01 января 2025, 14:30."""
    if dt is None:
        return "—"
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except ValueError:
            return dt
    _MONTHS = [
        "", "января", "февраля", "марта", "апреля", "мая", "июня",
        "июля", "августа", "сентября", "октября", "ноября", "декабря",
    ]
    local = dt.astimezone(timezone.utc)
    return f"{local.day:02d} {_MONTHS[local.month]} {local.year}, {local.hour:02d}:{local.minute:02d}"


def truncate(text: str, max_len: int = 200, suffix: str = "...") -> str:
    """Обрезает текст до max_len символов."""
    if len(text) <= max_len:
        return text
    return text[: max_len - len(suffix)] + suffix


def pluralize_ru(n: int, one: str, few: str, many: str) -> str:
    """Склоняет существительное по числу (1 документ, 2 документа, 5 документов)."""
    n_abs = abs(n) % 100
    if 11 <= n_abs <= 19:
        return f"{n} {many}"
    r = n_abs % 10
    if r == 1:
        return f"{n} {one}"
    if 2 <= r <= 4:
        return f"{n} {few}"
    return f"{n} {many}"
