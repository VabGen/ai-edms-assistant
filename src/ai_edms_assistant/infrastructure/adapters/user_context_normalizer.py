# src/ai_edms_assistant/infrastructure/adapters/user_context_normalizer.py
"""UserContextNormalizer — стандартизирует ключи user_context перед передачей в агент.
"""

from __future__ import annotations

from typing import Any


class UserContextNormalizer:
    """Normalises user_context dict to canonical camelCase keys.

    Stateless — all methods are static. Safe for concurrent use.

    Canonical output keys (always present after normalization):
        ``firstName``     — first name (Имя)
        ``lastName``      — last name (Фамилия), may be None
        ``middleName``    — patronymic (Отчество), may be None
        ``departmentName``— department display name, may be None
        ``postName``      — job title, may be None

    Example:
        >>> raw = {"first_name": "Иван", "authorDepartmentName": "Бухгалтерия"}
        >>> UserContextNormalizer.normalize(raw)
        {"firstName": "Иван", "lastName": None, ..., "departmentName": "Бухгалтерия", ...}
    """

    # ── Маппинг альтернативных ключей → канонический ──────────────────────────
    _FIELD_ALIASES: dict[str, tuple[str, ...]] = {
        "firstName": ("firstName", "first_name", "name"),
        "lastName": ("lastName", "last_name"),
        "middleName": ("middleName", "middle_name"),
        "departmentName": (
            "departmentName",
            "department_name",
            "authorDepartmentName",
        ),
        "postName": (
            "postName",
            "post_name",
            "authorPost",
            "full_post_name",
        ),
    }

    @classmethod
    def normalize(cls, raw: dict[str, Any] | Any | None) -> dict[str, Any]:
        """Normalize user_context to canonical camelCase keys.

        Accepts both plain ``dict`` and Pydantic models (via ``model_dump``).
        Returns a new dict — original is never mutated.

        Args:
            raw: Raw user_context (dict, Pydantic model, or None).

        Returns:
            Normalized dict with canonical keys. Unknown keys are preserved
            as-is (pass-through) to avoid data loss.
        """
        if raw is None:
            return {}

        # Pydantic model → dict
        if hasattr(raw, "model_dump"):
            ctx: dict[str, Any] = raw.model_dump(exclude_none=True)
        elif isinstance(raw, dict):
            ctx = dict(raw)
        else:
            return {}

        result: dict[str, Any] = {}

        # ── Нормализация известных полей ──────────────────────────────────────
        for canonical_key, aliases in cls._FIELD_ALIASES.items():
            value: Any = None
            for alias in aliases:
                candidate = ctx.get(alias)
                if candidate is not None:
                    value = candidate
                    break
            result[canonical_key] = value

        # ── Pass-through: неизвестные ключи сохраняем ────────────────────────
        known_aliases: set[str] = {
            alias
            for aliases in cls._FIELD_ALIASES.values()
            for alias in aliases
        }
        for key, val in ctx.items():
            if key not in known_aliases and key not in result:
                result[key] = val

        return result

    @classmethod
    def extract_display_name(cls, ctx: dict[str, Any]) -> str:
        """Extract display first name from normalized context.

        Args:
            ctx: Already-normalized user_context dict.

        Returns:
            First name string or fallback "пользователь".
        """
        name = ctx.get("firstName") or "пользователь"
        return str(name).strip()