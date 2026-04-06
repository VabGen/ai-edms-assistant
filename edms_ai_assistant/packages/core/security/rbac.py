# packages/core/security/rbac.py
"""
RBAC — Role-Based Access Control для EDMS AI Assistant.

Роли (иерархия): guest < user < power_user < admin < super_admin
Принцип: deny by default — если разрешение не выдано явно, доступ запрещён.

Использование:
    from edms_ai_assistant.packages.core.security.rbac import check_access, Role

    decision = check_access("user_123", "user", "document", "read")
    if decision.allowed:
        ...
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

logger = logging.getLogger(__name__)

ResourceType = Literal["document", "dialog", "user", "system", "rag", "mcp_tool"]
Action = Literal["read", "write", "delete", "execute", "admin"]


class Role(str, Enum):
    GUEST = "guest"
    USER = "user"
    POWER_USER = "power_user"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

    @property
    def level(self) -> int:
        return {"guest": 0, "user": 1, "power_user": 2, "admin": 3, "super_admin": 4}[
            self.value
        ]

    def has_at_least(self, other: "Role") -> bool:
        return self.level >= other.level

    @classmethod
    def from_string(cls, value: str) -> "Role":
        try:
            return cls(value.lower())
        except ValueError:
            return cls.USER


# Матрица разрешений: {role: {resource: {action: bool}}}
_PERMISSIONS: dict[str, dict[str, set[str]]] = {
    Role.GUEST: {
        "system": {"read"},
    },
    Role.USER: {
        "document": {"read"},
        "dialog": {"read", "write"},
        "rag": {"read"},
        "mcp_tool": {"execute"},
    },
    Role.POWER_USER: {
        "document": {"read", "write"},
        "dialog": {"read", "write", "delete"},
        "rag": {"read", "write"},
        "mcp_tool": {"execute"},
        "user": {"read"},
    },
    Role.ADMIN: {
        "document": {"read", "write", "delete"},
        "dialog": {"read", "write", "delete"},
        "rag": {"read", "write", "delete", "execute"},
        "mcp_tool": {"execute", "admin"},
        "user": {"read", "write"},
        "system": {"read", "write"},
    },
    Role.SUPER_ADMIN: {
        "document": {"read", "write", "delete", "admin"},
        "dialog": {"read", "write", "delete", "admin"},
        "rag": {"read", "write", "delete", "execute", "admin"},
        "mcp_tool": {"read", "write", "delete", "execute", "admin"},
        "user": {"read", "write", "delete", "admin"},
        "system": {"read", "write", "delete", "admin"},
    },
}


@dataclass
class AccessDecision:
    """Результат проверки доступа."""

    allowed: bool
    user_id: str
    role: Role
    resource_type: ResourceType
    action: Action
    reason: str
    resource_id: str | None = None


def check_access(
    user_id: str,
    role: Role | str,
    resource_type: ResourceType,
    action: Action,
    resource_id: str | None = None,
) -> AccessDecision:
    """
    Проверяет права доступа.

    Args:
        user_id:       Идентификатор пользователя.
        role:          Роль (Role enum или строка).
        resource_type: Тип ресурса.
        action:        Действие.
        resource_id:   ID конкретного ресурса (опционально).

    Returns:
        AccessDecision с результатом проверки.
    """
    if isinstance(role, str):
        role = Role.from_string(role)

    role_perms = _PERMISSIONS.get(role, {})
    resource_perms = role_perms.get(resource_type, set())
    allowed = action in resource_perms

    reason = "Доступ разрешён" if allowed else f"Недостаточно прав: {resource_type}:{action}"
    if not allowed:
        logger.warning(
            "Access denied: user=%s role=%s resource=%s action=%s",
            user_id, role.value, resource_type, action,
        )

    return AccessDecision(
        allowed=allowed,
        user_id=user_id,
        role=role,
        resource_type=resource_type,
        action=action,
        reason=reason,
        resource_id=resource_id,
    )
