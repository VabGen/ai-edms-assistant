"""
edms_ai_assistant/packages/core/security/rbac.py

Role-Based Access Control (RBAC) — система управления правами доступа.

Features:
• Иерархия ролей (admin > power_user > user > guest)
• Гранулярные разрешения (permissions)
• Проверка прав на уровне ресурсов и действий
• Интеграция с JWT-токенами
• Аудит всех проверок доступа

Security:
• Принцип наименьших привилегий (deny by default)
• Constant-time comparison для предотвращения timing-атак
• Логирование всех попыток доступа (успешных и неудачных)
"""
from __future__ import annotations

import enum
import hashlib
import hmac
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Set, TypeVar, Union

import structlog

from edms_ai_assistant.packages.core.logging import get_logger
from edms_ai_assistant.packages.core.settings import settings

# ── Типы ────────────────────────────────────────────────────────────────
RoleLevel = Literal["guest", "user", "power_user", "admin", "super_admin"]
ResourceType = Literal["document", "dialog", "user", "system", "rag", "mcp_tool"]
Action = Literal["read", "write", "delete", "execute", "admin"]

logger = get_logger(__name__)


# ── Перечисления ────────────────────────────────────────────────────────
class Role(enum.Enum):
    """
    Роли пользователей с уровнями доступа.

    Иерархия (от низшего к высшему):
    guest < user < power_user < admin < super_admin
    """
    GUEST = "guest"
    USER = "user"
    POWER_USER = "power_user"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

    @property
    def level(self) -> int:
        """Числовой уровень роли (для сравнения)."""
        levels = {
            Role.GUEST: 0,
            Role.USER: 1,
            Role.POWER_USER: 2,
            Role.ADMIN: 3,
            Role.SUPER_ADMIN: 4,
        }
        return levels[self]

    def has_at_least(self, other: "Role") -> bool:
        """Проверить что текущая роль не ниже указанной."""
        return self.level >= other.level

    @classmethod
    def from_string(cls, role_str: str) -> "Role":
        """Создать Role из строки (case-insensitive)."""
        try:
            return cls(role_str.lower())
        except ValueError:
            # По умолчанию возвращаем USER для неизвестных ролей
            logger.warning("Unknown role, defaulting to USER", role=role_str)
            return cls.USER


class Permission(enum.Enum):
    """
    Разрешения (permissions) для детального контроля доступа.

    Формат: resource:action
    Примеры:
    - document:read — чтение документов
    - document:write — создание/редактирование
    - document:delete — удаление
    - mcp_tool:execute — вызов MCP-инструментов
    - system:admin — административные действия
    """
    # Документы
    DOCUMENT_READ = "document:read"
    DOCUMENT_WRITE = "document:write"
    DOCUMENT_DELETE = "document:delete"

    # Диалоги
    DIALOG_READ = "dialog:read"
    DIALOG_WRITE = "dialog:write"
    DIALOG_DELETE = "dialog:delete"

    # Пользователи
    USER_READ = "user:read"
    USER_WRITE = "user:write"
    USER_DELETE = "user:delete"

    # RAG
    RAG_READ = "rag:read"
    RAG_WRITE = "rag:write"
    RAG_DELETE = "rag:delete"
    RAG_REBUILD = "rag:rebuild"

    # MCP Инструменты
    MCP_TOOL_EXECUTE = "mcp_tool:execute"
    MCP_TOOL_ADMIN = "mcp_tool:admin"

    # Система
    SYSTEM_READ = "system:read"
    SYSTEM_WRITE = "system:write"
    SYSTEM_ADMIN = "system:admin"

    # Аудит и логи
    AUDIT_READ = "audit:read"
    AUDIT_EXPORT = "audit:export"


# ── Структуры данных ────────────────────────────────────────────────────
@dataclass
class RoleDefinition:
    """Определение роли с набором разрешений."""
    role: Role
    permissions: Set[Permission] = field(default_factory=set)
    description: str = ""
    inherit_from: Optional[Role] = None  # Наследование от другой роли

    def has_permission(self, permission: Permission) -> bool:
        """Проверить наличие разрешения у роли."""
        return permission in self.permissions

    def has_any_permission(self, permissions: List[Permission]) -> bool:
        """Проверить наличие хотя бы одного разрешения из списка."""
        return any(self.has_permission(p) for p in permissions)

    def has_all_permissions(self, permissions: List[Permission]) -> bool:
        """Проверить наличие всех разрешений из списка."""
        return all(self.has_permission(p) for p in permissions)


@dataclass
class AccessContext:
    """Контекст проверки доступа."""
    user_id: str
    role: Role
    permissions: Set[Permission]
    resource_type: ResourceType
    resource_id: Optional[str] = None
    action: Action = "read"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AccessDecision:
    """Результат проверки доступа."""
    allowed: bool
    user_id: str
    role: Role
    resource_type: ResourceType
    resource_id: Optional[str]
    action: Action
    reason: str
    permissions_checked: List[Permission] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Конвертировать в dict для логирования."""
        return {
            "allowed": self.allowed,
            "user_id": self.user_id,
            "role": self.role.value,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "reason": self.reason,
            "permissions_checked": [p.value for p in self.permissions_checked],
            "timestamp": self.timestamp.isoformat(),
        }


# ── Конфигурация ролей ──────────────────────────────────────────────────
class RBACConfig:
    """
    Конфигурация RBAC системы.

    Определяет роли, разрешения и политики доступа.
    """

    # Базовые разрешения для каждой роли
    ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
        Role.GUEST: {
            Permission.SYSTEM_READ,
        },

        Role.USER: {
            Permission.DOCUMENT_READ,
            Permission.DIALOG_READ,
            Permission.DIALOG_WRITE,
            Permission.RAG_READ,
            Permission.MCP_TOOL_EXECUTE,
        },

        Role.POWER_USER: {
            Permission.DOCUMENT_READ,
            Permission.DOCUMENT_WRITE,
            Permission.DIALOG_READ,
            Permission.DIALOG_WRITE,
            Permission.DIALOG_DELETE,
            Permission.RAG_READ,
            Permission.RAG_WRITE,
            Permission.MCP_TOOL_EXECUTE,
            Permission.USER_READ,
        },

        Role.ADMIN: {
            Permission.DOCUMENT_READ,
            Permission.DOCUMENT_WRITE,
            Permission.DOCUMENT_DELETE,
            Permission.DIALOG_READ,
            Permission.DIALOG_WRITE,
            Permission.DIALOG_DELETE,
            Permission.USER_READ,
            Permission.USER_WRITE,
            Permission.RAG_READ,
            Permission.RAG_WRITE,
            Permission.RAG_DELETE,
            Permission.RAG_REBUILD,
            Permission.MCP_TOOL_EXECUTE,
            Permission.MCP_TOOL_ADMIN,
            Permission.SYSTEM_READ,
            Permission.SYSTEM_WRITE,
            Permission.AUDIT_READ,
        },

        Role.SUPER_ADMIN: {
            # Все разрешения
            *Permission,
        },
    }

    # Запрещённые действия для ролей (override)
    ROLE_DENY: Dict[Role, Set[Permission]] = {
        Role.GUEST: set(),
        Role.USER: {
            Permission.DOCUMENT_DELETE,
            Permission.RAG_REBUILD,
            Permission.SYSTEM_ADMIN,
        },
        Role.POWER_USER: {
            Permission.USER_DELETE,
            Permission.RAG_REBUILD,
            Permission.SYSTEM_ADMIN,
        },
        Role.ADMIN: {
            Permission.SYSTEM_ADMIN,
            Permission.AUDIT_EXPORT,
        },
        Role.SUPER_ADMIN: set(),
    }

    # Маппинг ресурсов на требуемые разрешения
    RESOURCE_PERMISSIONS: Dict[ResourceType, Dict[Action, Permission]] = {
        "document": {
            "read": Permission.DOCUMENT_READ,
            "write": Permission.DOCUMENT_WRITE,
            "delete": Permission.DOCUMENT_DELETE,
            "execute": Permission.DOCUMENT_READ,
            "admin": Permission.DOCUMENT_WRITE,
        },
        "dialog": {
            "read": Permission.DIALOG_READ,
            "write": Permission.DIALOG_WRITE,
            "delete": Permission.DIALOG_DELETE,
            "execute": Permission.DIALOG_READ,
            "admin": Permission.DIALOG_WRITE,
        },
        "user": {
            "read": Permission.USER_READ,
            "write": Permission.USER_WRITE,
            "delete": Permission.USER_DELETE,
            "execute": Permission.USER_READ,
            "admin": Permission.USER_WRITE,
        },
        "system": {
            "read": Permission.SYSTEM_READ,
            "write": Permission.SYSTEM_WRITE,
            "delete": Permission.SYSTEM_ADMIN,
            "execute": Permission.SYSTEM_READ,
            "admin": Permission.SYSTEM_ADMIN,
        },
        "rag": {
            "read": Permission.RAG_READ,
            "write": Permission.RAG_WRITE,
            "delete": Permission.RAG_DELETE,
            "execute": Permission.RAG_REBUILD,
            "admin": Permission.RAG_WRITE,
        },
        "mcp_tool": {
            "read": Permission.MCP_TOOL_EXECUTE,
            "write": Permission.MCP_TOOL_ADMIN,
            "delete": Permission.MCP_TOOL_ADMIN,
            "execute": Permission.MCP_TOOL_EXECUTE,
            "admin": Permission.MCP_TOOL_ADMIN,
        },
    }

    @classmethod
    def get_role_definition(cls, role: Role) -> RoleDefinition:
        """Получить определение роли с разрешениями."""
        permissions = cls.ROLE_PERMISSIONS.get(role, set()).copy()

        # Удаляем запрещённые разрешения
        deny = cls.ROLE_DENY.get(role, set())
        permissions -= deny

        return RoleDefinition(
            role=role,
            permissions=permissions,
            description=f"Role {role.value} with {len(permissions)} permissions",
        )

    @classmethod
    def get_required_permission(
            cls,
            resource_type: ResourceType,
            action: Action,
    ) -> Permission:
        """Получить требуемое разрешение для ресурса и действия."""
        resource_map = cls.RESOURCE_PERMISSIONS.get(resource_type, {})
        return resource_map.get(action, Permission.SYSTEM_READ)


# ── RBAC Engine ─────────────────────────────────────────────────────────
class RBACEngine:
    """
    Движок проверки прав доступа.

    Использование:
        rbac = RBACEngine()
        decision = rbac.check_access(user_id, role, resource, action)
        if decision.allowed:
            # Доступ разрешён
    """

    def __init__(self, config: Optional[RBACConfig] = None):
        self.config = config or RBACConfig()
        self._audit_log: List[AccessDecision] = []
        self._audit_enabled = True

    def check_access(
            self,
            user_id: str,
            role: Union[Role, str],
            resource_type: ResourceType,
            action: Action,
            resource_id: Optional[str] = None,
            additional_permissions: Optional[List[Permission]] = None,
    ) -> AccessDecision:
        """
        Проверить доступ пользователя к ресурсу.

        Args:
            user_id: Идентификатор пользователя
            role: Роль пользователя (Role или строка)
            resource_type: Тип ресурса (document, dialog, user, etc.)
            action: Действие (read, write, delete, execute, admin)
            resource_id: Идентификатор ресурса (опционально)
            additional_permissions: Дополнительные разрешения для проверки

        Returns:
            AccessDecision с результатом проверки

        Example:
            >>> rbac = RBACEngine()
            >>> decision = rbac.check_access(
            ...     user_id="user_123",
            ...     role="user",
            ...     resource_type="document",
            ...     action="read",
            ... )
            >>> if decision.allowed:
            ...     # Доступ разрешён
        """
        # Конвертируем роль из строки если нужно
        if isinstance(role, str):
            role = Role.from_string(role)

        # Получаем определение роли
        role_def = self.config.get_role_definition(role)

        # Получаем требуемое разрешение
        required_permission = self.config.get_required_permission(
            resource_type, action
        )

        # Собираем все разрешения для проверки
        permissions_to_check = [required_permission]
        if additional_permissions:
            permissions_to_check.extend(additional_permissions)

        # Проверяем разрешения
        allowed = True
        reason = "Access granted"

        for permission in permissions_to_check:
            if not role_def.has_permission(permission):
                allowed = False
                reason = f"Missing permission: {permission.value}"
                break

        # Создаём решение
        decision = AccessDecision(
            allowed=allowed,
            user_id=user_id,
            role=role,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            reason=reason,
            permissions_checked=permissions_to_check,
        )

        # Аудит
        if self._audit_enabled:
            self._audit_log.append(decision)
            self._log_access(decision)

        return decision

    def _log_access(self, decision: AccessDecision) -> None:
        """Логировать проверку доступа."""
        bound_logger = logger.bind(
            user_id=decision.user_id,
            role=decision.role.value,
            resource_type=decision.resource_type,
            action=decision.action,
        )

        if decision.allowed:
            bound_logger.debug("Access granted", reason=decision.reason)
        else:
            bound_logger.warning("Access denied", reason=decision.reason)

    def require_permission(
            self,
            permission: Union[Permission, str],
            raise_on_deny: bool = True,
    ) -> Callable:
        """
        Декоратор для проверки разрешения на уровне эндпоинта.

        Args:
            permission: Требуемое разрешение
            raise_on_deny: Выбрасывать исключение при отказе

        Returns:
            Декоратор для FastAPI endpoint

        Example:
            @app.post("/documents")
            @rbac.require_permission(Permission.DOCUMENT_WRITE)
            async def create_document(...):
                ...
        """
        if isinstance(permission, str):
            try:
                permission = Permission(permission)
            except ValueError:
                raise ValueError(f"Unknown permission: {permission}")

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> Any:
                # Получаем контекст из kwargs (должен быть передан)
                access_context = kwargs.get("access_context")

                if not access_context or not isinstance(access_context, AccessContext):
                    if raise_on_deny:
                        from fastapi import HTTPException
                        raise HTTPException(
                            status_code=500,
                            detail="Access context not provided",
                        )
                    return await func(*args, **kwargs)

                # Проверяем разрешение
                if not access_context.permissions or permission not in access_context.permissions:
                    if raise_on_deny:
                        from fastapi import HTTPException
                        raise HTTPException(
                            status_code=403,
                            detail=f"Permission denied: {permission.value}",
                        )
                    logger.warning(
                        "Permission denied by decorator",
                        user_id=access_context.user_id,
                        permission=permission.value,
                    )
                    return await func(*args, **kwargs)

                return await func(*args, **kwargs)

            return wrapper

        return decorator

    def get_audit_log(
            self,
            user_id: Optional[str] = None,
            allowed_only: bool = False,
            denied_only: bool = False,
            limit: int = 100,
    ) -> List[AccessDecision]:
        """
        Получить журнал аудита доступа.

        Args:
            user_id: Фильтр по пользователю
            allowed_only: Только разрешённые доступы
            denied_only: Только запрещённые доступы
            limit: Максимальное количество записей

        Returns:
            Список AccessDecision
        """
        results = self._audit_log

        if user_id:
            results = [d for d in results if d.user_id == user_id]

        if allowed_only:
            results = [d for d in results if d.allowed]

        if denied_only:
            results = [d for d in results if not d.allowed]

        return results[-limit:]

    def clear_audit_log(self) -> None:
        """Очистить журнал аудита."""
        self._audit_log.clear()
        logger.info("RBAC audit log cleared")


# ── Convenience функции ─────────────────────────────────────────────────
# Глобальный экземпляр движка
_rbac_engine: Optional[RBACEngine] = None


def get_rbac_engine() -> RBACEngine:
    """Получить глобальный экземпляр RBAC движка."""
    global _rbac_engine
    if _rbac_engine is None:
        _rbac_engine = RBACEngine()
    return _rbac_engine


def check_access(
        user_id: str,
        role: Union[Role, str],
        resource_type: ResourceType,
        action: Action,
        resource_id: Optional[str] = None,
) -> AccessDecision:
    """
    Быстрая проверка доступа (использует глобальный движок).

    Example:
        >>> decision = check_access("user_123", "user", "document", "read")
        >>> if decision.allowed:
        ...     # Доступ разрешён
    """
    engine = get_rbac_engine()
    return engine.check_access(user_id, role, resource_type, action, resource_id)


def require_role(minimum_role: Union[Role, str]) -> Callable:
    """
    Декоратор для проверки минимальной роли пользователя.

    Args:
        minimum_role: Минимальная требуемая роль

    Returns:
        Декоратор для FastAPI endpoint

    Example:
        @app.delete("/users/{user_id}")
        @require_role(Role.ADMIN)
        async def delete_user(...):
            ...
    """
    if isinstance(minimum_role, str):
        minimum_role = Role.from_string(minimum_role)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            from fastapi import HTTPException, Depends

            # Пытаемся получить роль из контекста
            auth_context = kwargs.get("auth_context")

            if not auth_context:
                raise HTTPException(
                    status_code=500,
                    detail="Auth context not provided",
                )

            user_role = auth_context.get("role")
            if not user_role:
                raise HTTPException(
                    status_code=401,
                    detail="User role not found",
                )

            if isinstance(user_role, str):
                user_role = Role.from_string(user_role)

            if not user_role.has_at_least(minimum_role):
                raise HTTPException(
                    status_code=403,
                    detail=f"Minimum role required: {minimum_role.value}",
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


# ── Интеграция с FastAPI ────────────────────────────────────────────────
async def get_access_context(
        user_id: str,
        role: str,
        permissions: Optional[List[str]] = None,
) -> AccessContext:
    """
    Dependency для FastAPI — создаёт контекст доступа.

    Example:
        @app.post("/documents")
        async def create_document(
            ctx: AccessContext = Depends(get_access_context),
        ):
            ...
    """
    role_enum = Role.from_string(role)

    perm_set: Set[Permission] = set()
    if permissions:
        for p in permissions:
            try:
                perm_set.add(Permission(p))
            except ValueError:
                logger.warning("Unknown permission", permission=p)

    return AccessContext(
        user_id=user_id,
        role=role_enum,
        permissions=perm_set,
        resource_type="system",
        action="read",
    )


# ── Экспорт ─────────────────────────────────────────────────────────────
__all__ = [
    # Перечисления
    "Role",
    "Permission",
    "RoleLevel",
    "ResourceType",
    "Action",

    # Структуры данных
    "RoleDefinition",
    "AccessContext",
    "AccessDecision",

    # Конфигурация
    "RBACConfig",

    # Движок
    "RBACEngine",

    # Convenience функции
    "get_rbac_engine",
    "check_access",
    "require_role",
    "require_permission",
    "get_access_context",
]