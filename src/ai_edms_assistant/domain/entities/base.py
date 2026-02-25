# src/ai_edms_assistant/domain/entities/base.py
from __future__ import annotations
from pydantic import BaseModel, ConfigDict


class DomainModel(BaseModel):
    """Base class for all domain entities and value objects.

    Enforces immutability by default via ``model_config``. Subclasses that
    represent full entities (mutable aggregates) must explicitly override
    ``frozen=False``.

    Design decisions:
        - ``arbitrary_types_allowed=True`` — разрешает UUID, datetime из stdlib
          без дополнительных адаптеров.
        - ``populate_by_name=True`` — позволяет обращаться к полям как по
          Python-имени, так и по alias (нужно для маппинга camelCase → snake_case
          в infrastructure/mappers).
        - ``use_enum_values=False`` — храним Enum-объекты, не строки. Это
          позволяет использовать ``isinstance`` проверки в domain logic.
        - Сериализация в JSON идёт через ``model.model_dump(mode='json')``,
          что корректно обрабатывает UUID, datetime, Enum.

    Note:
        Domain models не должны содержать методов, работающих с I/O.
        Все вычисления — через ``@property`` или чистые методы.
    """

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        populate_by_name=True,
        use_enum_values=False,
        str_strip_whitespace=True,
    )


class MutableDomainModel(DomainModel):
    """Base class for mutable domain aggregates (entities with lifecycle).

    Use for entities that change state during their lifecycle, such as
    ``Document``, ``Task``, ``Employee``. Value objects should use the
    immutable ``DomainModel`` base.

    Note:
        Mutability here means the Python object can be modified after
        construction. It does NOT imply thread safety.
    """

    model_config = ConfigDict(
        frozen=False,
        arbitrary_types_allowed=True,
        populate_by_name=True,
        use_enum_values=False,
        str_strip_whitespace=True,
    )
