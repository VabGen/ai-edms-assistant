# src/ai_edms_assistant/domain/repositories/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from uuid import UUID

from pydantic import Field

from ..entities.base import DomainModel

T = TypeVar("T")
ID = TypeVar("ID")


# ---------------------------------------------------------------------------
# Pagination primitives
# ---------------------------------------------------------------------------


class PageRequest(DomainModel):
    """Immutable pagination parameters.

    Python equivalent of Spring's ``Pageable``. Immutable by inheritance
    from ``DomainModel`` (``frozen=True``).

    Design note:
        ``sort`` uses the same format as the Java backend:
        ``"fieldName,ASC"`` or ``"fieldName,DESC"``.
        Multiple sort fields are not supported at this layer — pass a
        comma-separated compound expression if needed.

    Attributes:
        page: Zero-based page index (default 0).
        size: Number of elements per page (default 20).
        sort: Optional sort expression, e.g. ``"lastName,ASC"``.

    Example:
        >>> req = PageRequest(page=2, size=10, sort="regDate,DESC")
        >>> req.as_params()
        {'page': 2, 'size': 10, 'sort': 'regDate,DESC'}
    """

    page: int = Field(default=0, ge=0)
    size: int = Field(default=20, ge=1, le=200)
    sort: str | None = None

    def as_params(self) -> dict[str, str | int]:
        """Serializes to a query-params dict for HTTPX requests.

        Returns:
            Dict with ``page``, ``size``, and optionally ``sort`` keys.
            Suitable for direct use as ``params=`` in ``httpx.AsyncClient``.
        """
        params: dict[str, str | int] = {"page": self.page, "size": self.size}
        if self.sort:
            params["sort"] = self.sort
        return params

    def next_page(self) -> "PageRequest":
        """Returns a new ``PageRequest`` for the next page.

        Returns:
            New immutable ``PageRequest`` with ``page + 1``.
        """
        return self.model_copy(update={"page": self.page + 1})


class Page(DomainModel, Generic[T]):
    """Paginated result set with Slice semantics.

    Python equivalent of Spring's ``SliceDto<T>``. Uses ``has_next`` instead
    of total count to avoid an extra COUNT query on the backend — mirrors
    the ``BaseRepositoryImpl.findSlice`` pattern.

    Immutable by inheritance from ``DomainModel`` (``frozen=True``).
    Content items are typed via the Generic ``T`` parameter.

    Attributes:
        items: Content items for this page.
        page: Zero-based current page index.
        size: Requested page size.
        has_next: Whether a next page exists.
        total: Optional total element count when the API returns it
               (``numberOfElements`` in Java SliceDto).
    """

    items: list[T] = Field(default_factory=list)
    page: int = Field(default=0, ge=0)
    size: int = Field(default=20, ge=1)
    has_next: bool = False
    total: int | None = None

    @property
    def is_empty(self) -> bool:
        """Returns True when the page contains no items.

        Returns:
            ``True`` when ``items`` list is empty.
        """
        return len(self.items) == 0

    @property
    def count(self) -> int:
        """Returns the number of items on the current page.

        Returns:
            Integer count of items in this page.
        """
        return len(self.items)

    @classmethod
    def empty(cls) -> "Page[T]":
        """Factory method returning an empty page.

        Useful as a safe default return value in repository implementations
        when an API call returns no results.

        Returns:
            ``Page`` with empty ``items`` list and ``has_next=False``.

        Example:
            >>> return Page.empty()
        """
        return cls(items=[], page=0, size=0, has_next=False)


# ---------------------------------------------------------------------------
# Base Repository ABC
# ---------------------------------------------------------------------------


class AbstractRepository(ABC, Generic[T]):
    """Base port (interface) for all EDMS domain repositories.

    This system is an AI agent proxy — there is no local database.
    All data lives in the remote EDMS REST API, accessed via typed HTTP clients
    in the infrastructure layer. Every method receives a ``token`` (JWT) for
    per-request bearer authentication.

    Architecture contract:
        - This ABC lives in the ``domain`` layer. It defines *what* can be done,
          not *how*. Infrastructure implementations must not leak here.
        - Implementations reside in
          ``infrastructure/edms_api/repositories/edms_*_repository.py``.
        - ``DI container`` (``container.py``) binds each ABC to its
          implementation at startup. Use cases receive the ABC type via DI,
          never the concrete class directly.
        - ``organization_id`` is optional on all methods. For single-tenant
          deployments it is set at the HTTP client level. Pass it explicitly
          only when performing cross-tenant operations.

    Type parameters:
        T: The domain entity type managed by this repository.

    Example:
        >>> from ai_edms_assistant.domain.entities import Document
        >>> class MyUseCase:
        ...     def __init__(self, repo: AbstractRepository[Document]) -> None:
        ...         self._repo = repo
        ...
        ...     async def execute(self, doc_id: UUID, token: str) -> Document | None:
        ...         return await self._repo.get_by_id(doc_id, token)
    """

    @abstractmethod
    async def get_by_id(
        self,
        entity_id: UUID,
        token: str,
        organization_id: str | None = None,
    ) -> T | None:
        """Fetch a single entity by its UUID primary key.

        Corresponds to ``GET /api/{resource}/{id}``.

        Args:
            entity_id: UUID of the target entity.
            token: JWT bearer token for authentication.
            organization_id: Optional org scope for multi-tenant requests.

        Returns:
            The entity instance, or ``None`` on 404.
        """

    @abstractmethod
    async def get_by_ids(
        self,
        entity_ids: list[UUID],
        token: str,
        organization_id: str | None = None,
    ) -> list[T]:
        """Fetch multiple entities by a list of UUIDs.

        Corresponds to ``POST /api/{resource}/search`` with ``ids`` filter,
        mirroring Java ``findByIdIn``. The order of results is not guaranteed
        to match the order of ``entity_ids``.

        Args:
            entity_ids: List of UUIDs to fetch. Empty list returns ``[]``.
            token: JWT bearer token.
            organization_id: Optional org scope.

        Returns:
            List of found entities. Missing IDs are silently omitted
            (no error for non-existent entities).
        """

    @abstractmethod
    async def find_page(
        self,
        token: str,
        organization_id: str | None = None,
        pagination: PageRequest | None = None,
    ) -> Page[T]:
        """Fetch a paginated slice of entities.

        Corresponds to ``GET /api/{resource}`` with pagination params.
        Mirrors Java ``findSlice(Pageable)`` — returns ``has_next`` instead
        of total count to avoid expensive COUNT queries.

        Args:
            token: JWT bearer token.
            organization_id: Optional org scope.
            pagination: Pagination parameters. Defaults to page 0, size 20
                when ``None``.

        Returns:
            ``Page[T]`` with ``has_next`` indicating whether more data exists.
        """
