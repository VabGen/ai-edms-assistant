# src/ai_edms_assistant/application/use_cases/compare_documents.py
"""Compare Documents use case — full version support.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from ...domain.exceptions import DocumentNotFoundError
from ...domain.repositories import AbstractDocumentRepository
from ...domain.services import DocumentComparer
from ...domain.services.document_comparer import ComparisonFocus
from ..dto import ComparisonResultDto
from .base import AbstractUseCase


class CompareDocumentsRequest(BaseModel):
    """Request DTO for document comparison.

    Supports two modes:
    1. **Version mode**: укажи ``document_id`` + опционально ``version_1``/``version_2``.
    2. **Two-doc mode**: укажи ``base_document_id`` + ``new_document_id``.

    Attributes:
        token: JWT bearer token.
        document_id: UUID документа для сравнения версий (режим 1).
        version_1: Номер базовой версии. None = самая старая.
        version_2: Номер новой версии. None = самая новая.
        base_document_id: UUID базового документа (режим 2).
        new_document_id: UUID нового документа (режим 2).
        focus: Фокус сравнения (all / metadata / content / contract).
    """

    token: str

    # Режим 1: версии одного документа
    document_id: UUID | None = None
    version_1: int | None = None
    version_2: int | None = None

    # Режим 2: два разных документа
    base_document_id: UUID | None = None
    new_document_id: UUID | None = None

    focus: ComparisonFocus = Field(default="all")

    @model_validator(mode="after")
    def validate_mode(self) -> "CompareDocumentsRequest":
        """Ensure at least one valid comparison mode is specified.

        Returns:
            Validated request.

        Raises:
            ValueError: When neither mode has sufficient parameters.
        """
        has_version_mode = self.document_id is not None
        has_two_doc_mode = (
            self.base_document_id is not None
            and self.new_document_id is not None
        )
        if not has_version_mode and not has_two_doc_mode:
            raise ValueError(
                "Укажи либо document_id (для версий), "
                "либо base_document_id + new_document_id (для двух документов)."
            )
        return self


class CompareDocumentsUseCase(
    AbstractUseCase[CompareDocumentsRequest, ComparisonResultDto]
):
    """Use case: Compare two document versions and return a structured diff.

    Supports two modes:
    1. Сравнение версий одного документа через ``get_versions()``.
    2. Сравнение двух произвольных документов через ``get_by_id()`` x2.

    Business rules:
        - В режиме версий: если версий нет — возвращает пустой результат
          (не ошибку), т.к. документ существует, просто без версий.
        - В режиме двух документов: оба документа обязаны существовать.
    """

    def __init__(
        self,
        document_repository: AbstractDocumentRepository,
        document_comparer: DocumentComparer,
    ) -> None:
        """Initialize with injected dependencies.

        Args:
            document_repository: Repository for fetching documents and versions.
            document_comparer: Domain service for field-level diff.
        """
        self._doc_repo = document_repository
        self._comparer = document_comparer

    async def execute(self, request: CompareDocumentsRequest) -> ComparisonResultDto:
        """Execute the comparison use case.

        Args:
            request: Validated comparison request.

        Returns:
            ``ComparisonResultDto`` with summary and changed fields.

        Raises:
            DocumentNotFoundError: When a required document does not exist.
            ValueError: When version numbers are invalid.
        """
        if request.document_id is not None:
            return await self._execute_version_comparison(request)
        return await self._execute_two_doc_comparison(request)

    async def _execute_version_comparison(
        self,
        request: CompareDocumentsRequest,
    ) -> ComparisonResultDto:
        """Execute comparison of two versions of the same document.

        Fetches all versions via ``get_versions()``, selects the two
        requested ones, and runs ``compare_versions()``.

        Args:
            request: Request with ``document_id`` set.

        Returns:
            ``ComparisonResultDto`` with version metadata in summary.

        Raises:
            DocumentNotFoundError: When the document itself does not exist.
            ValueError: When requested version numbers are not found.
        """
        assert request.document_id is not None  # narrowing for mypy

        # 1. Проверяем что документ существует
        doc = await self._doc_repo.get_by_id(
            entity_id=request.document_id,
            token=request.token,
        )
        if doc is None:
            raise DocumentNotFoundError(document_id=request.document_id)

        # 2. Загружаем версии
        versions = await self._doc_repo.get_versions(
            document_id=request.document_id,
            token=request.token,
        )

        if len(versions) < 2:
            # Недостаточно версий — возвращаем информативный пустой результат
            return ComparisonResultDto(
                base_doc_id=str(request.document_id),
                new_doc_id=str(request.document_id),
                summary=(
                    f"У документа {doc.reg_number or request.document_id} "
                    f"недостаточно версий для сравнения "
                    f"(доступно: {len(versions)}, требуется минимум 2)."
                ),
                changed_fields=[],
                total_changes=0,
            )

        # 3. Выбираем версии
        base_doc, new_doc = self._select_versions(
            versions, request.version_1, request.version_2
        )

        if base_doc is None or new_doc is None:
            available = [
                getattr(v, "version_number_snapshot", "?") for v in versions
            ]
            raise ValueError(
                f"Версии {request.version_1} и/или {request.version_2} не найдены. "
                f"Доступные: {available}"
            )

        # 4. Сравниваем
        base_ver_num = getattr(base_doc, "version_number_snapshot", None)
        new_ver_num = getattr(new_doc, "version_number_snapshot", None)

        result = self._comparer.compare_versions(
            base=base_doc,
            new=new_doc,
            base_version_number=base_ver_num,
            new_version_number=new_ver_num,
            focus=request.focus,
        )

        return ComparisonResultDto.from_comparison_result(result)

    async def _execute_two_doc_comparison(
        self,
        request: CompareDocumentsRequest,
    ) -> ComparisonResultDto:
        """Execute comparison of two different documents.

        Args:
            request: Request with ``base_document_id`` + ``new_document_id``.

        Returns:
            ``ComparisonResultDto``.

        Raises:
            DocumentNotFoundError: When either document does not exist.
        """
        assert request.base_document_id is not None
        assert request.new_document_id is not None

        base_doc = await self._doc_repo.get_by_id(
            entity_id=request.base_document_id,
            token=request.token,
        )
        if base_doc is None:
            raise DocumentNotFoundError(document_id=request.base_document_id)

        new_doc = await self._doc_repo.get_by_id(
            entity_id=request.new_document_id,
            token=request.token,
        )
        if new_doc is None:
            raise DocumentNotFoundError(document_id=request.new_document_id)

        result = self._comparer.compare(
            base=base_doc, new=new_doc, focus=request.focus
        )
        return ComparisonResultDto.from_comparison_result(result)

    @staticmethod
    def _select_versions(
        versions: list,
        version_1: int | None,
        version_2: int | None,
    ) -> tuple:
        """Select base and new document from sorted version list.

        Args:
            versions: List of Document entities sorted oldest-first.
            version_1: Base version number. None = oldest.
            version_2: New version number. None = latest.

        Returns:
            Tuple (base_doc, new_doc). Either may be None if not found.
        """
        if not versions:
            return None, None

        if version_1 is None and version_2 is None:
            return versions[0], versions[-1]

        def find(target: int | None, fallback_idx: int):
            if target is None:
                return versions[fallback_idx]
            for v in versions:
                if getattr(v, "version_number_snapshot", None) == target:
                    return v
            return None

        return find(version_1, 0), find(version_2, -1)