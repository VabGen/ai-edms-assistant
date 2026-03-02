# src/ai_edms_assistant/application/tools/comparison_tool.py
"""Document comparison tool with full version support.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from ...domain.repositories import AbstractDocumentRepository
from ...domain.services import DocumentComparer
from ...domain.services.document_comparer import ComparisonFocus
from ..dto import ComparisonResultDto
from ..ports import AbstractLLMProvider, LLMMessage
from .base_tool import AbstractEdmsTool

logger = logging.getLogger(__name__)


class DocumentComparisonInput(BaseModel):
    """Input schema for document comparison.

    Supports two modes:
    1. **Сравнение версий одного документа** (рекомендуется):
       Укажи ``document_id`` — инструмент сам загрузит список версий
       и сравнит ``version_1`` vs ``version_2`` (или крайние, если не указаны).

    2. **Сравнение двух произвольных документов**:
       Укажи ``document_id_1`` + ``document_id_2``.
    """

    token: str = Field(..., description="JWT токен")

    # Режим 1: версии одного документа
    document_id: UUID | None = Field(
        default=None,
        description=(
            "UUID документа для сравнения версий. "
            "Если указан — версии загружаются автоматически."
        ),
    )
    version_1: int | None = Field(
        default=None,
        description="Номер базовой (старшей) версии. None = самая старая.",
    )
    version_2: int | None = Field(
        default=None,
        description="Номер новой версии. None = самая новая.",
    )

    # Режим 2: два разных документа
    document_id_1: UUID | None = Field(
        default=None,
        description="UUID первого (базового) документа.",
    )
    document_id_2: UUID | None = Field(
        default=None,
        description="UUID второго (нового) документа.",
    )

    comparison_focus: ComparisonFocus = Field(
        default="all",
        description=(
            "Аспект сравнения: "
            "'metadata' — рег. номер, статус, тип; "
            "'content' — текст, исполнитель; "
            "'contract' — договорные поля; "
            "'all' — все поля."
        ),
    )


class ComparisonTool(AbstractEdmsTool):
    """Tool for comparing two document versions or two different documents.

    Supports two operating modes:
    1. Version comparison for a single document (via get_versions()).
    2. Arbitrary two-document comparison (via get_by_id x2).

    The ``comparison_focus`` parameter filters which field groups are
    compared and is now passed through to ``DocumentComparer.compare()``.
    """

    name: str = "doc_compare"
    description: str = (
        "Сравнивает два документа или версии одного документа. "
        "Для сравнения версий укажи document_id. "
        "Для сравнения двух разных документов укажи document_id_1 и document_id_2."
    )
    args_schema: type[BaseModel] = DocumentComparisonInput

    def __init__(
        self,
        document_repository: AbstractDocumentRepository,
        document_comparer: DocumentComparer,
        llm_provider: AbstractLLMProvider,
        **kwargs: Any,
    ) -> None:
        """Initialize with injected dependencies.

        Args:
            document_repository: For fetching documents and versions.
            document_comparer: Domain service for field-level diff.
            llm_provider: For generating the LLM analysis report.
        """
        super().__init__(**kwargs)
        self._doc_repo = document_repository
        self._comparer = document_comparer
        self._llm = llm_provider

    async def _arun(
        self,
        token: str,
        document_id: UUID | None = None,
        version_1: int | None = None,
        version_2: int | None = None,
        document_id_1: UUID | None = None,
        document_id_2: UUID | None = None,
        comparison_focus: ComparisonFocus = "all",
    ) -> dict[str, Any]:
        """Execute document comparison.

        Args:
            token: JWT bearer token.
            document_id: For version comparison mode.
            version_1: Base version number (None = oldest).
            version_2: New version number (None = latest).
            document_id_1: Base document UUID (for two-doc mode).
            document_id_2: New document UUID (for two-doc mode).
            comparison_focus: Field group filter.

        Returns:
            Success response dict with comparison data and LLM analysis.
        """
        try:
            # ── Режим 1: сравнение версий одного документа ─────────────
            if document_id is not None:
                return await self._compare_versions(
                    document_id=document_id,
                    token=token,
                    version_1=version_1,
                    version_2=version_2,
                    focus=comparison_focus,
                )

            # ── Режим 2: сравнение двух документов ─────────────────────
            if document_id_1 and document_id_2:
                return await self._compare_two_documents(
                    doc_id_1=document_id_1,
                    doc_id_2=document_id_2,
                    token=token,
                    focus=comparison_focus,
                )

            return self._handle_error(
                ValueError(
                    "Не указаны документы для сравнения. "
                    "Укажи document_id (для версий) "
                    "или document_id_1 + document_id_2."
                )
            )

        except Exception as e:
            return self._handle_error(e)

    # ── Режим 1: сравнение версий ──────────────────────────────────────────

    async def _compare_versions(
        self,
        document_id: UUID,
        token: str,
        version_1: int | None,
        version_2: int | None,
        focus: ComparisonFocus,
    ) -> dict[str, Any]:
        """Fetch all versions and compare two of them.

        Fetching: GET /api/document/{id}/version (через репозиторий).
        Mapping: _map_version_to_entity() корректно извлекает document
                 из вложенного DocumentVersionDto.

        Args:
            document_id: UUID of the parent document.
            token: JWT bearer token.
            version_1: Base version number. None = oldest available.
            version_2: New version number. None = latest available.
            focus: Field group filter.

        Returns:
            Success response with comparison data.
        """
        # ── Шаг 1: загружаем все версии ───────────────────────────────
        versions = await self._doc_repo.get_versions(document_id, token)

        if not versions:
            # Документа нет версий — сравниваем с текущим состоянием
            logger.info(
                "no_versions_found_comparing_with_current",
                extra={"document_id": str(document_id)},
            )
            current = await self._doc_repo.get_by_id(document_id, token)
            if not current:
                return self._handle_error(
                    ValueError(f"Документ {document_id} не найден")
                )
            return self._success_response(
                data={
                    "summary": "У документа нет исторических версий. "
                               "Доступна только текущая версия.",
                    "changed_fields": [],
                    "llm_analysis": (
                        "У данного документа нет исторических версий для сравнения. "
                        f"Документ содержит: рег. номер {current.reg_number or 'не присвоен'}, "
                        f"статус: {current.status_label if current.status else 'не указан'}."
                    ),
                    "versions_available": [],
                    "total_changes": 0,
                },
                message="Версии для сравнения недоступны",
            )

        # ── Шаг 2: формируем список доступных версий для UI ───────────
        versions_info = self._build_versions_info(versions)

        # ── Шаг 3: выбираем две версии для сравнения ──────────────────
        base_doc, new_doc = self._select_versions(versions, version_1, version_2)

        if base_doc is None or new_doc is None:
            return self._handle_error(
                ValueError(
                    f"Версии {version_1} и/или {version_2} не найдены. "
                    f"Доступные версии: {[v['version_number'] for v in versions_info]}"
                )
            )

        # Получаем version_number для отображения
        base_ver_num = getattr(base_doc, "version_number_snapshot", None)
        new_ver_num = getattr(new_doc, "version_number_snapshot", None)

        # ── Шаг 4: сравниваем через domain service ─────────────────────
        comparison_result = self._comparer.compare_versions(
            base=base_doc,
            new=new_doc,
            base_version_number=base_ver_num,
            new_version_number=new_ver_num,
            focus=focus,
        )

        dto = ComparisonResultDto.from_comparison_result(comparison_result)

        # ── Шаг 5: генерируем LLM-анализ ──────────────────────────────
        llm_context = comparison_result.as_llm_context()
        llm_analysis = await self._generate_llm_analysis(
            context=llm_context,
            base_reg=base_doc.reg_number,
            new_reg=new_doc.reg_number,
            base_version=base_ver_num,
            new_version=new_ver_num,
        )

        # ── Детальные diffs для ответа ─────────────────────────────────
        detailed_diffs = [
            {
                "field": d.field_name,
                "label": d.field_label,
                "change_type": d.change_type.value,
                "old_value": str(d.old_value) if d.old_value is not None else None,
                "new_value": str(d.new_value) if d.new_value is not None else None,
            }
            for d in comparison_result.changed_fields
        ]

        return self._success_response(
            data={
                "mode": "version_comparison",
                "document_id": str(document_id),
                "base_version": base_ver_num,
                "new_version": new_ver_num,
                "summary": dto.summary,
                "changed_fields": dto.changed_fields,
                "detailed_diffs": detailed_diffs,
                "total_changes": dto.total_changes,
                "llm_analysis": llm_analysis,
                "versions_available": versions_info,
            },
            message=(
                f"Версии {base_ver_num} → {new_ver_num} сравнены, "
                f"найдено {dto.total_changes} изменений"
            ),
        )

    # ── Режим 2: два разных документа ─────────────────────────────────────

    async def _compare_two_documents(
        self,
        doc_id_1: UUID,
        doc_id_2: UUID,
        token: str,
        focus: ComparisonFocus,
    ) -> dict[str, Any]:
        """Compare two arbitrary documents by UUID.

        Args:
            doc_id_1: Base document UUID.
            doc_id_2: New document UUID.
            token: JWT bearer token.
            focus: Field group filter.

        Returns:
            Success response with comparison data.
        """
        doc1 = await self._doc_repo.get_by_id(doc_id_1, token)
        doc2 = await self._doc_repo.get_by_id(doc_id_2, token)

        if not doc1 or not doc2:
            missing = []
            if not doc1:
                missing.append(str(doc_id_1))
            if not doc2:
                missing.append(str(doc_id_2))
            return self._handle_error(
                ValueError(f"Документ(ы) не найден(ы): {', '.join(missing)}")
            )

        comparison_result = self._comparer.compare(
            base=doc1, new=doc2, focus=focus
        )
        dto = ComparisonResultDto.from_comparison_result(comparison_result)

        llm_context = comparison_result.as_llm_context()
        llm_analysis = await self._generate_llm_analysis(
            context=llm_context,
            base_reg=doc1.reg_number,
            new_reg=doc2.reg_number,
        )

        detailed_diffs = [
            {
                "field": d.field_name,
                "label": d.field_label,
                "change_type": d.change_type.value,
                "old_value": str(d.old_value) if d.old_value is not None else None,
                "new_value": str(d.new_value) if d.new_value is not None else None,
            }
            for d in comparison_result.changed_fields
        ]

        return self._success_response(
            data={
                "mode": "document_comparison",
                "base_document_id": str(doc_id_1),
                "new_document_id": str(doc_id_2),
                "base_reg_number": doc1.reg_number,
                "new_reg_number": doc2.reg_number,
                "summary": dto.summary,
                "changed_fields": dto.changed_fields,
                "detailed_diffs": detailed_diffs,
                "total_changes": dto.total_changes,
                "llm_analysis": llm_analysis,
            },
            message=f"Документы сравнены, найдено {dto.total_changes} изменений",
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _select_versions(
        self,
        versions: list,
        version_1: int | None,
        version_2: int | None,
    ) -> tuple:
        """Select base and new document from version list.

        Args:
            versions: List of Document entities (sorted oldest-first).
            version_1: Base version number or None (= oldest).
            version_2: New version number or None (= latest).

        Returns:
            Tuple (base_doc, new_doc). Either may be None if not found.
        """
        if not versions:
            return None, None

        if version_1 is None and version_2 is None:
            # По умолчанию: самая старая → самая новая
            return versions[0], versions[-1]

        def find_by_version(target: int | None, fallback_idx: int):
            if target is None:
                return versions[fallback_idx]
            for v in versions:
                if getattr(v, "version_number_snapshot", None) == target:
                    return v
            return None

        base = find_by_version(version_1, 0)
        new = find_by_version(version_2, -1)
        return base, new

    @staticmethod
    def _build_versions_info(versions: list) -> list[dict]:
        """Build a list of version metadata for UI display.

        Args:
            versions: List of Document entities with version_number_snapshot.

        Returns:
            List of dicts with version metadata.
        """
        result = []
        for v in versions:
            ver_num = getattr(v, "version_number_snapshot", None)
            result.append(
                {
                    "version_number": ver_num,
                    "document_id": str(v.id),
                    "reg_number": v.reg_number,
                    "status": v.status.value if v.status else None,
                    "create_date": (
                        v.create_date.isoformat() if v.create_date else None
                    ),
                }
            )
        return result

    async def _generate_llm_analysis(
        self,
        context: str,
        base_reg: str | None = None,
        new_reg: str | None = None,
        base_version: int | None = None,
        new_version: int | None = None,
    ) -> str:
        """Generate LLM analysis report from comparison context.

        Args:
            context: LLM-ready context string from ComparisonResult.as_llm_context().
            base_reg: Registration number of base document.
            new_reg: Registration number of new document.
            base_version: Version number of base doc.
            new_version: Version number of new doc.

        Returns:
            LLM-generated analysis string.
        """
        # Формируем информативный заголовок
        if base_version is not None and new_version is not None:
            header = (
                f"Документ {'рег. №' + base_reg if base_reg else 'без рег. номера'}. "
                f"Версия {base_version} → Версия {new_version}."
            )
        else:
            base_info = f"рег. №{base_reg}" if base_reg else "без рег. номера"
            new_info = f"рег. №{new_reg}" if new_reg else "без рег. номера"
            header = f"Документ 1 ({base_info}) vs Документ 2 ({new_info})."

        prompt = (
            f"{header}\n\n"
            f"Результаты сравнения:\n"
            f"{context}\n\n"
            "Составь краткий деловой отчёт об изменениях. "
            "Укажи: что изменилось, насколько существенны изменения, "
            "на что стоит обратить внимание. "
            "Пиши на русском языке, используй деловой стиль."
        )

        try:
            llm_resp = await self._llm.complete(
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.3,
            )
            return llm_resp.content.strip()
        except Exception as exc:
            logger.warning(
                "llm_analysis_failed_returning_context",
                extra={"error": str(exc)},
            )
            # Fallback: возвращаем структурированный контекст без LLM
            return context

    def _run(self, *args: Any, **kwargs: Any) -> None:
        """Sync execution not supported."""
        raise NotImplementedError("Use _arun")