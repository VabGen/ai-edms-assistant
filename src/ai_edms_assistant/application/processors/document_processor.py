# src/ai_edms_assistant/application/processors/document_processor.py
"""Document processor for enriching documents with vector embeddings.

MVP: Stub passthrough that logs structured metadata for observability.

Production path:
    Replace ``_process_stub()`` with a real embedding pipeline using
    ``sentence-transformers`` and FAISS / Qdrant as the vector store.
    The ``AbstractProcessor`` interface and public ``process()`` method
    remain unchanged — only the private implementation changes.

Architecture:
    Application Layer → Processors
    Input:  Domain ``Document`` entity (from Infrastructure / Service layer).
    Output: Same ``Document`` entity (potentially enriched in production).
    No direct I/O in this class — all async I/O happens in injected ports.
"""

from __future__ import annotations

import structlog

from ...domain.entities.document import Document
from .base_processor import AbstractProcessor

log = structlog.get_logger(__name__)


class DocumentProcessor(AbstractProcessor[Document, Document]):
    """Enrich a ``Document`` entity with vector embeddings.

    MVP behaviour:
        Passthrough — returns the document unmodified.
        Emits a structured ``debug`` log entry for observability.

    Production behaviour (TODO):
        1. Extract text via ``Document.get_full_text()``.
        2. Generate multilingual embeddings (``paraphrase-multilingual-mpnet``).
        3. Upsert to vector store (FAISS local / Qdrant remote).
        4. Attach ``embedding_id`` to document metadata.

    Follows ``AbstractProcessor[Document, Document]`` contract — callers
    always receive a ``Document``, never ``None``.
    """

    async def process(self, input_data: Document) -> Document:
        """Enrich document with vector embeddings.

        Args:
            input_data: Domain ``Document`` entity to process.

        Returns:
            The same ``Document`` entity (passthrough in stub).
            In production: document with ``embedding_id`` attached.

        Note:
            Stub implementation — no embeddings are generated.
            The structured log is intentional: it provides a trace of
            which documents pass through the processor pipeline, useful
            for debugging embedding coverage in production.
        """
        # ── Extract text for embedding (already used by production path) ──────
        text = input_data.get_full_text()
        text_length = len(text) if text else 0

        log.debug(
            "document_processor_stub",
            document_id=str(input_data.id),
            category=(
                input_data.document_category.value
                if input_data.document_category
                else None
            ),
            status=input_data.status.value if input_data.status else None,
            has_text=bool(text),
            text_length=text_length,
            has_attachments=input_data.has_attachments,
            attachments_count=len(input_data.attachments),
        )

        # Stub: return document as-is (no embedding generated)
        return await self._process_stub(input_data)

    # ── Private ────────────────────────────────────────────────────────────────

    @staticmethod
    async def _process_stub(document: Document) -> Document:
        """MVP stub — passthrough.

        Args:
            document: Input document entity.

        Returns:
            Unmodified document entity.

        TODO: Replace with ``_process_production()`` when embedding
        infrastructure is available.
        """
        return document

    # ── Production implementation (commented — ready for activation) ──────────
    #
    # async def _process_production(
    #     self,
    #     document: Document,
    #     vector_store: "AbstractVectorStore",
    # ) -> Document:
    #     """Generate and store embeddings for a document.
    #
    #     Args:
    #         document: Domain Document entity.
    #         vector_store: Injected vector store port (FAISS / Qdrant).
    #
    #     Returns:
    #         Document with embedding_id attached to metadata.
    #     """
    #     from sentence_transformers import SentenceTransformer
    #
    #     MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    #     model = SentenceTransformer(MODEL_NAME)
    #
    #     text = document.get_full_text()
    #     if not text or len(text) < 10:
    #         log.warning(
    #             "document_processor_skip_empty",
    #             document_id=str(document.id),
    #         )
    #         return document
    #
    #     embeddings = model.encode(text, normalize_embeddings=True)
    #
    #     embedding_id = await vector_store.upsert(
    #         vector=embeddings.tolist(),
    #         payload={
    #             "document_id": str(document.id),
    #             "category": document.document_category.value if document.document_category else None,
    #             "status": document.status.value if document.status else None,
    #             "text_preview": text[:200],
    #         },
    #     )
    #
    #     log.info(
    #         "document_embedding_stored",
    #         document_id=str(document.id),
    #         embedding_id=str(embedding_id),
    #         text_length=len(text),
    #     )
    #
    #     # In production: attach embedding_id to document metadata
    #     # (requires Document entity to have a metadata dict field)
    #     # document.metadata["embedding_id"] = str(embedding_id)
    #
    #     return document
