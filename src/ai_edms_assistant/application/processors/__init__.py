# src/ai_edms_assistant/application/processors/__init__.py
"""Data processors for file and document transformation.

Processors handle data enrichment workflows that don't fit into
domain services (which must be pure, no I/O) or use cases (which
orchestrate operations, not transform data).

Processors:
    AbstractProcessor: Base class for all processors with Generic typing.
    FileProcessor: Extract text from uploaded file bytes (stub).
    DocumentProcessor: Enrich document with vector embeddings (stub).
    AppealProcessor: Clean and normalize appeal text for NLP (basic impl).

Note:
    FileProcessor and DocumentProcessor are stub implementations for MVP.
    For production, integrate:
    - FileProcessor: pypdf2, python-docx, unstructured, pytesseract
    - DocumentProcessor: sentence-transformers, FAISS/Qdrant
    - AppealProcessor: SpaCy for advanced NLP (currently basic cleaning)
"""

from .appeal_processor import AppealProcessor
from .base_processor import AbstractProcessor
from .document_processor import DocumentProcessor
from .file_processor import FileProcessor

__all__ = [
    "AbstractProcessor",
    "FileProcessor",
    "DocumentProcessor",
    "AppealProcessor",
]
