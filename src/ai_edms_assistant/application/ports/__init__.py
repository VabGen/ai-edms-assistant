# src/ai_edms_assistant/application/ports/__init__.py
"""Application layer ports (interfaces) for infrastructure dependencies.

Ports define the contracts between the application layer and external
systems. They are ABCs that live in the application layer but are
implemented in the infrastructure layer.

This is the Hexagonal Architecture pattern: the application defines
*what* it needs (ports), and the infrastructure provides *how* (adapters).

Ports:
    AbstractLLMProvider: LLM completion, streaming, and embeddings.
    AbstractVectorStore: Vector similarity search and document ingestion.
    AbstractStorage: File upload / download to object storage.
    AbstractNLPExtractor: Structured data extraction from unstructured text.

Supporting types:
    LLMMessage: Single message in an LLM conversation.
    LLMResponse: Response from an LLM completion request.
    DocumentChunk: Text chunk for vector store ingestion.
    SearchResult: Result from a vector similarity search.

Architecture rules:
    - Application layer depends only on these ABCs, never on implementations.
    - Infrastructure layer provides concrete classes implementing these ABCs.
    - DI container (``container.py``) binds ABCs to implementations.
    - Use cases receive ports via constructor injection.
"""

from .llm_port import AbstractLLMProvider, LLMMessage, LLMResponse
from .nlp_port import AbstractNLPExtractor
from .storage_port import AbstractStorage
from .vector_store_port import AbstractVectorStore, DocumentChunk, SearchResult

__all__ = [
    "AbstractLLMProvider",
    "LLMMessage",
    "LLMResponse",
    "AbstractVectorStore",
    "DocumentChunk",
    "SearchResult",
    "AbstractStorage",
    "AbstractNLPExtractor",
]
