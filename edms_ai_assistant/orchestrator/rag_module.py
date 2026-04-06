# orchestrator/rag_module.py
"""
RAG-модуль EDMS AI Assistant.

Primary:  Qdrant AsyncClient (коллекции successful_dialogs, anti_examples)
Fallback: FAISS IndexFlatIP (при недоступности Qdrant)
Embeddings: sentence-transformers paraphrase-multilingual-MiniLM-L12-v2

Экспортирует:
    DialogRecord  — единица хранения диалога
    RAGModule     — основной класс
    get_rag()     — синглтон
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from config import settings

logger = logging.getLogger(__name__)

_COLLECTION_SUCCESS = "successful_dialogs"
_COLLECTION_ANTI = "anti_examples"
_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
_VECTOR_DIM = 384
_MAX_ANTI_EXAMPLES = 10


# ── Датакласс DialogRecord ────────────────────────────────────────────────


@dataclass
class DialogRecord:
    """Запись диалога для RAG-индекса."""

    id: str
    user_query: str
    normalized_query: str
    intent: str
    tool_used: str
    tool_args: dict[str, Any]
    response: str
    rating: int                 # -1 негатив, 0 нейтрально, 1 позитив
    embedding: list[float]
    timestamp: datetime
    is_anti_example: bool = False

    @classmethod
    def new(
        cls,
        user_query: str,
        normalized_query: str,
        intent: str,
        tool_used: str,
        tool_args: dict[str, Any],
        response: str,
        rating: int = 0,
        embedding: list[float] | None = None,
    ) -> "DialogRecord":
        return cls(
            id=str(uuid.uuid4()),
            user_query=user_query,
            normalized_query=normalized_query,
            intent=intent,
            tool_used=tool_used,
            tool_args=tool_args,
            response=response,
            rating=rating,
            embedding=embedding or [],
            timestamp=datetime.now(timezone.utc),
            is_anti_example=rating == -1,
        )


# ── FAISS-индекс (fallback) ────────────────────────────────────────────────


class _FAISSIndex:
    """
    In-memory FAISS IndexFlatIP для fallback при недоступности Qdrant.

    Хранит все записи в памяти — не персистентен между перезапусками.
    """

    def __init__(self, dim: int = _VECTOR_DIM) -> None:
        self._dim = dim
        self._records: list[DialogRecord] = []
        self._anti_records: list[DialogRecord] = []
        self._index: Any = None
        self._anti_index: Any = None
        self._available = False

    def _ensure_faiss(self) -> bool:
        if self._available:
            return True
        try:
            import faiss  # noqa: F401
            self._available = True
            return True
        except ImportError:
            logger.warning("FAISS not available — RAG will return empty results")
            return False

    def _get_or_create_index(self, collection: str) -> tuple[Any, list[DialogRecord]]:
        import faiss
        is_anti = collection == _COLLECTION_ANTI
        records = self._anti_records if is_anti else self._records
        if is_anti:
            if self._anti_index is None:
                self._anti_index = faiss.IndexFlatIP(self._dim)
            return self._anti_index, records
        if self._index is None:
            self._index = faiss.IndexFlatIP(self._dim)
        return self._index, records

    def add(self, record: DialogRecord, collection: str) -> None:
        if not self._ensure_faiss() or not record.embedding:
            return
        index, records = self._get_or_create_index(collection)
        vec = np.array([record.embedding], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        index.add(vec)
        records.append(record)

    def search(self, embedding: list[float], top_k: int, collection: str) -> list[DialogRecord]:
        if not self._ensure_faiss() or not embedding:
            return []
        index, records = self._get_or_create_index(collection)
        if index.ntotal == 0:
            return []
        vec = np.array([embedding], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        _, indices = index.search(vec, min(top_k, index.ntotal))
        return [records[i] for i in indices[0] if 0 <= i < len(records)]


# ── RAGModule ─────────────────────────────────────────────────────────────


class RAGModule:
    """
    Retrieval-Augmented Generation для EDMS AI Assistant.

    Primary:  Qdrant (async)
    Fallback: FAISS IndexFlatIP (in-memory, при недоступности Qdrant)
    """

    def __init__(self) -> None:
        self._qdrant: Any = None
        self._qdrant_available = False
        self._faiss = _FAISSIndex()
        self._encoder: Any = None
        self._encoder_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """
        Инициализирует Qdrant + sentence-transformers.
        При неудаче — fallback на FAISS.
        """
        await asyncio.gather(
            self._init_qdrant(),
            self._init_encoder(),
            return_exceptions=True,
        )

    async def _init_qdrant(self) -> None:
        try:
            from qdrant_client import AsyncQdrantClient
            from qdrant_client.models import Distance, VectorParams

            url = str(settings.QDRANT_URL) if hasattr(settings, "QDRANT_URL") else "http://localhost:6333"
            self._qdrant = AsyncQdrantClient(url=url, timeout=10)

            for collection_name in (_COLLECTION_SUCCESS, _COLLECTION_ANTI):
                collections = await self._qdrant.get_collections()
                existing = {c.name for c in collections.collections}
                if collection_name not in existing:
                    await self._qdrant.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=_VECTOR_DIM, distance=Distance.COSINE),
                    )

            self._qdrant_available = True
            logger.info("RAGModule: Qdrant initialized at %s", url)

        except Exception as exc:
            logger.warning(
                "RAGModule: Qdrant unavailable (%s) — using FAISS fallback", exc
            )
            self._qdrant_available = False

    async def _init_encoder(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            loop = asyncio.get_event_loop()
            self._encoder = await loop.run_in_executor(
                None, SentenceTransformer, _EMBED_MODEL
            )
            logger.info("RAGModule: encoder '%s' loaded", _EMBED_MODEL)
        except Exception as exc:
            logger.warning("RAGModule: encoder unavailable (%s)", exc)
            self._encoder = None

    async def _embed(self, text: str) -> list[float]:
        """
        Вычисляет embedding для текста.
        Возвращает нулевой вектор при недоступности энкодера.
        """
        if self._encoder is None:
            return [0.0] * _VECTOR_DIM
        async with self._encoder_lock:
            loop = asyncio.get_event_loop()
            vec = await loop.run_in_executor(
                None, self._encoder.encode, text
            )
            return vec.tolist()

    async def search_similar(
        self,
        query: str,
        top_k: int = 5,
        collection: str = _COLLECTION_SUCCESS,
    ) -> list[DialogRecord]:
        """
        Поиск похожих диалогов по векторному сходству.

        Args:
            query:      Текст запроса для поиска.
            top_k:      Количество результатов.
            collection: successful_dialogs | anti_examples

        Returns:
            Список DialogRecord, отсортированных по релевантности.
        """
        embedding = await self._embed(query)
        if not any(embedding):
            return []

        if self._qdrant_available:
            return await self._qdrant_search(embedding, top_k, collection)
        return self._faiss.search(embedding, top_k, collection)

    async def _qdrant_search(
        self,
        embedding: list[float],
        top_k: int,
        collection: str,
    ) -> list[DialogRecord]:
        try:
            results = await self._qdrant.search(
                collection_name=collection,
                query_vector=embedding,
                limit=top_k,
                with_payload=True,
            )
            records = []
            for hit in results:
                payload = hit.payload or {}
                records.append(DialogRecord(
                    id=payload.get("id", str(hit.id)),
                    user_query=payload.get("user_query", ""),
                    normalized_query=payload.get("normalized_query", ""),
                    intent=payload.get("intent", ""),
                    tool_used=payload.get("tool_used", ""),
                    tool_args=payload.get("tool_args", {}),
                    response=payload.get("response", ""),
                    rating=payload.get("rating", 0),
                    embedding=[],
                    timestamp=datetime.fromisoformat(
                        payload.get("timestamp", datetime.now(timezone.utc).isoformat())
                    ),
                    is_anti_example=payload.get("is_anti_example", False),
                ))
            return records
        except Exception as exc:
            logger.warning("Qdrant search failed: %s — falling back to FAISS", exc)
            self._qdrant_available = False
            return self._faiss.search(embedding, top_k, collection)

    async def add_dialog(self, record: DialogRecord) -> None:
        """
        Добавляет диалог в индекс.

        rating=1  → successful_dialogs
        rating=-1 → anti_examples
        rating=0  → successful_dialogs (нейтрально)
        """
        if not record.embedding:
            record.embedding = await self._embed(record.user_query)

        collection = _COLLECTION_ANTI if record.rating == -1 else _COLLECTION_SUCCESS

        if self._qdrant_available:
            await self._qdrant_add(record, collection)
        else:
            self._faiss.add(record, collection)

    async def _qdrant_add(self, record: DialogRecord, collection: str) -> None:
        try:
            from qdrant_client.models import PointStruct
            payload = {
                "id": record.id,
                "user_query": record.user_query,
                "normalized_query": record.normalized_query,
                "intent": record.intent,
                "tool_used": record.tool_used,
                "tool_args": record.tool_args,
                "response": record.response[:2000],
                "rating": record.rating,
                "timestamp": record.timestamp.isoformat(),
                "is_anti_example": record.is_anti_example,
            }
            point = PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, record.id)),
                vector=record.embedding,
                payload=payload,
            )
            await self._qdrant.upsert(collection_name=collection, points=[point])
        except Exception as exc:
            logger.error("Qdrant add failed: %s", exc)

    async def update_rating(self, dialog_id: str, rating: int) -> None:
        """
        Обновляет рейтинг диалога.

        rating=1  → перемещает в successful_dialogs
        rating=-1 → перемещает в anti_examples
        """
        if rating not in (-1, 0, 1):
            logger.warning("Invalid rating %d for dialog %s", rating, dialog_id)
            return

        if not self._qdrant_available:
            logger.debug("Rating update skipped: Qdrant unavailable")
            return

        try:
            qdrant_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, dialog_id))
            # Ищем в обеих коллекциях
            for collection in (_COLLECTION_SUCCESS, _COLLECTION_ANTI):
                try:
                    results = await self._qdrant.retrieve(
                        collection_name=collection,
                        ids=[qdrant_id],
                        with_payload=True,
                        with_vectors=True,
                    )
                    if not results:
                        continue
                    point = results[0]
                    payload = dict(point.payload or {})
                    payload["rating"] = rating
                    payload["is_anti_example"] = rating == -1

                    # Удаляем из старой коллекции
                    await self._qdrant.delete(
                        collection_name=collection, points_selector=[qdrant_id]
                    )

                    # Добавляем в правильную коллекцию
                    new_collection = _COLLECTION_ANTI if rating == -1 else _COLLECTION_SUCCESS
                    from qdrant_client.models import PointStruct
                    await self._qdrant.upsert(
                        collection_name=new_collection,
                        points=[PointStruct(
                            id=qdrant_id,
                            vector=point.vector,
                            payload=payload,
                        )],
                    )
                    return

                except Exception:
                    continue
        except Exception as exc:
            logger.error("Rating update failed for %s: %s", dialog_id, exc)

    async def build_few_shot_block(self, query: str) -> str:
        """
        Возвращает блок успешных примеров для системного промпта.

        Format:
            === УСПЕШНЫЕ ПРИМЕРЫ ===
            [1] Запрос: ...
                Инструмент: ...
                Ответ: ...
        """
        records = await self.search_similar(query, top_k=5, collection=_COLLECTION_SUCCESS)
        if not records:
            return ""

        lines = ["=== УСПЕШНЫЕ ПРИМЕРЫ ==="]
        for i, rec in enumerate(records, 1):
            lines.append(
                f"[{i}] Запрос: {rec.user_query}\n"
                f"    Намерение: {rec.intent} | Инструмент: {rec.tool_used}\n"
                f"    Ответ: {rec.response[:300]}"
            )
        return "\n\n".join(lines)

    async def build_anti_examples_block(self) -> str:
        """
        Возвращает блок антипримеров для системного промпта.

        Format:
            === ЧЕГО НЕЛЬЗЯ ДЕЛАТЬ ===
            [1] Запрос: ...
                Проблема: ...
        """
        records = await self.search_similar(
            "ошибка неправильный ответ", top_k=_MAX_ANTI_EXAMPLES,
            collection=_COLLECTION_ANTI,
        )
        if not records:
            return ""

        lines = ["=== ЧЕГО НЕЛЬЗЯ ДЕЛАТЬ ==="]
        for i, rec in enumerate(records, 1):
            lines.append(
                f"[{i}] Запрос: {rec.user_query}\n"
                f"    Инструмент: {rec.tool_used}\n"
                f"    Неправильный ответ: {rec.response[:200]}"
            )
        return "\n\n".join(lines)

    async def rebuild_index(self) -> dict[str, Any]:
        """Пересобирает индекс. Вызывается ежедневно из feedback-collector."""
        if not self._qdrant_available:
            return {"status": "skipped", "reason": "Qdrant unavailable"}

        try:
            success_count = (await self._qdrant.get_collection(_COLLECTION_SUCCESS)).points_count
            anti_count = (await self._qdrant.get_collection(_COLLECTION_ANTI)).points_count
            logger.info(
                "RAG index: success=%d anti=%d", success_count, anti_count
            )
            return {
                "status": "ok",
                "successful_dialogs": success_count,
                "anti_examples": anti_count,
            }
        except Exception as exc:
            logger.error("rebuild_index failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    async def close(self) -> None:
        if self._qdrant_available and self._qdrant:
            try:
                await self._qdrant.close()
            except Exception:
                pass


_rag_instance: RAGModule | None = None


async def get_rag() -> RAGModule:
    """Возвращает синглтон RAGModule (ленивая инициализация)."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGModule()
        await _rag_instance.initialize()
    return _rag_instance
