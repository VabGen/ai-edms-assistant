"""
orchestrator/rag_module.py — Модуль Retrieval-Augmented Generation.

Векторный поиск успешных диалогов для формирования few-shot примеров.
Первичное хранилище: Qdrant. Резерв: FAISS (локальный, без сервера).

Возможности:
  - Добавление записей с автогенерацией эмбеддингов
  - Семантический поиск похожих диалогов
  - Обновление рейтингов (RLHF-loop)
  - Формирование блоков few-shot и anti-examples для промпта
  - Ежедневная перестройка индекса
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import pickle
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY")
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
)
EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "384"))
FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "/tmp/edms_rag_faiss.pkl")

COLLECTION_SUCCESSFUL = "successful_dialogs"
COLLECTION_ANTI = "anti_examples"

# ---------------------------------------------------------------------------
# Датаклассы
# ---------------------------------------------------------------------------

@dataclass
class DialogRecord:
    """
    Запись диалога для RAG-индекса.

    Поля:
        id: уникальный идентификатор записи
        user_query: исходный запрос пользователя
        normalized_query: нормализованный запрос (плейсхолдеры)
        intent: распознанное намерение
        tool_used: вызванный MCP-инструмент
        tool_args: аргументы инструмента
        response: итоговый ответ ассистента
        rating: оценка пользователя (-1, 0, 1)
        embedding: вектор эмбеддинга (хранится отдельно в Qdrant)
        timestamp: время создания
        is_anti_example: True если это пример неправильного поведения
    """
    id: str
    user_query: str
    normalized_query: str
    intent: str
    tool_used: str
    tool_args: dict[str, Any]
    response: str
    rating: int = 0
    embedding: list[float] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_anti_example: bool = False

    def to_payload(self) -> dict[str, Any]:
        """Сериализовать запись в payload для Qdrant (без embedding)."""
        return {
            "id": self.id,
            "user_query": self.user_query,
            "normalized_query": self.normalized_query,
            "intent": self.intent,
            "tool_used": self.tool_used,
            "tool_args": self.tool_args,
            "response": self.response[:500],  # ограничиваем размер
            "rating": self.rating,
            "timestamp": self.timestamp.isoformat(),
            "is_anti_example": self.is_anti_example,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "DialogRecord":
        """Восстановить запись из payload Qdrant."""
        ts_raw = payload.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_raw) if ts_raw else datetime.now(timezone.utc)
        except ValueError:
            ts = datetime.now(timezone.utc)
        return cls(
            id=payload.get("id", ""),
            user_query=payload.get("user_query", ""),
            normalized_query=payload.get("normalized_query", ""),
            intent=payload.get("intent", ""),
            tool_used=payload.get("tool_used", ""),
            tool_args=payload.get("tool_args", {}),
            response=payload.get("response", ""),
            rating=payload.get("rating", 0),
            timestamp=ts,
            is_anti_example=payload.get("is_anti_example", False),
        )


# ---------------------------------------------------------------------------
# Провайдер эмбеддингов
# ---------------------------------------------------------------------------

class EmbeddingProvider:
    """
    Генератор эмбеддингов с помощью sentence-transformers.

    Модель: paraphrase-multilingual-MiniLM-L12-v2
    Поддерживает русский и английский языки.
    Синглтон — модель загружается один раз.
    """

    _instance: "EmbeddingProvider | None" = None
    _model: Any = None

    @classmethod
    def get(cls) -> "EmbeddingProvider":
        """Получить синглтон провайдера."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self) -> Any:
        """Ленивая загрузка модели sentence-transformers."""
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
            log.info("Loading embedding model: %s", EMBEDDING_MODEL)
            self._model = SentenceTransformer(EMBEDDING_MODEL)
            log.info("Embedding model loaded, dim=%d", EMBEDDING_DIM)
            return self._model
        except ImportError:
            log.warning("sentence-transformers not available, using hash-based embeddings")
            return None

    def embed(self, text: str) -> list[float]:
        """
        Генерировать эмбеддинг для текста.

        Параметры:
            text: входной текст (русский или английский)

        Возвращает:
            Список float — вектор размерности EMBEDDING_DIM
        """
        model = self._load_model()
        if model is not None:
            try:
                vector = model.encode(text, normalize_embeddings=True)
                return vector.tolist()
            except Exception as exc:
                log.warning("Embedding failed, falling back to hash: %s", exc)

        # Фоллбэк: детерминированный хэш-вектор
        return self._hash_embedding(text)

    def _hash_embedding(self, text: str) -> list[float]:
        """
        Детерминированный фоллбэк-эмбеддинг на основе хэша.
        Используется когда sentence-transformers недоступен.
        """
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Расширяем 32 байта до EMBEDDING_DIM float'ов
        values: list[float] = []
        extended = (h * ((EMBEDDING_DIM // len(h)) + 2))[:EMBEDDING_DIM]
        for byte in extended[:EMBEDDING_DIM]:
            # Нормализуем в [-1, 1]
            values.append((byte / 127.5) - 1.0)
        # L2-нормализация
        norm = sum(v * v for v in values) ** 0.5 or 1.0
        return [v / norm for v in values]

    async def embed_async(self, text: str) -> list[float]:
        """Асинхронная версия embed (выполняется в executor)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed, text)


# ---------------------------------------------------------------------------
# FAISS-фоллбэк
# ---------------------------------------------------------------------------

class FaissStore:
    """
    Локальное FAISS-хранилище для работы без Qdrant.

    Использует IndexFlatIP (косинусное сходство через inner product
    на нормализованных векторах).
    """

    def __init__(self, index_path: str = FAISS_INDEX_PATH) -> None:
        self._path = index_path
        self._index: Any = None
        self._records: list[DialogRecord] = []
        self._load()

    def _load(self) -> None:
        """Загрузить индекс с диска если существует."""
        try:
            import faiss  # type: ignore[import]
            if Path(self._path).exists():
                with open(self._path, "rb") as f:
                    saved = pickle.load(f)
                self._index = saved["index"]
                self._records = saved["records"]
                log.info("FAISS index loaded: %d records", len(self._records))
            else:
                self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
        except ImportError:
            log.warning("FAISS not available")
            self._index = None
        except Exception as exc:
            log.warning("FAISS load error: %s", exc)
            self._init_new_index()

    def _init_new_index(self) -> None:
        """Инициализировать пустой FAISS индекс."""
        try:
            import faiss  # type: ignore[import]
            self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
        except ImportError:
            self._index = None

    def save(self) -> None:
        """Сохранить индекс на диск."""
        if self._index is None:
            return
        try:
            with open(self._path, "wb") as f:
                pickle.dump({"index": self._index, "records": self._records}, f)
        except Exception as exc:
            log.warning("FAISS save error: %s", exc)

    def add(self, record: DialogRecord) -> None:
        """Добавить запись в FAISS индекс."""
        if self._index is None or not record.embedding:
            return
        try:
            import numpy as np  # type: ignore[import]
            vec = np.array([record.embedding], dtype=np.float32)
            self._index.add(vec)
            self._records.append(record)
            self.save()
        except Exception as exc:
            log.warning("FAISS add error: %s", exc)

    def search(self, embedding: list[float], top_k: int = 5) -> list[DialogRecord]:
        """Найти top-k похожих записей."""
        if self._index is None or len(self._records) == 0:
            return []
        try:
            import numpy as np  # type: ignore[import]
            vec = np.array([embedding], dtype=np.float32)
            scores, indices = self._index.search(vec, min(top_k, len(self._records)))
            results: list[DialogRecord] = []
            for idx in indices[0]:
                if 0 <= idx < len(self._records):
                    results.append(self._records[idx])
            return results
        except Exception as exc:
            log.warning("FAISS search error: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Основной класс RAGModule
# ---------------------------------------------------------------------------

class RAGModule:
    """
    Модуль RAG для EDMS AI Assistant.

    Использует Qdrant как основное хранилище и FAISS как резерв.
    Обеспечивает:
    - Добавление новых диалогов с автогенерацией эмбеддингов
    - Семантический поиск похожих диалогов
    - Обновление рейтингов и перемещение в anti_examples
    - Формирование few-shot и anti-examples блоков для промпта
    """

    def __init__(self) -> None:
        self._embedder = EmbeddingProvider.get()
        self._faiss = FaissStore()
        self._qdrant_client: Any = None
        self._qdrant_available = False

    async def initialize(self) -> None:
        """
        Инициализировать клиент Qdrant и создать коллекции.
        При недоступности Qdrant переходит на FAISS.
        """
        await self._init_qdrant()

    async def _init_qdrant(self) -> None:
        """Инициализация Qdrant клиента и коллекций."""
        try:
            from qdrant_client import AsyncQdrantClient
            from qdrant_client.models import Distance, VectorParams

            client = AsyncQdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                timeout=10,
            )
            # Проверяем доступность
            await client.get_collections()
            self._qdrant_client = client
            self._qdrant_available = True

            # Создаём коллекции если не существуют
            for collection_name in (COLLECTION_SUCCESSFUL, COLLECTION_ANTI):
                existing = await client.get_collections()
                names = [c.name for c in existing.collections]
                if collection_name not in names:
                    await client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=EMBEDDING_DIM,
                            distance=Distance.COSINE,
                        ),
                    )
                    log.info("Created Qdrant collection: %s", collection_name)

            log.info("Qdrant initialized: %s", QDRANT_URL)

        except Exception as exc:
            log.warning("Qdrant unavailable (%s), falling back to FAISS", exc)
            self._qdrant_available = False

    async def add_dialog(self, record: DialogRecord) -> None:
        """
        Добавить диалог в RAG-индекс.

        Автоматически генерирует эмбеддинг из текста запроса.
        Добавляет в соответствующую коллекцию (successful или anti_examples).

        Параметры:
            record: запись диалога для индексирования
        """
        # Генерируем эмбеддинг
        embedding_text = f"{record.intent} {record.user_query} {record.tool_used}"
        record.embedding = await self._embedder.embed_async(embedding_text)

        collection = COLLECTION_ANTI if record.is_anti_example else COLLECTION_SUCCESSFUL

        if self._qdrant_available:
            await self._add_to_qdrant(record, collection)
        else:
            self._faiss.add(record)

    async def _add_to_qdrant(self, record: DialogRecord, collection: str) -> None:
        """Добавить запись в Qdrant."""
        try:
            from qdrant_client.models import PointStruct

            point = PointStruct(
                id=record.id,
                vector=record.embedding,
                payload=record.to_payload(),
            )
            await self._qdrant_client.upsert(
                collection_name=collection,
                points=[point],
            )
        except Exception as exc:
            log.error("Qdrant add error: %s, falling back to FAISS", exc)
            self._faiss.add(record)

    async def search_similar(
        self,
        query: str,
        top_k: int = 5,
        collection: str = COLLECTION_SUCCESSFUL,
    ) -> list[DialogRecord]:
        """
        Найти семантически похожие диалоги.

        Параметры:
            query: текст поискового запроса
            top_k: количество результатов
            collection: коллекция для поиска (successful_dialogs или anti_examples)

        Возвращает:
            Список DialogRecord, отсортированный по релевантности
        """
        embedding = await self._embedder.embed_async(query)

        if self._qdrant_available:
            return await self._search_qdrant(embedding, top_k, collection)
        else:
            return self._faiss.search(embedding, top_k)

    async def _search_qdrant(
        self,
        embedding: list[float],
        top_k: int,
        collection: str,
    ) -> list[DialogRecord]:
        """Поиск в Qdrant."""
        try:
            results = await self._qdrant_client.search(
                collection_name=collection,
                query_vector=embedding,
                limit=top_k,
                with_payload=True,
            )
            records = []
            for point in results:
                payload = point.payload or {}
                record = DialogRecord.from_payload(payload)
                records.append(record)
            return records
        except Exception as exc:
            log.error("Qdrant search error: %s, falling back to FAISS", exc)
            return self._faiss.search(embedding, top_k)

    async def update_rating(self, dialog_id: str, rating: int) -> None:
        """
        Обновить рейтинг диалога.

        При rating=1: добавить в successful_dialogs (если ещё нет)
        При rating=-1: переместить в anti_examples

        Параметры:
            dialog_id: ID диалога
            rating: новый рейтинг (-1, 0, 1)
        """
        if not self._qdrant_available:
            # В FAISS обновить рейтинг нельзя без перестройки
            log.info("FAISS mode: rating update skipped for %s", dialog_id)
            return

        try:
            # Находим запись в successful_dialogs
            results = await self._qdrant_client.retrieve(
                collection_name=COLLECTION_SUCCESSFUL,
                ids=[dialog_id],
                with_payload=True,
                with_vectors=True,
            )

            if not results:
                return

            point = results[0]
            payload = dict(point.payload or {})
            payload["rating"] = rating
            vector = list(point.vector) if point.vector else []

            if rating == -1:
                # Перемещаем в anti_examples
                payload["is_anti_example"] = True

                from qdrant_client.models import PointStruct
                anti_point = PointStruct(
                    id=dialog_id,
                    vector=vector,
                    payload=payload,
                )
                await self._qdrant_client.upsert(
                    collection_name=COLLECTION_ANTI,
                    points=[anti_point],
                )
                # Удаляем из successful
                await self._qdrant_client.delete(
                    collection_name=COLLECTION_SUCCESSFUL,
                    points_selector=[dialog_id],
                )
                log.info("Dialog %s moved to anti_examples", dialog_id)
            else:
                # Обновляем рейтинг в successful
                await self._qdrant_client.set_payload(
                    collection_name=COLLECTION_SUCCESSFUL,
                    payload={"rating": rating},
                    points=[dialog_id],
                )

        except Exception as exc:
            log.error("Rating update error for %s: %s", dialog_id, exc)

    async def build_few_shot_block(self, query: str, top_k: int = 5) -> str:
        """
        Сформировать блок few-shot примеров для системного промпта.

        Находит top_k наиболее похожих успешных диалогов и форматирует их
        в читаемый блок для включения в промпт.

        Параметры:
            query: запрос пользователя
            top_k: количество примеров

        Возвращает:
            Форматированный блок строк для промпта
        """
        records = await self.search_similar(query, top_k=top_k, collection=COLLECTION_SUCCESSFUL)

        if not records:
            return ""

        lines = ["=== УСПЕШНЫЕ ПРИМЕРЫ РАБОТЫ АССИСТЕНТА ===\n"]
        for i, rec in enumerate(records, 1):
            lines.append(f"--- Пример {i} ---")
            lines.append(f"Запрос: {rec.user_query}")
            if rec.tool_used:
                lines.append(f"Инструмент: {rec.tool_used}")
                if rec.tool_args:
                    args_preview = json.dumps(
                        {k: v for k, v in list(rec.tool_args.items())[:3]},
                        ensure_ascii=False,
                    )
                    lines.append(f"Параметры: {args_preview}")
            lines.append(f"Ответ: {rec.response[:300]}")
            lines.append("")

        return "\n".join(lines)

    async def build_anti_examples_block(self, query: str, top_k: int = 3) -> str:
        """
        Сформировать блок антипримеров для системного промпта.

        Находит top_k примеров плохого поведения, похожих на текущий запрос.

        Параметры:
            query: запрос пользователя
            top_k: количество антипримеров

        Возвращает:
            Форматированный блок антипримеров для промпта
        """
        records = await self.search_similar(query, top_k=top_k, collection=COLLECTION_ANTI)

        if not records:
            return ""

        lines = ["=== ЧЕГО НЕ НУЖНО ДЕЛАТЬ (АНТИПРИМЕРЫ) ===\n"]
        for i, rec in enumerate(records, 1):
            lines.append(f"--- Антипример {i} ---")
            lines.append(f"Запрос: {rec.user_query}")
            lines.append(f"Ошибочный ответ: {rec.response[:200]}")
            lines.append("(Избегай подобного поведения)")
            lines.append("")

        return "\n".join(lines)

    async def rebuild_index(self) -> dict[str, int]:
        """
        Ежедневная перестройка RAG-индекса.

        Пересчитывает эмбеддинги для всех записей и пересоздаёт коллекции.
        Используется для актуализации модели эмбеддингов.

        Возвращает:
            Словарь со статистикой: total_rebuilt, successful, anti
        """
        if not self._qdrant_available:
            log.info("FAISS mode: index rebuild not supported")
            return {"total_rebuilt": 0, "successful": 0, "anti": 0}

        stats: dict[str, int] = {"total_rebuilt": 0, "successful": 0, "anti": 0}

        for collection_name, stat_key in [
            (COLLECTION_SUCCESSFUL, "successful"),
            (COLLECTION_ANTI, "anti"),
        ]:
            try:
                # Выгружаем все записи
                all_points = []
                offset = None
                while True:
                    result = await self._qdrant_client.scroll(
                        collection_name=collection_name,
                        limit=100,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,
                    )
                    points, next_offset = result
                    all_points.extend(points)
                    if next_offset is None:
                        break
                    offset = next_offset

                # Пересчитываем эмбеддинги
                from qdrant_client.models import PointStruct

                rebuilt = []
                for point in all_points:
                    payload = dict(point.payload or {})
                    text = f"{payload.get('intent', '')} {payload.get('user_query', '')} {payload.get('tool_used', '')}"
                    new_embedding = await self._embedder.embed_async(text)
                    rebuilt.append(
                        PointStruct(
                            id=point.id,
                            vector=new_embedding,
                            payload=payload,
                        )
                    )

                if rebuilt:
                    await self._qdrant_client.upsert(
                        collection_name=collection_name,
                        points=rebuilt,
                    )

                count = len(rebuilt)
                stats[stat_key] = count
                stats["total_rebuilt"] += count
                log.info("Rebuilt %d records in %s", count, collection_name)

            except Exception as exc:
                log.error("Rebuild error for %s: %s", collection_name, exc)

        return stats

    async def get_stats(self) -> dict[str, Any]:
        """Получить статистику RAG-хранилища."""
        stats: dict[str, Any] = {
            "backend": "qdrant" if self._qdrant_available else "faiss",
            "qdrant_url": QDRANT_URL,
        }

        if self._qdrant_available:
            try:
                for coll in (COLLECTION_SUCCESSFUL, COLLECTION_ANTI):
                    info = await self._qdrant_client.get_collection(coll)
                    stats[coll] = {
                        "vectors_count": info.vectors_count,
                        "indexed_vectors_count": info.indexed_vectors_count,
                    }
            except Exception as exc:
                stats["error"] = str(exc)
        else:
            stats["faiss_records"] = len(self._faiss._records)

        return stats
