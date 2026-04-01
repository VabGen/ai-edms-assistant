"""
RAG Module — Retrieval Augmented Generation с PostgreSQL + pgvector.

Функции:
1. Хранение успешных диалогов как векторных эмбеддингов
2. Поиск похожих диалогов по cosine similarity (pgvector)
3. Формирование few-shot примеров для системного промпта
4. Формирование анти-примеров из негативных диалогов
5. Ежедневное обновление индекса (новые позитивные диалоги)

Embedding модели:
- Основная: text-embedding-3-small (OpenAI) / nomic-embed-text (Ollama)
- Fallback: TF-IDF cosine similarity (без внешних сервисов)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import asyncpg
import httpx
import numpy as np

logger = logging.getLogger("rag_module")


# ── Schema ────────────────────────────────────────────────────────────────────

RAG_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS edms_ai;

CREATE TABLE IF NOT EXISTS edms_ai.rag_index (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dialog_id TEXT,
    user_query TEXT NOT NULL,
    normalized_query TEXT,
    intent TEXT,
    selected_tool TEXT,
    tool_args JSONB DEFAULT '{}',
    response_summary TEXT NOT NULL,
    full_response TEXT,
    feedback_score SMALLINT DEFAULT 1,
    is_anti_example BOOLEAN DEFAULT FALSE,
    embedding vector(768),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rag_embedding ON edms_ai.rag_index
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);
CREATE INDEX IF NOT EXISTS idx_rag_intent ON edms_ai.rag_index(intent);
CREATE INDEX IF NOT EXISTS idx_rag_is_anti ON edms_ai.rag_index(is_anti_example);
CREATE INDEX IF NOT EXISTS idx_rag_feedback ON edms_ai.rag_index(feedback_score);
"""


# ── Data models ───────────────────────────────────────────────────────────────


@dataclass
class RAGEntry:
    """Запись в RAG индексе."""
    user_query: str
    intent: str
    selected_tool: str
    response_summary: str
    full_response: str = ""
    normalized_query: str = ""
    tool_args: dict = None  # type: ignore[assignment]
    dialog_id: str = ""
    feedback_score: int = 1
    is_anti_example: bool = False

    def __post_init__(self) -> None:
        if self.tool_args is None:
            self.tool_args = {}


@dataclass
class RAGResult:
    """Результат поиска в RAG индексе."""
    entry: RAGEntry
    similarity: float
    entry_id: str = ""


# ── Embedding providers ───────────────────────────────────────────────────────


class EmbeddingProvider:
    """Базовый класс провайдера эмбеддингов."""

    EMBEDDING_DIM = 768

    async def embed(self, text: str) -> list[float] | None:
        raise NotImplementedError

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            emb = await self.embed(text)
            results.append(emb or [0.0] * self.EMBEDDING_DIM)
        return results


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Эмбеддинги через Ollama (nomic-embed-text или другая)."""

    EMBEDDING_DIM = 768

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def embed(self, text: str) -> list[float] | None:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                )
                resp.raise_for_status()
                data = resp.json()
                emb = data.get("embedding", [])
                # Нормализуем до нужной размерности
                if len(emb) != self.EMBEDDING_DIM:
                    emb = self._resize(emb, self.EMBEDDING_DIM)
                return emb
        except Exception as exc:
            logger.warning("OllamaEmbedding error: %s", exc)
            return None

    def _resize(self, emb: list[float], target_dim: int) -> list[float]:
        arr = np.array(emb, dtype=np.float32)
        if len(arr) > target_dim:
            return arr[:target_dim].tolist()
        padded = np.zeros(target_dim, dtype=np.float32)
        padded[:len(arr)] = arr
        return padded.tolist()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Эмбеддинги через OpenAI-совместимый API."""

    EMBEDDING_DIM = 768

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def embed(self, text: str) -> list[float] | None:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{self.base_url}/embeddings",
                    json={"model": self.model, "input": text[:8192]},
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                resp.raise_for_status()
                return resp.json()["data"][0]["embedding"]
        except Exception as exc:
            logger.warning("OpenAIEmbedding error: %s", exc)
            return None


class FallbackEmbeddingProvider(EmbeddingProvider):
    """
    Fallback: TF-IDF-подобные эмбеддинги без внешних сервисов.

    Простой детерминированный хэш-вектор.
    Низкое качество поиска, но работает без зависимостей.
    """

    EMBEDDING_DIM = 768

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._dim = self.EMBEDDING_DIM

    async def embed(self, text: str) -> list[float]:
        words = text.lower().split()
        vec = np.zeros(self._dim, dtype=np.float32)
        for word in words:
            if word not in self._vocab:
                # Детерминированный хэш слова → позиция в векторе
                idx = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % self._dim
                self._vocab[word] = idx
            idx = self._vocab[word]
            vec[idx] += 1.0
        # L2 нормализация
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


# ── RAG Module ────────────────────────────────────────────────────────────────


class RAGModule:
    """
    Модуль RAG с PostgreSQL + pgvector.

    Жизненный цикл:
        rag = RAGModule(dsn="postgresql://...", embedding_provider=...)
        await rag.init()

        # Поиск при каждом запросе
        results = await rag.search(query, intent="search_documents", top_k=3)
        few_shot = rag.format_few_shot(results)

        # Добавление новых записей (из лога + фидбека)
        await rag.add_entry(entry)

        # Ежедневное обновление
        await rag.rebuild_from_logs(dialog_logs)
    """

    MAX_QUERY_LEN = 512
    MIN_SIMILARITY = 0.6

    def __init__(
        self,
        dsn: str,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None
        self._embedder = embedding_provider or self._create_default_embedder()

    def _create_default_embedder(self) -> EmbeddingProvider:
        """Создать эмбеддер на основе переменных окружения."""
        ollama_url = os.getenv("OLLAMA_URL", "")
        openai_key = os.getenv("OPENAI_API_KEY", "")
        embed_url = os.getenv("LLM_EMBEDDING_URL", "")
        embed_model = os.getenv("LLM_EMBEDDING_MODEL", "nomic-embed-text")

        if ollama_url or (embed_url and "11434" in embed_url):
            url = ollama_url or embed_url.replace("/v1", "").replace("/api", "")
            logger.info("RAG: Using Ollama embeddings at %s model=%s", url, embed_model)
            return OllamaEmbeddingProvider(base_url=url, model=embed_model)

        if openai_key:
            logger.info("RAG: Using OpenAI embeddings")
            return OpenAIEmbeddingProvider(api_key=openai_key)

        logger.warning("RAG: No embedding service configured, using fallback (low quality)")
        return FallbackEmbeddingProvider()

    async def init(self) -> None:
        """Инициализация: подключение и создание таблиц."""
        try:
            self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=5)
            async with self._pool.acquire() as conn:
                await conn.execute(RAG_SCHEMA)
            count = await self._count_entries()
            logger.info("RAGModule initialized: %d entries in index", count)
        except Exception as exc:
            logger.error("RAGModule init error: %s — RAG disabled", exc)
            self._pool = None

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    async def add_entry(self, entry: RAGEntry) -> bool:
        """Добавить запись в RAG индекс с генерацией эмбеддинга."""
        if not self._pool:
            return False
        try:
            text_for_embed = f"{entry.intent} {entry.user_query} {entry.selected_tool}"
            embedding = await self._embedder.embed(text_for_embed[:self.MAX_QUERY_LEN])
            if not embedding:
                logger.warning("Failed to generate embedding for entry")
                return False

            async with self._pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO edms_ai.rag_index
                        (dialog_id, user_query, normalized_query, intent,
                         selected_tool, tool_args, response_summary, full_response,
                         feedback_score, is_anti_example, embedding)
                    VALUES ($1,$2,$3,$4,$5,$6::jsonb,$7,$8,$9,$10,$11::vector)
                """,
                    entry.dialog_id, entry.user_query, entry.normalized_query,
                    entry.intent, entry.selected_tool,
                    json.dumps(entry.tool_args or {}),
                    entry.response_summary[:500], entry.full_response[:2000],
                    entry.feedback_score, entry.is_anti_example,
                    str(embedding),
                )
            return True
        except Exception as exc:
            logger.warning("add_entry error: %s", exc)
            return False

    async def search(
        self,
        query: str,
        intent: str | None = None,
        top_k: int = 3,
        include_anti: bool = False,
    ) -> list[RAGResult]:
        """
        Найти похожие диалоги в RAG индексе.

        Args:
            query: Запрос пользователя
            intent: Фильтр по намерению (опционально)
            top_k: Количество результатов
            include_anti: Включать ли анти-примеры

        Returns:
            Список RAGResult, отсортированных по убыванию similarity
        """
        if not self._pool:
            return []
        try:
            search_text = f"{intent or ''} {query}"
            embedding = await self._embedder.embed(search_text[:self.MAX_QUERY_LEN])
            if not embedding:
                return []

            conditions = ["is_anti_example = $2"]
            params: list[Any] = [str(embedding), include_anti]
            if intent:
                conditions.append(f"(intent = ${len(params)+1} OR intent IS NULL)")
                params.append(intent)
            params.append(top_k)

            where_clause = " AND ".join(conditions)
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(f"""
                    SELECT id::text, user_query, normalized_query, intent,
                           selected_tool, tool_args, response_summary, full_response,
                           feedback_score, is_anti_example,
                           1 - (embedding <=> $1::vector) AS similarity
                    FROM edms_ai.rag_index
                    WHERE {where_clause}
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> $1::vector
                    LIMIT ${len(params)}
                """, *params)

            results = []
            for row in rows:
                if row["similarity"] < self.MIN_SIMILARITY:
                    continue
                entry = RAGEntry(
                    user_query=row["user_query"],
                    normalized_query=row["normalized_query"] or "",
                    intent=row["intent"] or "",
                    selected_tool=row["selected_tool"] or "",
                    response_summary=row["response_summary"],
                    full_response=row["full_response"] or "",
                    tool_args=json.loads(row["tool_args"]) if row["tool_args"] else {},
                    feedback_score=row["feedback_score"],
                    is_anti_example=row["is_anti_example"],
                )
                results.append(RAGResult(
                    entry=entry,
                    similarity=float(row["similarity"]),
                    entry_id=row["id"],
                ))
            return results
        except Exception as exc:
            logger.warning("search error: %s", exc)
            return []

    async def search_anti_examples(
        self,
        query: str,
        top_k: int = 2,
    ) -> list[RAGResult]:
        """Найти анти-примеры (что делать НЕ надо) для данного запроса."""
        return await self.search(query, top_k=top_k, include_anti=True)

    def format_few_shot(self, results: list[RAGResult]) -> str:
        """Форматирует успешные примеры для системного промпта."""
        if not results:
            return ""
        parts = ["Примеры успешных ответов ассистента на похожие запросы:"]
        for i, r in enumerate(results, 1):
            parts.append(f"\n--- Пример {i} (похожесть: {r.similarity:.2f}) ---")
            parts.append(f"Запрос: {r.entry.user_query}")
            if r.entry.intent:
                parts.append(f"Намерение: {r.entry.intent}")
            if r.entry.selected_tool:
                parts.append(f"Использованный инструмент: {r.entry.selected_tool}")
            parts.append(f"Ответ: {r.entry.response_summary}")
        return "\n".join(parts)

    def format_anti_examples(self, results: list[RAGResult]) -> str:
        """Форматирует анти-примеры для предупреждения ошибок."""
        if not results:
            return ""
        parts = ["ПРЕДУПРЕЖДЕНИЕ — избегай следующих ошибок (похожие запросы, негативная оценка):"]
        for i, r in enumerate(results, 1):
            parts.append(f"\n--- Анти-пример {i} ---")
            parts.append(f"Запрос: {r.entry.user_query}")
            parts.append(f"Ошибочный ответ: {r.entry.response_summary}")
        return "\n".join(parts)

    async def rebuild_from_logs(
        self,
        positive_dialogs: list[dict],
        negative_dialogs: list[dict],
    ) -> dict[str, int]:
        """
        Ежедневное обновление RAG индекса из логов диалогов.

        Args:
            positive_dialogs: Диалоги с позитивной оценкой
            negative_dialogs: Диалоги с негативной оценкой

        Returns:
            Статистика: {'added': N, 'anti_added': M, 'skipped': K}
        """
        added = 0
        anti_added = 0
        skipped = 0

        for dialog in positive_dialogs:
            if not dialog.get("user_query") or not dialog.get("final_response"):
                skipped += 1
                continue
            entry = RAGEntry(
                dialog_id=dialog.get("dialog_id", ""),
                user_query=dialog["user_query"],
                normalized_query=dialog.get("normalized_query", ""),
                intent=dialog.get("intent", ""),
                selected_tool=dialog.get("selected_tool", ""),
                tool_args=json.loads(dialog.get("tool_args", "{}")),
                response_summary=dialog["final_response"][:500],
                full_response=dialog.get("final_response", ""),
                feedback_score=1,
                is_anti_example=False,
            )
            if await self.add_entry(entry):
                added += 1
            else:
                skipped += 1

        for dialog in negative_dialogs:
            if not dialog.get("user_query"):
                skipped += 1
                continue
            entry = RAGEntry(
                dialog_id=dialog.get("dialog_id", ""),
                user_query=dialog["user_query"],
                intent=dialog.get("intent", ""),
                selected_tool=dialog.get("selected_tool", ""),
                response_summary=dialog.get("final_response", "")[:500],
                feedback_score=-1,
                is_anti_example=True,
            )
            if await self.add_entry(entry):
                anti_added += 1
            else:
                skipped += 1

        logger.info(
            "RAG rebuild complete: +%d examples, +%d anti-examples, %d skipped",
            added, anti_added, skipped,
        )
        return {"added": added, "anti_added": anti_added, "skipped": skipped}

    async def _count_entries(self) -> int:
        if not self._pool:
            return 0
        try:
            async with self._pool.acquire() as conn:
                return await conn.fetchval("SELECT COUNT(*) FROM edms_ai.rag_index")
        except Exception:
            return 0

    async def get_stats(self) -> dict[str, Any]:
        """Статистика RAG индекса."""
        if not self._pool:
            return {"status": "disabled"}
        try:
            async with self._pool.acquire() as conn:
                total = await conn.fetchval("SELECT COUNT(*) FROM edms_ai.rag_index")
                anti = await conn.fetchval(
                    "SELECT COUNT(*) FROM edms_ai.rag_index WHERE is_anti_example = TRUE"
                )
                by_intent = await conn.fetch(
                    """SELECT intent, COUNT(*) AS cnt FROM edms_ai.rag_index
                       WHERE NOT is_anti_example GROUP BY intent ORDER BY cnt DESC LIMIT 10"""
                )
            return {
                "status": "active",
                "total_entries": total,
                "anti_examples": anti,
                "positive_examples": total - anti,
                "by_intent": {r["intent"]: r["cnt"] for r in by_intent},
                "embedding_model": type(self._embedder).__name__,
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}
