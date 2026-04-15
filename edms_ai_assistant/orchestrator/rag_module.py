"""
rag_module.py — RAG Module для EDMS AI Ассистента.

Использует PostgreSQL + pgvector для хранения и поиска векторов.
Fallback: TF-IDF cosine similarity (без внешних сервисов).

Функциональность:
1. Хранение успешных диалогов (векторное представление)
2. Поиск похожих диалогов по cosine similarity (pgvector)
3. Форматирование few-shot примеров для промпта
4. Поддержка анти-примеров (негативная обратная связь)
5. Rebuild индекса из логов диалогов
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import asyncpg
import httpx

logger = logging.getLogger("rag")

# ── SQL Schema ────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
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
    """Запись в RAG-индексе."""
    user_query: str
    response_summary: str
    intent: str = ""
    selected_tool: str = ""
    tool_args: dict = field(default_factory=dict)
    full_response: str = ""
    feedback_score: int = 1
    is_anti_example: bool = False
    dialog_id: str | None = None


@dataclass
class RAGResult:
    """Результат поиска в RAG-индексе."""
    user_query: str
    response_summary: str
    intent: str
    selected_tool: str
    similarity: float
    is_anti_example: bool = False


# ── Embedding ─────────────────────────────────────────────────────────────────

async def _get_embedding(text: str, embedding_url: str, model: str) -> list[float] | None:
    """
    Получить эмбеддинг текста через API.

    Поддерживает OpenAI-совместимый API (Ollama, локальные модели).
    """
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{embedding_url.rstrip('/')}/embeddings",
                json={"model": model, "input": text[:2000]},
                headers={"Content-Type": "application/json"},
            )
            if resp.is_success:
                data = resp.json()
                return data["data"][0]["embedding"]
    except Exception as exc:
        logger.warning("Embedding API error: %s — using TF-IDF fallback", exc)
    return None


def _tfidf_vector(text: str, vocab: list[str], idf: dict[str, float]) -> list[float]:
    """Простое TF-IDF векторное представление без внешних зависимостей."""
    words = re.findall(r"\b\w+\b", text.lower())
    tf: dict[str, float] = {}
    for w in words:
        tf[w] = tf.get(w, 0) + 1
    n = len(words) or 1
    vec = []
    for term in vocab:
        tfidf = (tf.get(term, 0) / n) * idf.get(term, 1.0)
        vec.append(tfidf)
    return vec


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Косинусное сходство двух векторов."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── RAGModule ─────────────────────────────────────────────────────────────────

class RAGModule:
    """
    RAG-модуль с pgvector (PostgreSQL) и TF-IDF fallback.

    Использование:
        rag = RAGModule(postgres_dsn, embedding_url, embedding_model)
        await rag.initialize()
        results = await rag.search("найди входящий документ ВХ-001")
        few_shot = rag.format_few_shot(results)
    """

    def __init__(
        self,
        postgres_dsn: str,
        embedding_url: str,
        embedding_model: str = "nomic-embed-text",
    ) -> None:
        self._dsn = postgres_dsn
        self._embedding_url = embedding_url
        self._embedding_model = embedding_model
        self._pool: asyncpg.Pool | None = None
        # Кэш для TF-IDF fallback
        self._vocab: list[str] = []
        self._idf: dict[str, float] = {}
        self._fallback_entries: list[dict[str, Any]] = []

    async def initialize(self) -> None:
        """Инициализировать пул и схему БД."""
        try:
            self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=5)
            async with self._pool.acquire() as conn:
                await conn.execute(_SCHEMA_SQL)
            logger.info("RAGModule: pgvector pool initialized")
        except Exception as exc:
            logger.error("RAGModule init error: %s — will use in-memory fallback", exc)
            self._pool = None

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    async def add_entry(self, entry: RAGEntry) -> bool:
        """Добавить запись в RAG-индекс."""
        # Получаем эмбеддинг
        embedding = await _get_embedding(
            entry.user_query, self._embedding_url, self._embedding_model
        )

        if self._pool:
            return await self._add_to_pg(entry, embedding)

        # Fallback: в памяти
        self._fallback_entries.append({
            "user_query": entry.user_query,
            "response_summary": entry.response_summary,
            "intent": entry.intent,
            "selected_tool": entry.selected_tool,
            "feedback_score": entry.feedback_score,
            "is_anti_example": entry.is_anti_example,
        })
        self._rebuild_tfidf_vocab()
        return True

    async def _add_to_pg(self, entry: RAGEntry, embedding: list[float] | None) -> bool:
        """Добавить запись в PostgreSQL."""
        try:
            async with self._pool.acquire() as conn:
                if embedding:
                    emb_str = "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"
                    await conn.execute(
                        """
                        INSERT INTO edms_ai.rag_index
                            (dialog_id, user_query, normalized_query, intent, selected_tool,
                             tool_args, response_summary, full_response,
                             feedback_score, is_anti_example, embedding)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11::vector)
                        """,
                        entry.dialog_id, entry.user_query, entry.user_query.lower(),
                        entry.intent, entry.selected_tool,
                        json.dumps(entry.tool_args), entry.response_summary,
                        entry.full_response, entry.feedback_score,
                        entry.is_anti_example, emb_str,
                    )
                else:
                    # Без эмбеддинга
                    await conn.execute(
                        """
                        INSERT INTO edms_ai.rag_index
                            (dialog_id, user_query, normalized_query, intent, selected_tool,
                             tool_args, response_summary, full_response,
                             feedback_score, is_anti_example)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                        """,
                        entry.dialog_id, entry.user_query, entry.user_query.lower(),
                        entry.intent, entry.selected_tool,
                        json.dumps(entry.tool_args), entry.response_summary,
                        entry.full_response, entry.feedback_score,
                        entry.is_anti_example,
                    )
            return True
        except Exception as exc:
            logger.error("RAG _add_to_pg error: %s", exc)
            return False

    async def search(
        self,
        query: str,
        intent: str | None = None,
        top_k: int = 3,
        exclude_anti: bool = True,
    ) -> list[RAGResult]:
        """
        Найти похожие диалоги для few-shot.

        Args:
            query: Запрос пользователя
            intent: Фильтр по intent (опционально)
            top_k: Количество результатов (3-5 по спецификации)
            exclude_anti: Исключить анти-примеры
        """
        embedding = await _get_embedding(
            query, self._embedding_url, self._embedding_model
        )

        if self._pool and embedding:
            return await self._search_pg(query, embedding, intent, top_k, exclude_anti)

        # Fallback: TF-IDF
        return self._search_fallback(query, intent, top_k, exclude_anti)

    async def _search_pg(
        self,
        query: str,
        embedding: list[float],
        intent: str | None,
        top_k: int,
        exclude_anti: bool,
    ) -> list[RAGResult]:
        """Поиск через pgvector cosine similarity."""
        try:
            emb_str = "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"
            conditions = ["embedding IS NOT NULL"]
            params: list[Any] = [emb_str, top_k + 2]

            if exclude_anti:
                conditions.append("is_anti_example = FALSE")
            if intent:
                conditions.append(f"intent = ${len(params) + 1}")
                params.append(intent)

            where = " AND ".join(conditions)
            sql = f"""
                SELECT user_query, response_summary, intent, selected_tool, is_anti_example,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM edms_ai.rag_index
                WHERE {where}
                ORDER BY embedding <=> $1::vector
                LIMIT $2
            """
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)
            results = []
            for row in rows:
                if row["similarity"] >= 0.65:  # порог релевантности
                    results.append(RAGResult(
                        user_query=row["user_query"],
                        response_summary=row["response_summary"],
                        intent=row["intent"] or "",
                        selected_tool=row["selected_tool"] or "",
                        similarity=float(row["similarity"]),
                        is_anti_example=row["is_anti_example"],
                    ))
            return results[:top_k]
        except Exception as exc:
            logger.error("RAG _search_pg error: %s", exc)
            return []

    async def search_anti_examples(self, intent: str | None = None, limit: int = 5) -> list[RAGResult]:
        """Найти анти-примеры для обновления промпта."""
        if not self._pool:
            return [
                RAGResult(
                    user_query=e["user_query"],
                    response_summary=e["response_summary"],
                    intent=e.get("intent", ""),
                    selected_tool=e.get("selected_tool", ""),
                    similarity=1.0,
                    is_anti_example=True,
                )
                for e in self._fallback_entries
                if e.get("is_anti_example") and (not intent or e.get("intent") == intent)
            ][:limit]

        try:
            conditions = ["is_anti_example = TRUE"]
            params: list[Any] = [limit]
            if intent:
                conditions.append(f"intent = ${len(params) + 1}")
                params.append(intent)
            sql = f"""
                SELECT user_query, response_summary, intent, selected_tool
                FROM edms_ai.rag_index
                WHERE {" AND ".join(conditions)}
                ORDER BY created_at DESC LIMIT $1
            """
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)
            return [
                RAGResult(
                    user_query=row["user_query"],
                    response_summary=row["response_summary"],
                    intent=row["intent"] or "",
                    selected_tool=row["selected_tool"] or "",
                    similarity=1.0,
                    is_anti_example=True,
                )
                for row in rows
            ]
        except Exception as exc:
            logger.error("RAG search_anti_examples error: %s", exc)
            return []

    def _search_fallback(
        self,
        query: str,
        intent: str | None,
        top_k: int,
        exclude_anti: bool,
    ) -> list[RAGResult]:
        """TF-IDF поиск в памяти (fallback)."""
        if not self._fallback_entries or not self._vocab:
            return []

        query_vec = _tfidf_vector(query, self._vocab, self._idf)
        scored: list[tuple[float, dict]] = []

        for entry in self._fallback_entries:
            if exclude_anti and entry.get("is_anti_example"):
                continue
            if intent and entry.get("intent") != intent:
                continue
            entry_vec = _tfidf_vector(entry["user_query"], self._vocab, self._idf)
            sim = _cosine_similarity(query_vec, entry_vec)
            if sim >= 0.3:
                scored.append((sim, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RAGResult(
                user_query=e["user_query"],
                response_summary=e["response_summary"],
                intent=e.get("intent", ""),
                selected_tool=e.get("selected_tool", ""),
                similarity=sim,
                is_anti_example=e.get("is_anti_example", False),
            )
            for sim, e in scored[:top_k]
        ]

    def _rebuild_tfidf_vocab(self) -> None:
        """Перестроить TF-IDF словарь из fallback-записей."""
        from collections import Counter
        doc_count = len(self._fallback_entries)
        word_doc_count: Counter = Counter()
        for entry in self._fallback_entries:
            words = set(re.findall(r"\b\w+\b", entry["user_query"].lower()))
            word_doc_count.update(words)
        self._vocab = [w for w, c in word_doc_count.most_common(500)]
        self._idf = {
            w: math.log(doc_count / (c + 1)) + 1
            for w, c in word_doc_count.items()
            if w in set(self._vocab)
        }

    def format_few_shot(self, results: list[RAGResult]) -> str:
        """Форматировать результаты поиска как few-shot примеры для промпта."""
        if not results:
            return ""

        lines: list[str] = []
        for i, r in enumerate(results, 1):
            tool_info = f" (инструмент: {r.selected_tool})" if r.selected_tool else ""
            lines.append(f"Пример {i}:")
            lines.append(f"  Запрос: {r.user_query}")
            lines.append(f"  Ответ{tool_info}: {r.response_summary[:200]}")
        return "\n".join(lines)

    def format_anti_examples(self, results: list[RAGResult]) -> str:
        """Форматировать анти-примеры для промпта."""
        if not results:
            return ""

        lines: list[str] = ["Ошибки, которых нужно избегать:"]
        for r in results:
            lines.append(f"  ❌ Запрос: {r.user_query}")
            lines.append(f"     Неправильный ответ: {r.response_summary[:150]}")
        return "\n".join(lines)

    async def rebuild_from_logs(self, dialog_logs: list[dict[str, Any]]) -> int:
        """
        Перестроить RAG-индекс из логов диалогов.

        Args:
            dialog_logs: Список записей из dialog_logs с feedback >= 1

        Returns:
            Количество добавленных записей
        """
        added = 0
        for log in dialog_logs:
            if not log.get("user_query") or not log.get("final_response"):
                continue
            entry = RAGEntry(
                user_query=log["user_query"],
                response_summary=log["final_response"][:300],
                intent=log.get("intent", ""),
                selected_tool=log.get("selected_tool", ""),
                tool_args=log.get("tool_args") or {},
                full_response=log.get("final_response", ""),
                feedback_score=int(log.get("user_feedback", 1)),
                is_anti_example=int(log.get("user_feedback", 1)) < 0,
                dialog_id=str(log.get("dialog_id", "")),
            )
            success = await self.add_entry(entry)
            if success:
                added += 1

        logger.info("RAG rebuild_from_logs: added %d entries", added)
        return added

    async def get_stats(self) -> dict[str, Any]:
        """Статистика RAG-индекса."""
        if not self._pool:
            return {
                "backend": "in-memory (fallback)",
                "total": len(self._fallback_entries),
                "anti_examples": sum(1 for e in self._fallback_entries if e.get("is_anti_example")),
            }
        try:
            async with self._pool.acquire() as conn:
                total = await conn.fetchval("SELECT COUNT(*) FROM edms_ai.rag_index")
                anti = await conn.fetchval(
                    "SELECT COUNT(*) FROM edms_ai.rag_index WHERE is_anti_example = TRUE"
                )
                with_emb = await conn.fetchval(
                    "SELECT COUNT(*) FROM edms_ai.rag_index WHERE embedding IS NOT NULL"
                )
            return {
                "backend": "pgvector",
                "total": total,
                "anti_examples": anti,
                "with_embeddings": with_emb,
            }
        except Exception as exc:
            return {"error": str(exc)}
