# src/ai_edms_assistant/infrastructure/llm/chains/rag_chain.py
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from ....application.ports import AbstractVectorStore, SearchResult


class RAGChain:
    """RAG (Retrieval-Augmented Generation) chain using LCEL.

    Combines semantic search (vector store) with LLM generation to answer
    questions based on document context.

    Architecture:
        1. Retrieve relevant chunks from vector store
        2. Format context + question into prompt
        3. Generate answer with LLM
        4. Parse and return result
    """

    def __init__(
            self,
            llm: BaseChatModel,
            vector_store: AbstractVectorStore,
            top_k: int = 5,
    ):
        """Initialize RAG chain.

        Args:
            llm: LangChain chat model for generation.
            vector_store: Vector store for retrieval.
            top_k: Number of chunks to retrieve.
        """
        self._llm = llm
        self._vector_store = vector_store
        self._top_k = top_k
        self._chain = self._build_chain()

    def _build_chain(self) -> Runnable:
        """Build LCEL chain: retrieve → format → generate → parse.

        Returns:
            Runnable LCEL chain.
        """
        # Define prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Ты — экспертный помощник системы документооборота. "
                    "Используй предоставленный контекст для ответа на вопрос. "
                    "Если ответа нет в контексте, так и скажи.",
                ),
                (
                    "user",
                    "КОНТЕКСТ:\n{context}\n\nВОПРОС: {question}\n\nОТВЕТ:",
                ),
            ]
        )

        # Build chain
        chain = prompt | self._llm | StrOutputParser()
        return chain

    async def arun(
            self,
            query: str,
            document_id: str | None = None,
            collection_name: str = "default",
    ) -> str:
        """Run RAG chain asynchronously.

        Args:
            query: User question.
            document_id: Optional document UUID to restrict search.
            collection_name: Vector store collection name.

        Returns:
            Generated answer as string.
        """
        # 1. Retrieve relevant chunks
        filter_metadata = {"document_id": document_id} if document_id else None
        results = await self._vector_store.search(
            query=query,
            top_k=self._top_k,
            collection_name=collection_name,
            filter_metadata=filter_metadata,
        )

        # 2. Format context
        context = self._format_context(results)

        # 3. Generate answer
        answer = await self._chain.ainvoke(
            {
                "context": context,
                "question": query,
            }
        )

        return answer

    def _format_context(self, results: list[SearchResult]) -> str:
        """Format retrieved chunks into context string.

        Args:
            results: List of search results.

        Returns:
            Formatted context string.
        """
        if not results:
            return "[Контекст отсутствует]"

        chunks = []
        for idx, result in enumerate(results, 1):
            score = f"{result.score:.2f}" if result.score else "N/A"
            chunks.append(f"[{idx}] (score: {score})\n{result.text}")

        return "\n\n".join(chunks)
