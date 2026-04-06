# mcp-server/clients/document_creator_client.py
"""
EDMS Document Creator Client.
Перенесён из edms_ai_assistant/clients/document_creator_client.py.
"""
from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import Any

import httpx

from .base_client import EdmsHttpClient

logger = logging.getLogger(__name__)

_CATEGORY_LABELS: dict[str, str] = {
    "APPEAL": "Обращение",
    "INCOMING": "Входящий",
    "OUTGOING": "Исходящий",
    "INTERN": "Внутренний",
    "CONTRACT": "Договор",
    "MEETING": "Совещание",
    "MEETING_QUESTION": "Вопрос повестки",
    "QUESTION": "Вопрос",
    "CUSTOM": "Произвольный",
}


class DocumentCreatorClient(EdmsHttpClient):
    """HTTP-клиент для создания документа из файла (3 шага)."""

    async def find_profile_by_category(
        self, token: str, doc_category: str
    ) -> dict[str, Any] | None:
        """Найти активный профиль документа: GET /api/doc-profile."""
        normalized = doc_category.strip().upper()
        params: dict[str, Any] = {
            "docCategoryConst": normalized,
            "active": "true",
            "withAccess": "true",
            "listAttribute": "true",
        }
        result = await self._make_request(
            "GET", "api/doc-profile", token=token, params=params
        )
        if isinstance(result, list) and result:
            return result[0]
        if isinstance(result, dict):
            content: list = result.get("content") or []
            if content:
                return content[0]
        return None

    async def create_document(
        self, token: str, profile_id: str
    ) -> dict[str, Any] | None:
        """Создать документ из профиля: POST /api/document."""
        result = await self._make_request(
            "POST", "api/document",
            token=token, json={"id": str(profile_id)},
        )
        return result if isinstance(result, dict) and result else None

    async def upload_attachment(
        self,
        token: str,
        document_id: str,
        file_path: str,
        file_name: str | None = None,
    ) -> dict[str, Any] | None:
        """Загрузить файл как вложение: POST /api/document/{id}/attachment."""
        path = Path(file_path)
        if not path.exists():
            logger.error("Attachment file not found: '%s'", file_path)
            return None

        display_name = (file_name or path.name).strip()
        content_type, _ = mimetypes.guess_type(display_name)
        content_type = content_type or "application/octet-stream"

        url = f"{self.base_url}/api/document/{document_id}/attachment"
        headers = {"Authorization": f"Bearer {token}"}

        try:
            client = await self._get_client()
            with open(file_path, "rb") as fh:
                response = await client.post(
                    url,
                    headers=headers,
                    files={"file": (display_name, fh, content_type)},
                    timeout=self.timeout,
                )
            response.raise_for_status()

            if response.status_code == 204 or not response.content:
                return {}

            return response.json()
        except httpx.HTTPStatusError as exc:
            logger.error("Attachment upload failed [HTTP %d]", exc.response.status_code)
            raise
        except Exception as exc:
            logger.error("Attachment upload error: %s", exc, exc_info=True)
            raise