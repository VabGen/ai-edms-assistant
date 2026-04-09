# edms_ai_assistant/clients/document_creator_client.py
"""
EDMS AI Assistant — Document Creator Client.

  1. GET  /api/doc-profile   → поиск профиля по категории документа
  2. POST /api/document      → создание документа из профиля
  3. POST /api/document/{id}/attachment → загрузка файла как вложения
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import Any

import httpx

from edms_ai_assistant.mcp_server.clients.base_client import EdmsHttpClient
from edms_ai_assistant.shared.utils.utils import prepare_auth_headers

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
    """HTTP client for the document-creation-from-file workflow.

    All three steps are stateless and can be used independently.
    The client reuses the shared httpx.AsyncClient from EdmsHttpClient.
    """

    # ── 1. Profile lookup ─────────────────────────────────────────────────────

    async def find_profile_by_category(
        self,
        token: str,
        doc_category: str,
    ) -> dict[str, Any] | None:
        """Find the first active accessible DocumentProfile for the given category.

        Calls::

            GET /api/doc-profile?docCategoryConst=<CAT>
                                 &active=true
                                 &withAccess=true
                                 &listAttribute=true

        ``listAttribute=true`` makes the server return a plain ``List<DocumentProfileDto>``
        instead of a paged Slice — faster and simpler for our use-case.

        Args:
            token: JWT bearer token.
            doc_category: ``DocumentCategoryConstants`` value, e.g. ``"APPEAL"``.

        Returns:
            First matching ``DocumentProfileDto`` as dict, or ``None`` if not found.
        """
        normalized = doc_category.strip().upper()
        params: dict[str, Any] = {
            "docCategoryConst": normalized,
            "active": "true",
            "withAccess": "true",
            "listAttribute": "true",
        }
        logger.info(
            "Looking up profile for category '%s' (%s)",
            _CATEGORY_LABELS.get(normalized, normalized),
            normalized,
        )

        result = await self._make_request(
            "GET", "api/doc-profile", token=token, params=params
        )

        if isinstance(result, list) and result:
            profile = result[0]
            logger.info(
                "Profile found: '%s' (id=%s…)",
                profile.get("name", "?"),
                str(profile.get("id", ""))[:8],
            )
            return profile

        if isinstance(result, dict):
            content: list = result.get("content") or []
            if content:
                return content[0]

        logger.warning("No active profile found for category '%s'", normalized)
        return None

    # ── 2. Document creation ──────────────────────────────────────────────────

    async def create_document(
        self,
        token: str,
        profile_id: str,
    ) -> dict[str, Any] | None:
        """Create a new document from the given profile.

        Calls ``POST /api/document`` with body ``{"id": profile_id}``.
        Returns the full ``DocumentWithPermissions`` wrapper that the
        controller produces (contains keys ``"document"`` and ``"permission"``).

        Args:
            token: JWT bearer token.
            profile_id: UUID string of the ``DocumentProfile`` to use.

        Returns:
            ``DocumentWithPermissions`` as dict, or ``None`` on failure.
        """
        logger.info("Creating document from profile %s…", str(profile_id)[:8])

        result = await self._make_request(
            "POST",
            "api/document",
            token=token,
            json={"id": str(profile_id)},
        )

        if isinstance(result, dict) and result:
            doc_id: str = str(result.get("id") or "") or str(
                (result.get("document") or {}).get("id") or ""
            )
            logger.info("Document created: id=%s…", doc_id[:8] if doc_id else "?")
            return result

        logger.error("Document creation returned empty response")
        return None

    # ── 3. Attachment upload ──────────────────────────────────────────────────

    async def upload_attachment(
        self,
        token: str,
        document_id: str,
        file_path: str,
        file_name: str | None = None,
    ) -> dict[str, Any] | None:
        """Upload a local file as ``MAIN_ATTACHMENT`` to the document.

        Calls ``POST /api/document/{documentId}/attachment`` as
        ``multipart/form-data``.

        Why not ``_make_request``?  EdmsHttpClient._make_request always sends
        JSON.  Attachment upload requires a multipart body — so we build the
        httpx request directly, reusing ``self._get_client()`` for connection
        pooling.

        Args:
            token: JWT bearer token.
            document_id: Target document UUID string.
            file_path: Absolute local filesystem path to the file.
            file_name: Override display name (defaults to the file's basename).

        Returns:
            ``AttachmentDocumentDto`` as dict on success, ``None`` if the file
            is not found locally.

        Raises:
            httpx.HTTPStatusError: On 4xx/5xx responses from EDMS.
        """
        path = Path(file_path)
        if not path.exists():
            logger.error("Attachment file not found: '%s'", file_path)
            return None

        display_name = (file_name or path.name).strip()
        content_type, _ = mimetypes.guess_type(display_name)
        content_type = content_type or "application/octet-stream"

        url = f"{self.base_url}/api/document/{document_id}/attachment"

        headers = {
            k: v
            for k, v in prepare_auth_headers(token).items()
            if k.lower() != "content-type"
        }

        logger.info(
            "Uploading attachment '%s' (%s) → document %s…",
            display_name,
            content_type,
            document_id[:8],
        )

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
                logger.info("Attachment uploaded (204 No Content)")
                return {}

            result: dict[str, Any] = response.json()
            logger.info(
                "Attachment uploaded: att_id=%s…",
                str(result.get("id", "?"))[:8],
            )
            return result

        except httpx.HTTPStatusError as exc:
            logger.error(
                "Attachment upload failed [HTTP %d]: %s",
                exc.response.status_code,
                exc.response.text[:300],
            )
            raise
        except Exception as exc:
            logger.error("Attachment upload unexpected error: %s", exc, exc_info=True)
            raise
