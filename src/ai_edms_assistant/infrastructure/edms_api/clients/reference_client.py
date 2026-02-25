# src/ai_edms_assistant/infrastructure/edms_api/clients/reference_client.py
"""
EDMS Reference HTTP Client.

Wraps lookup (справочник) endpoints: country, region, district, city,
citizen-type, correspondent, delivery-method, subject, department, group.

All methods return the entity ID (str UUID) or None — callers only need
the ID to include in document operation payloads.
"""

from __future__ import annotations

import structlog
from typing import Any

from ..http_client import EdmsHttpClient

logger = structlog.get_logger(__name__)


class EdmsReferenceClient(EdmsHttpClient):
    """
    Low-level client for EDMS reference/lookup (справочник) endpoints.

    Pattern: POST /api/{entity}/fts-name → returns first match with ID.
    Fallback logic (e.g. 'Курьер') is implemented here to avoid duplication.
    """

    # ── Generic lookup ────────────────────────────────────────────────────────

    async def _find_by_fts(
        self,
        entity: str,
        name: str,
        token: str,
        id_field: str = "id",
    ) -> str | None:
        """
        POST /api/{entity}/fts-name — FTS lookup, returns ID string or None.

        Args:
            entity:   API sub-path (e.g. "country", "delivery-method").
            name:     Search string.
            token:    JWT bearer token.
            id_field: JSON key containing the entity ID (default "id").
        """
        try:
            data = await self._make_request(
                "POST",
                f"api/{entity}/fts-name",
                token=token,
                json={"search": name},
            )
            if data and isinstance(data, dict):
                return str(data[id_field]) if data.get(id_field) else None
            return None
        except Exception as exc:
            logger.warning(
                "reference_fts_failed", entity=entity, name=name, error=str(exc)
            )
            return None

    # ── Geography ─────────────────────────────────────────────────────────────

    async def find_country(self, name: str, token: str) -> str | None:
        """POST /api/country/fts-name → countryId."""
        return await self._find_by_fts("country", name, token)

    async def find_region(self, name: str, token: str) -> str | None:
        """POST /api/region/fts-name → regionId."""
        return await self._find_by_fts("region", name, token)

    async def find_district(self, name: str, token: str) -> str | None:
        """POST /api/district/fts-name → districtId."""
        return await self._find_by_fts("district", name, token)

    async def find_city(self, name: str, token: str) -> str | None:
        """POST /api/city/fts-name → cityId."""
        return await self._find_by_fts("city", name, token)

    # ── Appeal lookups ────────────────────────────────────────────────────────

    async def find_citizen_type(self, name: str, token: str) -> str | None:
        """POST /api/citizen-type/fts-name → citizenTypeId."""
        return await self._find_by_fts("citizen-type", name, token)

    async def find_correspondent(self, name: str, token: str) -> str | None:
        """POST /api/correspondent/fts-name → correspondentId."""
        return await self._find_by_fts("correspondent", name, token)

    async def find_delivery_method(
        self,
        name: str,
        token: str,
        fallback: str = "Курьер",
    ) -> str | None:
        """
        POST /api/delivery-method/fts-name → deliveryMethodId.

        Falls back to 'Курьер' when name not found and fallback != name.
        """
        result = await self._find_by_fts("delivery-method", name, token)
        if not result and name != fallback:
            logger.info("delivery_method_fallback", original=name, fallback=fallback)
            result = await self._find_by_fts("delivery-method", fallback, token)
        return result

    # ── Subjects ──────────────────────────────────────────────────────────────

    async def get_parent_subjects(self, token: str) -> list[dict[str, Any]]:
        """GET /api/subject/parents?listAttribute=true"""
        try:
            data = await self._make_request(
                "GET",
                "api/subject/parents",
                token=token,
                params={"listAttribute": "true"},
            )
            return data if isinstance(data, list) else []
        except Exception as exc:
            logger.error("subjects_parent_failed", error=str(exc))
            return []

    async def get_child_subjects(
        self,
        parent_id: str,
        token: str,
    ) -> list[dict[str, Any]]:
        """GET /api/subject/children?parentId={parent_id}"""
        try:
            data = await self._make_request(
                "GET",
                "api/subject/children",
                token=token,
                params={"parentId": parent_id},
            )
            return data if isinstance(data, list) else []
        except Exception as exc:
            logger.error(
                "subjects_children_failed", parent_id=parent_id, error=str(exc)
            )
            return []
