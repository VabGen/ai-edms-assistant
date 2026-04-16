# edms_ai_assistant/orchestrator/clients/reference_client.py
"""
Клиент справочников EDMS.

ИСПРАВЛЕНИЕ: импорт get_chat_model теперь из orchestrator.llm (не из несуществующего llm).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from edms_ai_assistant.orchestrator.clients.base_client import EdmsHttpClient

logger = logging.getLogger(__name__)


class ReferenceClient(EdmsHttpClient):
    """Клиент для работы со справочниками (география, тематики, корреспонденты)."""

    _CANONICAL_NAME_FIELDS: dict[str, tuple] = {
        "country": ("fullName", "name", "shortName"),
        "region": ("nameRegion", "name", "shortName"),
        "district": ("nameDistrict", "name", "shortName"),
        "city": ("nameCity", "name", "shortName"),
        "citizen-type": ("name", "shortName"),
        "correspondent": ("name", "fullName", "shortName"),
        "delivery-method": ("name", "shortName"),
        "department": ("name", "fullName", "shortName"),
        "group": ("name", "shortName"),
    }

    async def _find_entity_with_name(
        self,
        token: str,
        endpoint: str,
        search_name: str,
        entity_label: str,
    ) -> dict[str, str] | None:
        """Двухшаговый поиск в справочнике: fts-name → GET /{id} → канонический name.

        Args:
            token: JWT-токен.
            endpoint: Эндпоинт справочника (напр. "region").
            search_name: Название для поиска.
            entity_label: Русское название для логов.

        Returns:
            {"id": str, "name": str} или None.
        """
        if not search_name or not search_name.strip():
            return None

        search_query = search_name.strip()
        try:
            fts_result = await self._make_request(
                "GET", f"api/{endpoint}/fts-name", token=token, params={"fts": search_query}
            )
        except Exception as exc:
            import httpx as _httpx
            if isinstance(exc, _httpx.HTTPStatusError) and exc.response.status_code == 404:
                logger.warning("[REF] %s не найден: '%s'", entity_label, search_query)
            else:
                logger.error("[REF] FTS ошибка %s '%s': %s", entity_label, search_query, exc)
            return None

        if not fts_result:
            return None

        fts_data = fts_result[0] if isinstance(fts_result, list) and fts_result else (
            fts_result if isinstance(fts_result, dict) else None
        )
        if not fts_data:
            return None

        entity_id = str(fts_data.get("id", "")).strip()
        if not entity_id or entity_id == "None":
            return None

        try:
            record = await self._make_request("GET", f"api/{endpoint}/{entity_id}", token=token)
        except Exception as exc:
            logger.warning("[REF] GET /%s/%s ошибка: %s", endpoint, entity_id, exc)
            fts_name = self._extract_canonical_name(fts_data, endpoint) or search_query
            return {"id": entity_id, "name": fts_name}

        if not record or not isinstance(record, dict):
            fts_name = self._extract_canonical_name(fts_data, endpoint) or search_query
            return {"id": entity_id, "name": fts_name}

        canonical_name = self._extract_canonical_name(record, endpoint) or search_query
        return {"id": entity_id, "name": canonical_name}

    def _extract_canonical_name(self, record: dict[str, Any], endpoint: str) -> str | None:
        """Извлекает каноническое название из записи справочника."""
        priority_fields = self._CANONICAL_NAME_FIELDS.get(endpoint, ("name", "shortName", "fullName"))
        for field in priority_fields:
            val = record.get(field)
            if val and isinstance(val, str) and val.strip():
                return val.strip()
        return None

    async def _find_entity_id(self, token: str, endpoint: str, name: str, entity_label: str) -> str | None:
        """Legacy-хелпер: возвращает только id."""
        result = await self._find_entity_with_name(token, endpoint, name, entity_label)
        return result["id"] if result else None

    # ── Гео-справочники ───────────────────────────────────────────────────────

    async def find_country_with_name(self, token: str, name: str) -> dict[str, str] | None:
        return await self._find_entity_with_name(token, "country", name, "Страна")

    async def find_region_with_name(self, token: str, name: str) -> dict[str, str] | None:
        return await self._find_entity_with_name(token, "region", name, "Регион")

    async def find_district_with_name(self, token: str, name: str) -> dict[str, str] | None:
        return await self._find_entity_with_name(token, "district", name, "Район")

    async def find_city_with_name(self, token: str, name: str) -> dict[str, str] | None:
        return await self._find_entity_with_name(token, "city", name, "Город")

    async def find_city_with_hierarchy(self, token: str, city_name: str) -> dict[str, Any] | None:
        """Поиск города с полной иерархией (город→район→регион) за 2 запроса."""
        if not city_name or not city_name.strip():
            return None
        query = city_name.strip()
        try:
            fts_result = await self._make_request(
                "GET", "api/city/fts-name", token=token, params={"fts": query}
            )
        except Exception as exc:
            logger.error("[REF] City FTS error: %s", exc)
            return None

        if not fts_result:
            return None
        fts_city = fts_result[0] if isinstance(fts_result, list) else fts_result
        if not isinstance(fts_city, dict):
            return None
        city_id = str(fts_city.get("id", "")).strip()
        if not city_id or city_id == "None":
            return None

        try:
            city_dto = await self._make_request(
                "GET", f"api/city/{city_id}", token=token,
                params={"includes": "DISTRICT_WITH_REGION"}
            )
        except Exception as exc:
            logger.error("[REF] City GET error: %s", exc)
            return None

        if not city_dto or not isinstance(city_dto, dict):
            return None

        result: dict[str, Any] = {"id": city_id, "name": city_dto.get("nameCity") or query}
        district = city_dto.get("district")
        district_id = str(city_dto.get("districtId", "")).strip() or None
        if district and isinstance(district, dict):
            result["districtId"] = district.get("id") or district_id
            result["districtName"] = district.get("nameDistrict") or None
            region = district.get("region")
            region_id = str(district.get("regionId", "")).strip() or None
            if region and isinstance(region, dict):
                result["regionId"] = region.get("id") or region_id
                result["regionName"] = region.get("nameRegion") or None
            elif region_id:
                result["regionId"] = region_id
        elif district_id:
            result["districtId"] = district_id

        return result

    # ── Legacy API (только id) ────────────────────────────────────────────────

    async def find_country(self, token: str, name: str) -> str | None:
        return await self._find_entity_id(token, "country", name, "Страна")

    async def find_region(self, token: str, name: str) -> str | None:
        return await self._find_entity_id(token, "region", name, "Регион")

    async def find_district(self, token: str, name: str) -> str | None:
        return await self._find_entity_id(token, "district", name, "Район")

    async def find_city(self, token: str, name: str) -> str | None:
        return await self._find_entity_id(token, "city", name, "Город")

    async def find_citizen_type(self, token: str, name: str) -> str | None:
        return await self._find_entity_id(token, "citizen-type", name, "Вид обращения")

    async def find_correspondent(self, token: str, name: str) -> str | None:
        return await self._find_entity_id(token, "correspondent", name, "Корреспондент")

    async def find_delivery_method(self, token: str, name: str) -> str | None:
        result = await self._find_entity_id(token, "delivery-method", name, "Способ доставки")
        if not result and name != "Курьер":
            return await self._find_entity_id(token, "delivery-method", "Курьер", "Способ доставки")
        return result

    async def find_department(self, token: str, name: str) -> str | None:
        return await self._find_entity_id(token, "department", name, "Подразделение")

    async def find_group(self, token: str, name: str) -> str | None:
        return await self._find_entity_id(token, "group", name, "Группа")

    # ── Тематики ──────────────────────────────────────────────────────────────

    async def get_parent_subjects(self, token: str) -> list[dict]:
        try:
            result = await self._make_request(
                "GET", "api/subject/parents", token=token, params={"listAttribute": "true"}
            )
            return result if isinstance(result, list) else []
        except Exception as exc:
            logger.error("[REF] Ошибка получения родительских тем: %s", exc)
            return []

    async def get_child_subjects(self, token: str, parent_id: str) -> list[dict]:
        try:
            result = await self._make_request("GET", f"api/subject/parent/{parent_id}", token=token)
            return result if isinstance(result, list) else []
        except Exception as exc:
            logger.error("[REF] Ошибка получения дочерних тем для %s: %s", parent_id, exc)
            return []

    async def find_best_subject(self, token: str, text: str) -> str | None:
        """LLM-выбор тематики обращения: двухуровневый (родитель → дочерняя)."""
        # ИСПРАВЛЕНО: импортируем get_chat_model из правильного модуля
        from edms_ai_assistant.orchestrator.llm import get_chat_model

        parents = await self.get_parent_subjects(token)
        if not parents:
            return None

        themes_text = "\n".join(f"{i + 1}. {s['name']}" for i, s in enumerate(parents))
        llm = get_chat_model()

        prompt = (
            f"Выбери ОДНУ наиболее подходящую тему для обращения.\n\n"
            f"СПИСОК ТЕМ:\n{themes_text}\n\n"
            f"ТЕКСТ ОБРАЩЕНИЯ (фрагмент):\n{text[:800]}\n\n"
            f"Ответь ТОЛЬКО номером (например: 3)"
        )

        try:
            response = await llm.ainvoke(prompt)
            choice_text = response.content.strip()
            match = re.search(r"\d+", choice_text)
            if not match:
                return None
            index = int(match.group(0)) - 1
            if not (0 <= index < len(parents)):
                return None
            parent = parents[index]
            parent_id = str(parent["id"])

            children = await self.get_child_subjects(token, parent_id)
            if not children:
                return parent_id

            children_text = "\n".join(f"{i + 1}. {c['name']}" for i, c in enumerate(children))
            prompt2 = (
                f"Выбери ОДНУ наиболее подходящую подтему.\n\n"
                f"СПИСОК ПОДТЕМ:\n{children_text}\n\n"
                f"ТЕКСТ ОБРАЩЕНИЯ:\n{text[:800]}\n\n"
                f"Ответь ТОЛЬКО номером."
            )
            response2 = await llm.ainvoke(prompt2)
            match2 = re.search(r"\d+", response2.content.strip())
            if not match2:
                return parent_id
            child_index = int(match2.group(0)) - 1
            if not (0 <= child_index < len(children)):
                return parent_id
            return str(children[child_index]["id"])

        except Exception as exc:
            logger.error("[REF] Ошибка выбора темы: %s", exc, exc_info=True)
            return None