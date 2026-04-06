# mcp-server/clients/reference_client.py
"""
EDMS Reference Client — справочники (гео, корреспонденты, типы обращений).
Перенесён из edms_ai_assistant/clients/reference_client.py.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from .base_client import EdmsHttpClient

logger = logging.getLogger(__name__)


class ReferenceClient(EdmsHttpClient):
    """Клиент для EDMS справочников (география, доставка, группы и т.д.)."""

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
        """
        Двухшаговый поиск: fts-name → GET /{id} → каноническое имя.

        Returns:
            {"id": str, "name": str} или None.
        """
        if not search_name or not search_name.strip():
            return None

        search_query = search_name.strip()
        try:
            fts_result = await self._make_request(
                "GET", f"api/{endpoint}/fts-name",
                token=token, params={"fts": search_query},
            )
        except Exception as exc:
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 404:
                logger.warning("%s не найден: '%s'", entity_label, search_query)
            else:
                logger.error("FTS ошибка %s '%s': %s", entity_label, search_query, exc)
            return None

        if not fts_result:
            return None

        fts_data = (
            fts_result[0]
            if isinstance(fts_result, list) and fts_result
            else (fts_result if isinstance(fts_result, dict) else None)
        )
        if not fts_data:
            return None

        entity_id = str(fts_data.get("id", "")).strip()
        if not entity_id or entity_id == "None":
            return None

        try:
            record = await self._make_request("GET", f"api/{endpoint}/{entity_id}", token=token)
        except Exception as exc:
            logger.warning("GET /%s/%s ошибка: %s", endpoint, entity_id, exc)
            fts_name = self._extract_canonical_name(fts_data, endpoint) or search_query
            return {"id": entity_id, "name": fts_name}

        if not record or not isinstance(record, dict):
            fts_name = self._extract_canonical_name(fts_data, endpoint) or search_query
            return {"id": entity_id, "name": fts_name}

        canonical_name = self._extract_canonical_name(record, endpoint) or search_query
        return {"id": entity_id, "name": canonical_name}

    def _extract_canonical_name(self, record: dict[str, Any], endpoint: str) -> str | None:
        """Извлекает каноническое имя из записи справочника."""
        priority_fields = self._CANONICAL_NAME_FIELDS.get(
            endpoint, ("name", "shortName", "fullName")
        )
        for field in priority_fields:
            val = record.get(field)
            if val and isinstance(val, str) and val.strip():
                return val.strip()
        return None

    async def _find_entity_id(
        self, token: str, endpoint: str, name: str, entity_label: str
    ) -> str | None:
        """Возвращает только id (без имени)."""
        result = await self._find_entity_with_name(token, endpoint, name, entity_label)
        return result["id"] if result else None

    async def find_country_with_name(self, token: str, name: str) -> dict[str, str] | None:
        return await self._find_entity_with_name(token, "country", name, "Страна")

    async def find_region_with_name(self, token: str, name: str) -> dict[str, str] | None:
        return await self._find_entity_with_name(token, "region", name, "Регион")

    async def find_district_with_name(self, token: str, name: str) -> dict[str, str] | None:
        return await self._find_entity_with_name(token, "district", name, "Район")

    async def find_city_with_name(self, token: str, name: str) -> dict[str, str] | None:
        return await self._find_entity_with_name(token, "city", name, "Город")

    async def find_city_with_hierarchy(
        self, token: str, city_name: str
    ) -> dict[str, Any] | None:
        """Поиск города с иерархией (район + регион) за один запрос."""
        if not city_name or not city_name.strip():
            return None

        query = city_name.strip()
        try:
            fts_result = await self._make_request(
                "GET", "api/city/fts-name", token=token, params={"fts": query}
            )
        except Exception:
            return None

        if not fts_result:
            return None

        fts_city = fts_result[0] if isinstance(fts_result, list) else fts_result
        if not isinstance(fts_city, dict):
            return None

        city_id = str(fts_city.get("id", "")).strip()
        if not city_id:
            return None

        try:
            city_dto = await self._make_request(
                "GET", f"api/city/{city_id}",
                token=token, params={"includes": "DISTRICT_WITH_REGION"},
            )
        except Exception:
            return None

        if not city_dto or not isinstance(city_dto, dict):
            return None

        result: dict[str, Any] = {"id": city_id, "name": city_dto.get("nameCity") or query}
        district = city_dto.get("district")
        if district and isinstance(district, dict):
            result["districtId"] = district.get("id")
            result["districtName"] = district.get("nameDistrict")
            region = district.get("region")
            if region and isinstance(region, dict):
                result["regionId"] = region.get("id")
                result["regionName"] = region.get("nameRegion")

        return result

    async def find_citizen_type(self, token: str, name: str) -> str | None:
        return await self._find_entity_id(token, "citizen-type", name, "Вид обращения")

    async def find_correspondent(self, token: str, name: str) -> str | None:
        return await self._find_entity_id(token, "correspondent", name, "Корреспондент")

    async def find_delivery_method(self, token: str, name: str) -> str | None:
        result = await self._find_entity_id(token, "delivery-method", name, "Способ доставки")
        if not result and name != "Курьер":
            return await self._find_entity_id(token, "delivery-method", "Курьер", "Способ доставки")
        return result

    async def get_parent_subjects(self, token: str) -> list[dict]:
        """Родительские темы: GET api/subject/parents."""
        try:
            result = await self._make_request(
                "GET", "api/subject/parents",
                token=token, params={"listAttribute": "true"},
            )
            return result if isinstance(result, list) else []
        except Exception as exc:
            logger.error("Ошибка получения тем: %s", exc)
            return []

    async def get_child_subjects(self, token: str, parent_id: str) -> list[dict]:
        """Дочерние темы: GET api/subject/parent/{parent_id}."""
        try:
            result = await self._make_request(
                "GET", f"api/subject/parent/{parent_id}", token=token
            )
            return result if isinstance(result, list) else []
        except Exception as exc:
            logger.error("Ошибка получения подтем для %s: %s", parent_id, exc)
            return []

    async def find_best_subject(self, token: str, text: str) -> str | None:
        """LLM-выбор лучшей темы (двухуровневый: родитель → дочерняя)."""
        import re
        from ..llm import get_llm_response

        parents = await self.get_parent_subjects(token)
        if not parents:
            return None

        themes_text = "\n".join(f"{i + 1}. {s['name']}" for i, s in enumerate(parents))
        prompt = (
            f"Выбери ОДНУ наиболее подходящую тему.\n\nСПИСОК ТЕМ:\n{themes_text}\n\n"
            f"ТЕКСТ:\n{text[:800]}\n\nОтвечай ТОЛЬКО номером (например: 3)"
        )

        try:
            response_text = await get_llm_response(prompt)
            match = re.search(r"\d+", response_text)
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
                f"Выбери ОДНУ подтему.\n\nСПИСОК:\n{children_text}\n\n"
                f"ТЕКСТ:\n{text[:800]}\n\nОтвечай ТОЛЬКО номером (например: 2)"
            )
            response2 = await get_llm_response(prompt2)
            match2 = re.search(r"\d+", response2)
            if not match2:
                return parent_id
            child_index = int(match2.group(0)) - 1
            if not (0 <= child_index < len(children)):
                return parent_id
            return str(children[child_index]["id"])
        except Exception as exc:
            logger.error("Ошибка выбора темы: %s", exc)
            return None