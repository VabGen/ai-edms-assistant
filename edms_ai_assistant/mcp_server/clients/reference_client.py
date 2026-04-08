# edms_ai_assistant/clients/reference_client.py
import logging
from typing import Any

from .base_client import EdmsHttpClient

logger = logging.getLogger(__name__)


class ReferenceClient(EdmsHttpClient):
    """
    Client for EDMS reference book (справочники) API.
    """

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
        Two-step reference lookup: fts-name → GET /{id} → canonical name.

        Step 1: GET fts-name → получаем id из поискового индекса.
        Step 2: GET /{id}    → получаем эталонное name из справочника.

        Fallback: если GET /{id} недоступен — используем name из fts-ответа.

        Args:
            token: JWT authorization token.
            endpoint: API endpoint prefix (e.g. "region", "district", "city").
            search_name: Human-readable name to search for.
            entity_label: Русское название для логов (e.g. "Район").

        Returns:
            {"id": str, "name": str} из справочника, или None.
        """
        if not search_name or not search_name.strip():
            logger.debug(
                "[REFERENCE-CLIENT] Пропуск поиска %s: пустое значение", entity_label
            )
            return None

        search_query = search_name.strip()

        # ── Шаг 1: FTS-поиск → получаем id ──────────────────────────────────
        try:
            logger.info("[REFERENCE-CLIENT] Поиск %s: '%s'", entity_label, search_query)
            fts_result = await self._make_request(
                "GET",
                f"api/{endpoint}/fts-name",
                token=token,
                params={"fts": search_query},
            )
        except Exception as exc:
            import httpx as _httpx

            if (
                isinstance(exc, _httpx.HTTPStatusError)
                and exc.response.status_code == 404
            ):
                logger.warning(
                    "[REFERENCE-CLIENT] %s не найден в справочнике: '%s'",
                    entity_label,
                    search_query,
                )
            else:
                logger.error(
                    "[REFERENCE-CLIENT] FTS ошибка %s '%s': %s",
                    entity_label,
                    search_query,
                    exc,
                    exc_info=True,
                )
            return None

        if not fts_result:
            logger.warning(
                "[REFERENCE-CLIENT] %s не найден по FTS: '%s'",
                entity_label,
                search_query,
            )
            return None

        fts_data = (
            fts_result[0]
            if isinstance(fts_result, list) and fts_result
            else (fts_result if isinstance(fts_result, dict) else None)
        )
        if not fts_data or not isinstance(fts_data, dict):
            logger.warning(
                "[REFERENCE-CLIENT] Пустой FTS-ответ для %s: '%s'",
                entity_label,
                search_query,
            )
            return None

        entity_id = str(fts_data.get("id", "")).strip()
        if not entity_id or entity_id == "None":
            logger.warning(
                "[REFERENCE-CLIENT] FTS не вернул id для %s: '%s'",
                entity_label,
                search_query,
            )
            return None

        logger.debug("[REFERENCE-CLIENT] %s FTS → id=%s", entity_label, entity_id)

        # ── Шаг 2: GET /{id} → канонический name из справочника ─────────────
        try:
            record = await self._make_request(
                "GET",
                f"api/{endpoint}/{entity_id}",
                token=token,
            )
        except Exception as exc:
            logger.warning(
                "[REFERENCE-CLIENT] GET /%s/%s ошибка: %s — используем FTS name",
                endpoint,
                entity_id,
                exc,
            )
            fts_name = self._extract_canonical_name(fts_data, endpoint) or search_query
            return {"id": entity_id, "name": fts_name}

        if not record or not isinstance(record, dict):
            logger.warning(
                "[REFERENCE-CLIENT] GET /%s/%s вернул пустой ответ — используем FTS name",
                endpoint,
                entity_id,
            )
            fts_name = self._extract_canonical_name(fts_data, endpoint) or search_query
            return {"id": entity_id, "name": fts_name}

        canonical_name = self._extract_canonical_name(record, endpoint) or search_query
        logger.info(
            "[REFERENCE-CLIENT] %s: '%s' → id=%s, name='%s' (canonical)",
            entity_label,
            search_query,
            entity_id,
            canonical_name,
        )
        return {"id": entity_id, "name": canonical_name}

    def _extract_canonical_name(
        self, record: dict[str, Any], endpoint: str
    ) -> str | None:
        """
        Extracts the canonical name from a DB record using endpoint-specific
        field priority from _CANONICAL_NAME_FIELDS.

        Args:
            record: Raw dict returned by the API (fts or GET /{id}).
            endpoint: API endpoint prefix used as lookup key.

        Returns:
            First non-empty string value found, or None.
        """
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
        """Legacy helper: возвращает только id (без name)."""
        result = await self._find_entity_with_name(token, endpoint, name, entity_label)
        return result["id"] if result else None

    # ─────────────────────────────────────────────────────────────────
    # Гео-справочники: {id, canonical name}
    # ─────────────────────────────────────────────────────────────────

    async def find_country_with_name(
        self, token: str, name: str
    ) -> dict[str, str] | None:
        """Two-step country lookup → {id, canonical name}."""
        return await self._find_entity_with_name(token, "country", name, "Страна")

    async def find_region_with_name(
        self, token: str, name: str
    ) -> dict[str, str] | None:
        """Two-step region lookup → {id, canonical name}."""
        return await self._find_entity_with_name(token, "region", name, "Регион")

    async def find_district_with_name(
        self, token: str, name: str
    ) -> dict[str, str] | None:
        """Two-step district lookup → {id, canonical name}."""
        return await self._find_entity_with_name(token, "district", name, "Район")

    async def find_city_with_name(self, token: str, name: str) -> dict[str, str] | None:
        """Two-step city lookup → {id, canonical name}."""
        return await self._find_entity_with_name(token, "city", name, "Город")

    async def find_city_with_hierarchy(
        self, token: str, city_name: str
    ) -> dict[str, Any] | None:
        """
        City lookup with full geo hierarchy in ONE request.

        Protocol:
          Step 1: GET /api/city/fts-name?fts={name}
                  → city id
          Step 2: GET /api/city/{id}?includes=DISTRICT_WITH_REGION
                  → city + district + region embedded in one response

        Real API response structure (confirmed from CityController):
          {
            "id":         "1376963f-...",
            "nameCity":   "Минск",
            "districtId": "9c13bbd7-...",
            "district": {
              "id":          "9c13bbd7-...",
              "nameDistrict":"Минск",
              "regionId":    "b54ce610-...",
              "region": {
                "id":        "b54ce610-...",
                "nameRegion":"Минск"
              }
            }
          }

        Args:
            token: JWT authorization token.
            city_name: City name to search for.

        Returns:
            Dict with id, name, districtId, districtName, regionId, regionName.
            Returns None if city not found.
        """
        if not city_name or not city_name.strip():
            return None

        query = city_name.strip()
        logger.info("[REFERENCE-CLIENT] City hierarchy lookup: '%s'", query)

        # ── Шаг 1: fts-name → id города ─────────────────────────────────────
        try:
            fts_result = await self._make_request(
                "GET",
                "api/city/fts-name",
                token=token,
                params={"fts": query},
            )
        except Exception as exc:
            import httpx as _httpx

            if (
                isinstance(exc, _httpx.HTTPStatusError)
                and exc.response.status_code == 404
            ):
                logger.warning("[REFERENCE-CLIENT] City not found: '%s'", query)
            else:
                logger.error(
                    "[REFERENCE-CLIENT] City FTS error: %s", exc, exc_info=True
                )
            return None

        if not fts_result:
            logger.warning("[REFERENCE-CLIENT] City not found via FTS: '%s'", query)
            return None

        fts_city = fts_result[0] if isinstance(fts_result, list) else fts_result
        if not isinstance(fts_city, dict):
            return None

        city_id = str(fts_city.get("id", "")).strip()
        if not city_id or city_id == "None":
            return None

        # ── Шаг 2: GET /city/{id}?includes=DISTRICT_WITH_REGION ──────────────
        try:
            city_dto = await self._make_request(
                "GET",
                f"api/city/{city_id}",
                token=token,
                params={"includes": "DISTRICT_WITH_REGION"},
            )
        except Exception as exc:
            logger.error("[REFERENCE-CLIENT] City GET error: %s", exc, exc_info=True)
            return None

        if not city_dto or not isinstance(city_dto, dict):
            return None

        # ── Извлекаем поля из вложенной структуры ────────────────────────────
        result: dict[str, Any] = {
            "id": city_id,
            "name": city_dto.get("nameCity") or query,
        }

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

        logger.info(
            "[REFERENCE-CLIENT] Hierarchy resolved: city=%s district=%s region=%s",
            result["name"],
            result.get("districtName", "—"),
            result.get("regionName", "—"),
        )
        return result

    # ─────────────────────────────────────────────────────────────────
    # Legacy API (только id, обратная совместимость)
    # ─────────────────────────────────────────────────────────────────

    async def find_country(self, token: str, name: str) -> str | None:
        """Поиск страны (только ID)."""
        return await self._find_entity_id(token, "country", name, "Страна")

    async def find_region(self, token: str, name: str) -> str | None:
        """Поиск региона (только ID)."""
        return await self._find_entity_id(token, "region", name, "Регион")

    async def find_district(self, token: str, name: str) -> str | None:
        """Поиск района (только ID)."""
        return await self._find_entity_id(token, "district", name, "Район")

    async def find_city(self, token: str, name: str) -> str | None:
        """Поиск города (только ID)."""
        return await self._find_entity_id(token, "city", name, "Город")

    async def find_citizen_type(self, token: str, name: str) -> str | None:
        """Поиск вида обращения (только ID)."""
        return await self._find_entity_id(token, "citizen-type", name, "Вид обращения")

    async def find_correspondent(self, token: str, name: str) -> str | None:
        """Поиск корреспондента (только ID)."""
        return await self._find_entity_id(token, "correspondent", name, "Корреспондент")

    async def find_delivery_method(self, token: str, name: str) -> str | None:
        """Поиск способа доставки с fallback на 'Курьер'."""
        result = await self._find_entity_id(
            token, "delivery-method", name, "Способ доставки"
        )
        if not result and name != "Курьер":
            logger.info("[REFERENCE-CLIENT] Fallback: используем 'Курьер'")
            return await self._find_entity_id(
                token, "delivery-method", "Курьер", "Способ доставки"
            )
        return result

    async def find_department(self, token: str, name: str) -> str | None:
        """Поиск подразделения (только ID)."""
        return await self._find_entity_id(token, "department", name, "Подразделение")

    async def find_group(self, token: str, name: str) -> str | None:
        """Поиск группы (только ID)."""
        return await self._find_entity_id(token, "group", name, "Группа")

    # ─────────────────────────────────────────────────────────────────
    # Тематики (Subject)
    # ─────────────────────────────────────────────────────────────────

    async def get_parent_subjects(self, token: str) -> list[dict]:
        """Получить список родительских тем."""
        try:
            result = await self._make_request(
                "GET",
                "api/subject/parents",
                token=token,
                params={"listAttribute": "true"},
            )
            count = len(result) if isinstance(result, list) else 0
            logger.info("[REFERENCE-CLIENT] Загружено родительских тем: %d", count)
            return result if isinstance(result, list) else []
        except Exception as exc:
            logger.error(
                "[REFERENCE-CLIENT] Ошибка получения родительских тем: %s", exc
            )
            return []

    async def get_child_subjects(self, token: str, parent_id: str) -> list[dict]:
        """Получить дочерние темы по parent_id."""
        try:
            result = await self._make_request(
                "GET",
                f"api/subject/parent/{parent_id}",
                token=token,
            )
            count = len(result) if isinstance(result, list) else 0
            logger.info("[REFERENCE-CLIENT] Дочерних тем для %s: %d", parent_id, count)
            return result if isinstance(result, list) else []
        except Exception as exc:
            logger.error(
                "[REFERENCE-CLIENT] Ошибка получения дочерних тем для %s: %s",
                parent_id,
                exc,
            )
            return []

    async def find_best_subject(self, token: str, text: str) -> str | None:
        """LLM-based two-level subject selection (parent → child)."""
        import re

        from edms_ai_assistant.llm import get_chat_model

        parents = await self.get_parent_subjects(token)
        if not parents:
            logger.warning("[REFERENCE-CLIENT] Родительские темы не загружены")
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
                logger.warning(
                    "[REFERENCE-CLIENT] LLM не вернула номер: %s", choice_text
                )
                return None

            index = int(match.group(0)) - 1
            if not (0 <= index < len(parents)):
                logger.warning(
                    "[REFERENCE-CLIENT] Неверный индекс родительской темы: %d", index
                )
                return None

            parent = parents[index]
            parent_id = str(parent["id"])
            logger.info(
                "[REFERENCE-CLIENT] Родительская тема: %s (ID: %s)",
                parent["name"],
                parent_id,
            )

            children = await self.get_child_subjects(token, parent_id)
            if not children:
                return parent_id

            children_text = "\n".join(
                f"{i + 1}. {c['name']}" for i, c in enumerate(children)
            )
            prompt2 = (
                f"Выбери ОДНУ наиболее подходящую подтему.\n\n"
                f"СПИСОК ПОДТЕМ:\n{children_text}\n\n"
                f"ТЕКСТ ОБРАЩЕНИЯ (фрагмент):\n{text[:800]}\n\n"
                f"Ответь ТОЛЬКО номером (например: 2)"
            )

            response2 = await llm.ainvoke(prompt2)
            choice2_text = response2.content.strip()
            match2 = re.search(r"\d+", choice2_text)
            if not match2:
                logger.warning(
                    "[REFERENCE-CLIENT] LLM не вернула номер подтемы: %s", choice2_text
                )
                return parent_id

            child_index = int(match2.group(0)) - 1
            if not (0 <= child_index < len(children)):
                logger.warning(
                    "[REFERENCE-CLIENT] Неверный индекс подтемы: %d", child_index
                )
                return parent_id

            child = children[child_index]
            child_id = str(child["id"])
            logger.info(
                "[REFERENCE-CLIENT] Дочерняя тема: %s (ID: %s)", child["name"], child_id
            )
            return child_id

        except Exception as exc:
            logger.error(
                "[REFERENCE-CLIENT] Ошибка выбора темы: %s", exc, exc_info=True
            )
            return None
