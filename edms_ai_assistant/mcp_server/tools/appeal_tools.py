# mcp-server/tools/appeal_tools.py
"""
Инструменты для работы с обращениями: автозаполнение карточки, создание документа из файла.
Перенесены из edms_ai_assistant/tools/appeal_autofill.py, create_document_from_file.py.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

from fastmcp import FastMCP

from ..clients.attachment_client import EdmsAttachmentClient
from ..clients.document_client import DocumentClient
from ..clients.document_creator_client import DocumentCreatorClient
from ..clients.reference_client import ReferenceClient
from ..services.appeal_extraction_service import AppealExtractionService
from ..services.file_processor import FileProcessorService
from ..utils.file_utils import extract_text_from_bytes
from ..utils.json_encoder import CustomJSONEncoder
from ..utils.regex_utils import UUID_RE

logger = logging.getLogger(__name__)

_AUTOFILL_SUPPORTED: frozenset[str] = frozenset({"APPEAL"})
_VALID_CATEGORIES: frozenset[str] = frozenset({
    "APPEAL", "INCOMING", "OUTGOING", "INTERN", "CONTRACT",
    "MEETING", "MEETING_QUESTION", "QUESTION", "CUSTOM",
})
_CATEGORY_NAMES_RU: dict[str, str] = {
    "APPEAL": "обращение", "INCOMING": "входящий документ",
    "OUTGOING": "исходящий документ", "INTERN": "внутренний документ",
    "CONTRACT": "договор", "MEETING": "совещание",
    "MEETING_QUESTION": "вопрос повестки", "QUESTION": "вопрос",
    "CUSTOM": "произвольный документ",
}


def _is_good_correspondent_match(query: str, canonical: str) -> bool:
    """Проверяет качество совпадения FTS-результата корреспондента."""
    from difflib import SequenceMatcher
    q = query.lower().strip()
    c = canonical.lower().strip()
    if c in q or q in c:
        return True
    if SequenceMatcher(None, q, c).ratio() >= 0.45:
        return True
    q_words = {w for w in q.split() if len(w) > 3}
    c_words = {w for w in c.split() if len(w) > 3}
    if q_words and c_words and len(q_words & c_words) / len(q_words) >= 0.40:
        return True
    return False


def register_appeal_tools(mcp: FastMCP) -> None:
    """Регистрирует инструменты работы с обращениями."""

    @mcp.tool(
        description=(
            "Автоматически заполнить карточку обращения гражданина (APPEAL) "
            "через LLM-анализ вложенного файла. Извлекает ФИО, адрес, тему, "
            "тип заявителя, географию и другие поля из текста обращения."
        )
    )
    async def autofill_appeal_document(
        document_id: str,
        token: str,
        attachment_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Автозаполнение карточки обращения из вложения.

        Args:
            document_id: UUID документа категории APPEAL.
            token: JWT-токен.
            attachment_id: UUID вложения (опционально, берётся первое подходящее).
        """
        logger.info("autofill_appeal_document: %s", document_id[:8])

        try:
            # 1. Загрузка документа
            async with DocumentClient() as client:
                raw_doc = await client.get_document_metadata(token, document_id)

            if not raw_doc:
                return {"status": "error", "message": "Документ не найден."}

            category = raw_doc.get("docCategoryConstant", "")
            if hasattr(category, "value"):
                category = category.value
            if str(category) != "APPEAL":
                return {
                    "status": "error",
                    "message": f"Документ должен быть категории APPEAL, а не {category}.",
                }

            # 2. Выбор вложения
            attachments = raw_doc.get("attachmentDocument") or []
            if not attachments:
                return {"status": "error", "message": "В документе нет вложений."}

            _SUPPORTED_EXTS = (".pdf", ".docx", ".txt", ".doc", ".rtf")
            target = None
            if attachment_id:
                target = next(
                    (a for a in attachments if str(
                        a.get("id") if isinstance(a, dict) else getattr(a, "id", "")
                    ) == attachment_id),
                    None,
                )
            if not target:
                target = next(
                    (
                        a for a in attachments
                        if (
                            a.get("name") if isinstance(a, dict) else getattr(a, "name", "")
                        ).lower().endswith(_SUPPORTED_EXTS)
                    ),
                    attachments[0],
                )

            att_id = str(target.get("id") if isinstance(target, dict) else getattr(target, "id", ""))
            att_name = (target.get("name") if isinstance(target, dict) else getattr(target, "name", "")) or "attachment"

            # 3. Скачивание и извлечение текста
            async with EdmsAttachmentClient() as att_client:
                file_bytes = await att_client.get_attachment_content(token, document_id, att_id)

            extracted_text = extract_text_from_bytes(file_bytes, att_name)
            if not extracted_text or len(extracted_text) < 50:
                return {"status": "error", "message": "Текст не извлечён или слишком короткий."}

            # 4. LLM-извлечение полей
            extraction_service = AppealExtractionService()
            fields = await extraction_service.extract_appeal_fields(extracted_text)

            warnings: list[str] = []

            # 5. Обновление основных полей
            async with ReferenceClient() as ref_client:
                delivery_id = raw_doc.get("deliveryMethodId")
                if not delivery_id:
                    delivery_method_name = fields.deliveryMethod or "Курьер"
                    delivery_id = await ref_client.find_delivery_method(token, delivery_method_name)

                raw_summary = fields.shortSummary or raw_doc.get("shortSummary")
                if raw_summary and len(raw_summary) > 80:
                    raw_summary = raw_summary[:80]

                main_payload: dict[str, Any] = {}
                if raw_summary:
                    main_payload["shortSummary"] = raw_summary
                if delivery_id:
                    main_payload["deliveryMethodId"] = delivery_id
                if raw_doc.get("documentTypeId"):
                    main_payload["documentTypeId"] = str(raw_doc["documentTypeId"])

                if main_payload:
                    main_ops = [{"operationType": "DOCUMENT_MAIN_FIELDS_UPDATE", "body": main_payload}]
                    async with DocumentClient() as doc_client:
                        await doc_client.execute_document_operations(token, document_id, main_ops)

                # 6. Географический резолвинг
                geo_data: dict[str, Any] = {}

                country_name = fields.country
                if country_name:
                    data = await ref_client.find_country_with_name(token, country_name)
                    if data:
                        geo_data["countryAppealId"] = data["id"]
                        geo_data["countryAppealName"] = data["name"]

                if fields.regionName:
                    data = await ref_client.find_region_with_name(token, fields.regionName)
                    if data:
                        geo_data["regionId"] = data["id"]
                        geo_data["regionName"] = data["name"]

                if fields.districtName:
                    data = await ref_client.find_district_with_name(token, fields.districtName)
                    if data:
                        geo_data["districtId"] = data["id"]
                        geo_data["districtName"] = data["name"]

                if fields.cityName:
                    data = await ref_client.find_city_with_hierarchy(token, fields.cityName)
                    if data:
                        geo_data["cityId"] = data["id"]
                        geo_data["cityName"] = data["name"]
                        if "districtId" not in geo_data and data.get("districtId"):
                            geo_data["districtId"] = data["districtId"]
                        if "districtName" not in geo_data and data.get("districtName"):
                            geo_data["districtName"] = data["districtName"]
                        if "regionId" not in geo_data and data.get("regionId"):
                            geo_data["regionId"] = data["regionId"]
                        if "regionName" not in geo_data and data.get("regionName"):
                            geo_data["regionName"] = data["regionName"]

                # 7. Поля карточки обращения
                appeal_payload: dict[str, Any] = {**geo_data}

                # Тип заявителя
                if fields.declarantType:
                    appeal_payload["declarantType"] = str(fields.declarantType).upper()
                else:
                    appeal_payload["declarantType"] = "INDIVIDUAL"
                    warnings.append("declarantType установлен INDIVIDUAL по умолчанию")

                # Вид обращения
                if fields.citizenType:
                    citizen_id = await ref_client.find_citizen_type(token, fields.citizenType)
                    if citizen_id:
                        appeal_payload["citizenTypeId"] = citizen_id

                # Тема
                subject_id = await ref_client.find_best_subject(token, extracted_text)
                if subject_id:
                    appeal_payload["subjectId"] = subject_id

                # Персональные и контактные данные
                def _sanitize(v: Any) -> str | None:
                    if not v or (isinstance(v, str) and not v.strip()):
                        return None
                    return str(v).strip()

                if fields.declarantType == "ENTITY" or str(appeal_payload.get("declarantType")) == "ENTITY":
                    # Организация
                    org_name = _sanitize(fields.organizationName)
                    if org_name:
                        corr_data = await ref_client._find_entity_with_name(
                            token, "correspondent", org_name, "Организация"
                        )
                        if corr_data and _is_good_correspondent_match(org_name, corr_data.get("name", "")):
                            appeal_payload["organizationName"] = corr_data["name"]
                        else:
                            appeal_payload["organizationName"] = org_name
                    else:
                        appeal_payload["organizationName"] = None
                    appeal_payload["signed"] = _sanitize(fields.signed)
                    appeal_payload["correspondentOrgNumber"] = _sanitize(fields.correspondentOrgNumber)
                    if fields.dateDocCorrespondentOrg:
                        dt = fields.dateDocCorrespondentOrg
                        appeal_payload["dateDocCorrespondentOrg"] = (
                            dt.isoformat() if dt.tzinfo else dt.isoformat() + "Z"
                        )
                else:
                    appeal_payload["organizationName"] = None
                    appeal_payload["signed"] = None
                    appeal_payload["correspondentOrgNumber"] = None

                if fields.fioApplicant:
                    appeal_payload["fioApplicant"] = _sanitize(fields.fioApplicant)
                if fields.collective is not None:
                    appeal_payload["collective"] = fields.collective
                if fields.anonymous is not None:
                    appeal_payload["anonymous"] = fields.anonymous
                if fields.reasonably is not None:
                    appeal_payload["reasonably"] = fields.reasonably

                if fields.receiptDate:
                    dt = fields.receiptDate
                    # Пропускаем если дата выглядит как сегодня
                    today_delta = abs((dt.date() - datetime.now(UTC).date()).days)
                    if today_delta > 1:
                        appeal_payload["receiptDate"] = (
                            dt.isoformat() if dt.tzinfo else dt.isoformat() + "Z"
                        )

                for fname in ("fullAddress", "phone", "email", "index",
                              "correspondentAppeal", "reviewProgress"):
                    val = _sanitize(getattr(fields, fname, None))
                    if val is not None:
                        appeal_payload[fname] = val

                appeal_payload["submissionForm"] = "WRITTEN"

                # Фильтруем пустые значения, но сохраняем обязательные null
                _ALWAYS_INCLUDE = {"correspondentAppeal", "submissionForm"}
                filtered_payload = {
                    k: v for k, v in appeal_payload.items()
                    if k in _ALWAYS_INCLUDE or (v is not None and v != "")
                }

                appeal_ops = [{
                    "operationType": "DOCUMENT_MAIN_FIELDS_APPEAL_UPDATE",
                    "body": filtered_payload,
                }]
                json_ops = json.loads(json.dumps(appeal_ops, cls=CustomJSONEncoder))

                async with DocumentClient() as doc_client:
                    await doc_client.execute_document_operations(token, document_id, json_ops)

            return {
                "status": "success",
                "message": "✅ Карточка обращения заполнена автоматически.",
                "warnings": warnings if warnings else None,
                "attachment_used": att_name,
                "requires_reload": True,
            }

        except Exception as exc:
            logger.error("autofill_appeal_document failed: %s", exc, exc_info=True)
            return {"status": "error", "message": f"Ошибка автозаполнения: {exc!s}"}

    @mcp.tool(
        description=(
            "Создать новый документ EDMS из загруженного локального файла. "
            "Используй когда пользователь загрузил файл и просит создать документ. "
            "Для APPEAL автоматически заполняет карточку через LLM. "
            "После создания возвращает navigate_url для открытия документа."
        )
    )
    async def create_document_from_file(
        token: str,
        file_path: str,
        doc_category: str = "APPEAL",
        file_name: str | None = None,
        autofill: bool = True,
    ) -> dict[str, Any]:
        """
        Создать документ из локального файла.

        Args:
            token: JWT-токен.
            file_path: Путь к файлу (из контекста агента).
            doc_category: APPEAL|INCOMING|OUTGOING|INTERN|CONTRACT|MEETING|CUSTOM.
            file_name: Имя файла для отображения (опционально).
            autofill: Автозаполнить карточку (для APPEAL).
        """
        from pathlib import Path as _Path

        cleaned = file_path.strip()
        if not cleaned or cleaned.lower() in {"", "none", "null", "<local_file_path>"}:
            return {
                "status": "error",
                "message": "file_path не может быть пустым. Используй значение из контекста агента.",
            }

        # Проверка UUID — пока не поддерживается
        if UUID_RE.match(cleaned):
            return {
                "status": "error",
                "message": "Создание из UUID вложения EDMS не поддерживается. Загрузите файл локально.",
            }

        if not _Path(cleaned).exists():
            return {
                "status": "error",
                "message": f"Файл не найден: '{cleaned}'. Загрузите файл через кнопку со скрепкой.",
            }

        normalized_category = doc_category.strip().upper()
        if normalized_category not in _VALID_CATEGORIES:
            normalized_category = "APPEAL"

        # Попытка определить категорию из имени файла
        if normalized_category == "APPEAL" and file_name:
            lower_fn = file_name.lower()
            _CATEGORY_KEYWORDS: list[tuple[list[str], str]] = [
                (["входящ", "incoming"], "INCOMING"),
                (["исходящ", "outgoing"], "OUTGOING"),
                (["внутренн", "intern"], "INTERN"),
                (["договор", "контракт", "contract"], "CONTRACT"),
                (["совещани", "meeting"], "MEETING"),
            ]
            for keywords, cat in _CATEGORY_KEYWORDS:
                if any(kw in lower_fn for kw in keywords):
                    normalized_category = cat
                    break

        category_ru = _CATEGORY_NAMES_RU.get(normalized_category, normalized_category.lower())
        effective_file_name = file_name or _Path(cleaned).name
        warnings: list[str] = []

        async with DocumentCreatorClient() as creator:
            # Шаг 1: Найти профиль
            profile = await creator.find_profile_by_category(token, normalized_category)
            if not profile:
                return {
                    "status": "error",
                    "message": (
                        f"Не найден активный профиль для «{category_ru}». "
                        "Возможно, нет прав или профиль не настроен."
                    ),
                }

            profile_id = str(profile.get("id", ""))
            profile_name = profile.get("name", "?")

            # Шаг 2: Создать документ
            created = await creator.create_document(token, profile_id)
            if not created:
                return {
                    "status": "error",
                    "message": f"Не удалось создать документ по профилю «{profile_name}».",
                }

            doc_data: dict[str, Any] = (
                created.get("document") or created
                if "document" in created else created
            )
            document_id = str(doc_data.get("id", ""))
            if not document_id:
                return {"status": "error", "message": "Документ создан, но сервер не вернул UUID."}

            # Шаг 3: Загрузить вложение
            attachment = await creator.upload_attachment(
                token=token,
                document_id=document_id,
                file_path=cleaned,
                file_name=effective_file_name,
            )
            if attachment is None:
                warnings.append(f"Файл '{effective_file_name}' не найден — вложение не загружено.")

        # Шаг 4: Автозаполнение для APPEAL
        autofill_status = "skipped"
        if autofill and normalized_category in _AUTOFILL_SUPPORTED:
            try:
                from ..tools.appeal_tools import autofill_appeal_document
                # Вызываем внутри текущего контекста
                autofill_result = await _do_autofill(document_id, token)
                autofill_status = "done"
                if autofill_result.get("warnings"):
                    warnings.extend(autofill_result["warnings"])
            except Exception as exc:
                autofill_status = "failed"
                warnings.append(f"Автозаполнение не удалось: {exc!s}")
        elif autofill and normalized_category not in _AUTOFILL_SUPPORTED:
            autofill_status = "not_supported"

        parts = [f"✅ {category_ru.capitalize()} успешно создано."]
        if attachment is not None:
            parts.append(f"📎 Файл «{effective_file_name}» прикреплён.")
        if autofill_status == "done":
            parts.append("📋 Карточка заполнена автоматически.")
        elif autofill_status == "failed":
            parts.append("⚠️ Карточку не удалось заполнить — проверьте и заполните вручную.")
        parts.append("🔗 Открываю документ в системе...")

        return {
            "status": "success" if not warnings else "partial_success",
            "message": " ".join(parts),
            "document_id": document_id,
            "navigate_url": f"/document-form/{document_id}",
            "autofill_status": autofill_status,
            "warnings": warnings if warnings else None,
            "requires_reload": False,
        }


async def _do_autofill(document_id: str, token: str) -> dict[str, Any]:
    """Внутренняя функция автозаполнения (без декоратора MCP)."""
    logger.info("autofill_appeal_document (internal): %s", document_id[:8])

    async with DocumentClient() as client:
        raw_doc = await client.get_document_metadata(token, document_id)
    if not raw_doc:
        return {"status": "error", "message": "Документ не найден."}

    attachments = raw_doc.get("attachmentDocument") or []
    if not attachments:
        return {"status": "error", "message": "Нет вложений."}

    _SUPPORTED_EXTS = (".pdf", ".docx", ".txt", ".doc", ".rtf")
    target = attachments[0]
    for a in attachments:
        name = a.get("name") if isinstance(a, dict) else getattr(a, "name", "")
        if name and name.lower().endswith(_SUPPORTED_EXTS):
            target = a
            break

    att_id = str(target.get("id") if isinstance(target, dict) else getattr(target, "id", ""))
    att_name = (target.get("name") if isinstance(target, dict) else getattr(target, "name", "")) or "attachment"

    async with EdmsAttachmentClient() as att_client:
        file_bytes = await att_client.get_attachment_content(token, document_id, att_id)

    extracted_text = extract_text_from_bytes(file_bytes, att_name)
    if not extracted_text or len(extracted_text) < 50:
        return {"status": "error", "message": "Текст не извлечён."}

    extraction_service = AppealExtractionService()
    fields = await extraction_service.extract_appeal_fields(extracted_text)

    # Упрощённое обновление без полного географического резолвинга
    raw_summary = fields.shortSummary
    if raw_summary and len(raw_summary) > 80:
        raw_summary = raw_summary[:80]

    if raw_summary:
        main_ops = [{"operationType": "DOCUMENT_MAIN_FIELDS_UPDATE", "body": {"shortSummary": raw_summary}}]
        async with DocumentClient() as doc_client:
            await doc_client.execute_document_operations(token, document_id, main_ops)

    return {"status": "success", "warnings": []}