# edms_ai_assistant/mcp_server/tools/appeal_tools.py
"""
Инструменты для работы с обращениями граждан: автозаполнение и создание из файла.

Переписано под монорепо:
  - FastMCP @mcp.tool() вместо langchain_core @tool()
  - llm_client.get_llm_response() вместо get_chat_model()
  - Сырые dict вместо generated DocumentDto
  - Все импорты из edms_ai_assistant.*
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from edms_ai_assistant.mcp_server.clients.attachment_client import EdmsAttachmentClient
from edms_ai_assistant.mcp_server.clients.document_client import DocumentClient
from edms_ai_assistant.mcp_server.clients.document_creator_client import (
    DocumentCreatorClient,
)
from edms_ai_assistant.mcp_server.clients.reference_client import ReferenceClient
from edms_ai_assistant.mcp_server.services.appeal_extraction_service import (
    AppealExtractionService,
)
from edms_ai_assistant.shared.utils.utils import CustomJSONEncoder, UUID_RE, extract_text_from_bytes

logger = logging.getLogger(__name__)

_AUTOFILL_SUPPORTED: frozenset[str] = frozenset({"APPEAL"})
_VALID_CATEGORIES: frozenset[str] = frozenset(
    {
        "APPEAL", "INCOMING", "OUTGOING", "INTERN", "CONTRACT",
        "MEETING", "MEETING_QUESTION", "QUESTION", "CUSTOM",
    }
)
_CATEGORY_NAMES_RU: dict[str, str] = {
    "APPEAL": "обращение",
    "INCOMING": "входящий документ",
    "OUTGOING": "исходящий документ",
    "INTERN": "внутренний документ",
    "CONTRACT": "договор",
    "MEETING": "совещание",
    "MEETING_QUESTION": "вопрос повестки",
    "QUESTION": "вопрос",
    "CUSTOM": "произвольный документ",
}


# ── Вспомогательные классы ────────────────────────────────────────────────────


class ValueSanitizer:
    """Утилиты для очистки и валидации данных."""

    EMPTY_PLACEHOLDERS = {
        "none", "null", "nil", "n/a", "na", "unknown",
        "not specified", "no", "нет", "неизвестно", "н/д",
    }

    @classmethod
    def is_empty(cls, value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            trimmed = value.strip()
            if not trimmed or trimmed.lower() in cls.EMPTY_PLACEHOLDERS:
                return True
        return False

    @classmethod
    def sanitize_string(cls, value: str | None) -> str | None:
        if cls.is_empty(value):
            return None
        cleaned = (
            str(value)
            .replace("\u201c", '"').replace("\u201d", '"')
            .replace("\u201e", '"').replace("\u00ab", '"').replace("\u00bb", '"')
            .strip()
        )
        return cleaned if cleaned else None

    @classmethod
    def fix_datetime_format(cls, dt: Any) -> str | None:
        if dt is None:
            return None
        if isinstance(dt, str):
            dt = dt.replace(" ", "T")
            if not dt.endswith("Z") and "+00:00" in dt:
                dt = dt.replace("+00:00", "Z")
            return dt
        if isinstance(dt, datetime):
            return dt.isoformat() if dt.tzinfo else dt.isoformat() + "Z"
        return None


def _is_good_correspondent_match(query: str, canonical: str) -> bool:
    """Проверяет качество FTS-совпадения корреспондента."""
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
    orig_words = [w for w in query.strip().split() if len(w) > 3]
    if len(orig_words) >= 2:
        abbrev = "".join(w[0].upper() for w in orig_words)
        if re.search(r"\b" + re.escape(abbrev) + r"\b", canonical, re.IGNORECASE):
            return True
    return False


@dataclass(frozen=True)
class AutofillResult:
    """Иммутабельный результат автозаполнения."""

    status: str
    message: str
    warnings: list[str] | None = None
    attachment_used: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"status": self.status, "message": self.message}
        if self.warnings:
            result["warnings"] = self.warnings
        if self.attachment_used:
            result["attachment_used"] = self.attachment_used
        return result


# ── Оркестратор автозаполнения ────────────────────────────────────────────────


class AppealAutofillOrchestrator:
    """Главный оркестратор процесса автозаполнения обращения."""

    MIN_TEXT_LENGTH = 50
    SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".doc", ".rtf")

    def __init__(
        self, document_id: str, token: str, attachment_id: str | None
    ) -> None:
        self.document_id = document_id
        self.token = token
        self.attachment_id = attachment_id
        self.warnings: list[str] = []
        self._last_extracted_text: str | None = None
        self._last_short_summary: str | None = None

    async def execute(self) -> AutofillResult:
        """Выполняет полный цикл автозаполнения."""
        raw_doc = await self._load_document()

        # Проверка категории
        category = str(raw_doc.get("docCategoryConstant", "") or "")
        if hasattr(category, "value"):
            category = category.value  # type: ignore[union-attr]
        if category != "APPEAL":
            raise ValueError(
                f"Документ должен быть категории APPEAL, а не {category}"
            )

        # Выбор вложения
        attachments = raw_doc.get("attachmentDocument") or []
        if not attachments:
            raise ValueError("В документе отсутствуют вложения")

        target = self._select_attachment(attachments)
        att_id = str(target.get("id") if isinstance(target, dict) else getattr(target, "id", ""))
        att_name = (target.get("name") if isinstance(target, dict) else getattr(target, "name", "")) or "attachment"

        # Извлечение текста
        async with EdmsAttachmentClient() as client:
            file_bytes = await client.get_attachment_content(
                self.token, self.document_id, att_id
            )

        extracted_text = extract_text_from_bytes(file_bytes, att_name)
        if not extracted_text or len(extracted_text) < self.MIN_TEXT_LENGTH:
            raise ValueError("Текст не извлечён или слишком короткий")

        self._last_extracted_text = extracted_text

        # LLM-анализ
        extraction_service = AppealExtractionService()
        fields = await extraction_service.extract_appeal_fields(extracted_text)
        self._last_short_summary = fields.shortSummary

        # Обновление документа
        await self._update_document(raw_doc, fields, extracted_text)

        return AutofillResult(
            status="success",
            message="Документ успешно заполнен",
            warnings=self.warnings if self.warnings else None,
            attachment_used=att_name,
        )

    def _select_attachment(self, attachments: list) -> Any:
        """Выбирает подходящее вложение."""
        if self.attachment_id:
            target = next(
                (
                    a for a in attachments
                    if str(a.get("id") if isinstance(a, dict) else getattr(a, "id", "")) == self.attachment_id
                ),
                None,
            )
            if not target:
                self.warnings.append(
                    f"Вложение ID={self.attachment_id} не найдено, используется автоподбор"
                )

        else:
            target = None

        if not target:
            target = next(
                (
                    a for a in attachments
                    if (a.get("name") if isinstance(a, dict) else getattr(a, "name", "")).lower().endswith(
                        self.SUPPORTED_EXTENSIONS
                    )
                ),
                attachments[0],
            )
        return target

    async def _load_document(self) -> dict[str, Any]:
        """Загружает метаданные документа."""
        async with DocumentClient() as client:
            raw = await client.get_document_metadata(self.token, self.document_id)
        if not raw:
            raise ValueError(f"Документ {self.document_id} не найден")
        return raw

    async def _update_document(
        self,
        raw_doc: dict[str, Any],
        fields: Any,
        extracted_text: str,
    ) -> None:
        """Выполняет обновление основных полей и полей обращения."""
        async with DocumentClient() as doc_client:
            async with ReferenceClient() as ref_client:
                await self._execute_main_fields_update(
                    doc_client, ref_client, raw_doc, fields
                )
                await self._execute_appeal_fields_update(
                    doc_client, ref_client, raw_doc, fields, extracted_text
                )

    async def _execute_main_fields_update(
        self,
        doc_client: DocumentClient,
        ref_client: ReferenceClient,
        raw_doc: dict[str, Any],
        fields: Any,
    ) -> None:
        delivery_id = raw_doc.get("deliveryMethodId")
        if not delivery_id:
            delivery_method_name = (
                fields.deliveryMethod
                if not ValueSanitizer.is_empty(fields.deliveryMethod)
                else "Курьер"
            )
            delivery_id = await ref_client.find_delivery_method(
                self.token, delivery_method_name
            )

        raw_summary = (
            ValueSanitizer.sanitize_string(fields.shortSummary)
            if not ValueSanitizer.is_empty(fields.shortSummary)
            else raw_doc.get("shortSummary")
        )
        if raw_summary and len(raw_summary) > 80:
            raw_summary = raw_summary[:80]
            logger.warning("shortSummary обрезан до 80 символов: '%s'", raw_summary)

        main_payload: dict[str, Any] = {}
        if raw_summary:
            main_payload["shortSummary"] = raw_summary
        if delivery_id:
            main_payload["deliveryMethodId"] = delivery_id
        if raw_doc.get("documentTypeId"):
            main_payload["documentTypeId"] = str(raw_doc["documentTypeId"])
        for field_name in ("pages", "additionalPages", "exemplarCount", "note"):
            val = raw_doc.get(field_name)
            if val is not None:
                main_payload[field_name] = val

        if main_payload:
            await self._execute_operation(
                doc_client, "DOCUMENT_MAIN_FIELDS_UPDATE", main_payload
            )

    async def _execute_appeal_fields_update(
        self,
        doc_client: DocumentClient,
        ref_client: ReferenceClient,
        raw_doc: dict[str, Any],
        fields: Any,
        extracted_text: str,
    ) -> None:
        d = raw_doc.get("documentAppeal") or {}

        # ── Geography ─────────────────────────────────────────────────────────
        geo_data: dict[str, Any] = {}

        # Страна
        country_name = (
            d.get("countryAppealName") if not ValueSanitizer.is_empty(d.get("countryAppealName")) else fields.country
        )
        if country_name:
            data = await ref_client.find_country_with_name(self.token, country_name)
            if data:
                geo_data["countryAppealId"] = data["id"]
                geo_data["countryAppealName"] = data["name"]

        # Регион — только если явно указан
        region_name = (
            d.get("regionName") if not ValueSanitizer.is_empty(d.get("regionName")) else fields.regionName
        )
        if region_name:
            data = await ref_client.find_region_with_name(self.token, region_name)
            if data:
                geo_data["regionId"] = data["id"]
                geo_data["regionName"] = data["name"]

        # Район — только если явно указан
        district_name = (
            d.get("districtName") if not ValueSanitizer.is_empty(d.get("districtName")) else fields.districtName
        )
        if district_name:
            data = await ref_client.find_district_with_name(self.token, district_name)
            if data:
                geo_data["districtId"] = data["id"]
                geo_data["districtName"] = data["name"]

        # Город — с иерархией
        city_name = (
            d.get("cityName") if not ValueSanitizer.is_empty(d.get("cityName")) else fields.cityName
        )
        if city_name:
            data = await ref_client.find_city_with_hierarchy(self.token, city_name)
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

        # ── Основной payload ─────────────────────────────────────────────────
        appeal_payload: dict[str, Any] = {**geo_data}

        # Корреспондент (пересылающий орган)
        corr_id = d.get("correspondentAppealId")
        if corr_id:
            appeal_payload["correspondentAppealId"] = str(corr_id)
            appeal_payload["correspondentAppeal"] = ValueSanitizer.sanitize_string(
                d.get("correspondentAppeal")
            )
        else:
            corr_name = d.get("correspondentAppeal") or fields.correspondentAppeal
            if corr_name and not ValueSanitizer.is_empty(corr_name):
                canonical = await ref_client._find_entity_with_name(
                    self.token, "correspondent", corr_name, "Корреспондент"
                )
                if canonical and _is_good_correspondent_match(corr_name, canonical["name"]):
                    appeal_payload["correspondentAppealId"] = canonical["id"]
                    appeal_payload["correspondentAppeal"] = ValueSanitizer.sanitize_string(corr_name)
                else:
                    appeal_payload["correspondentAppeal"] = ValueSanitizer.sanitize_string(corr_name)
                    appeal_payload["correspondentAppealId"] = None
            else:
                appeal_payload["correspondentAppeal"] = None
                appeal_payload["correspondentAppealId"] = None

        # Тип заявителя
        declarant_type = None
        if fields.declarantType:
            raw_dt = str(fields.declarantType).upper()
            if raw_dt in ("INDIVIDUAL", "ENTITY"):
                declarant_type = raw_dt
        if not declarant_type:
            raw_dt = str(d.get("declarantType", "") or "").upper()
            declarant_type = raw_dt if raw_dt in ("INDIVIDUAL", "ENTITY") else "INDIVIDUAL"
            if declarant_type == "INDIVIDUAL":
                self.warnings.append("declarantType установлен INDIVIDUAL по умолчанию")

        appeal_payload["declarantType"] = declarant_type

        # Вид обращения
        if d.get("citizenTypeId"):
            appeal_payload["citizenTypeId"] = str(d["citizenTypeId"])
        elif not ValueSanitizer.is_empty(fields.citizenType):
            cid = await ref_client.find_citizen_type(self.token, fields.citizenType)
            if cid:
                appeal_payload["citizenTypeId"] = cid

        # Тема
        if d.get("subjectId"):
            appeal_payload["subjectId"] = str(d["subjectId"])
        else:
            subject_id = await ref_client.find_best_subject(self.token, extracted_text)
            if subject_id:
                appeal_payload["subjectId"] = subject_id

        # Данные заявителя (ФИО)
        fio = d.get("fioApplicant") or fields.fioApplicant
        if not ValueSanitizer.is_empty(fio):
            appeal_payload["fioApplicant"] = ValueSanitizer.sanitize_string(fio)

        # Дата документа корреспондента
        date_doc = d.get("dateDocCorrespondentOrg") or fields.dateDocCorrespondentOrg
        if date_doc:
            appeal_payload["dateDocCorrespondentOrg"] = ValueSanitizer.fix_datetime_format(date_doc)

        # Поля для юридического лица
        if declarant_type == "ENTITY":
            raw_org = d.get("organizationName") or fields.organizationName
            if not ValueSanitizer.is_empty(raw_org):
                corr_data = await ref_client._find_entity_with_name(
                    self.token, "correspondent", raw_org, "Организация"
                )
                if corr_data and _is_good_correspondent_match(raw_org, corr_data.get("name", "")):
                    appeal_payload["organizationName"] = corr_data["name"]
                else:
                    appeal_payload["organizationName"] = ValueSanitizer.sanitize_string(raw_org)
            else:
                appeal_payload["organizationName"] = None
            appeal_payload["signed"] = ValueSanitizer.sanitize_string(
                d.get("signed") or fields.signed
            )
            appeal_payload["correspondentOrgNumber"] = ValueSanitizer.sanitize_string(
                d.get("correspondentOrgNumber") or fields.correspondentOrgNumber
            )
        else:
            appeal_payload["organizationName"] = None
            appeal_payload["signed"] = None
            appeal_payload["correspondentOrgNumber"] = None

        # Булевы поля
        for bool_field in ("collective", "anonymous", "reasonably"):
            llm_val = getattr(fields, bool_field, None)
            db_val = d.get(bool_field)
            if llm_val is True:
                appeal_payload[bool_field] = True
            elif db_val is not None:
                appeal_payload[bool_field] = db_val
            else:
                appeal_payload[bool_field] = llm_val

        # Дата поступления
        raw_receipt = d.get("receiptDate") or fields.receiptDate
        if raw_receipt:
            receipt_dt = raw_receipt if isinstance(raw_receipt, datetime) else None
            if receipt_dt and abs((receipt_dt.date() - datetime.now(UTC).date()).days) <= 1:
                appeal_payload["receiptDate"] = None
            else:
                appeal_payload["receiptDate"] = ValueSanitizer.fix_datetime_format(raw_receipt)
        else:
            appeal_payload["receiptDate"] = None

        # Текстовые поля
        for text_field in (
            "fullAddress", "phone", "email", "index",
            "indexDateCoverLetter", "reviewProgress",
        ):
            db_val = d.get(text_field)
            llm_val = getattr(fields, text_field, None)
            val = db_val if not ValueSanitizer.is_empty(db_val) else llm_val
            appeal_payload[text_field] = ValueSanitizer.sanitize_string(val)

        # submissionForm
        submission_form = d.get("submissionForm")
        appeal_payload["submissionForm"] = (
            submission_form if not ValueSanitizer.is_empty(submission_form) else "WRITTEN"
        )

        # DB-only поля
        for db_field in ("solutionResultId", "nomenclatureAffairId"):
            val = d.get(db_field)
            if val:
                appeal_payload[db_field] = str(val)

        # Фильтрация пустых (кроме обязательных)
        _ALWAYS_INCLUDE = {"correspondentAppeal", "correspondentAppealId", "submissionForm"}
        filtered_payload = {
            k: v for k, v in appeal_payload.items()
            if k in _ALWAYS_INCLUDE or (v is not None and not ValueSanitizer.is_empty(v))
        }

        if not filtered_payload.get("declarantType"):
            raise ValueError("declarantType обязателен, но не установлен")
        if not filtered_payload.get("submissionForm"):
            filtered_payload["submissionForm"] = "WRITTEN"

        await self._execute_operation(
            doc_client, "DOCUMENT_MAIN_FIELDS_APPEAL_UPDATE", filtered_payload
        )

    async def _execute_operation(
        self,
        client: DocumentClient,
        operation_type: str,
        body: dict[str, Any],
    ) -> None:
        """Выполняет операцию через execute API."""
        payload = [{"operationType": operation_type, "body": body}]
        json_safe = json.loads(json.dumps(payload, cls=CustomJSONEncoder))
        await client._make_request(
            "POST",
            f"api/document/{self.document_id}/execute",
            token=self.token,
            json=json_safe,
        )
        logger.info("%s executed successfully for %s", operation_type, self.document_id[:8])


async def _generate_summary_variants(
    text: str, current_summary: str | None
) -> list[str]:
    """Генерирует 3 варианта краткого содержания через LLM."""
    from edms_ai_assistant.llm_client import get_llm_response

    prompt = (
        "Сформулируй РОВНО 3 варианта краткого содержания (заголовка) обращения. "
        "Каждый вариант — отдельная строка, максимум 80 символов. "
        "Стили: 1) краткий (суть в 5-8 словах), "
        "2) официальный (с указанием типа обращения), "
        "3) описательный (с ключевыми деталями). "
        "Отвечай ТОЛЬКО тремя строками без нумерации.\n\n"
        f"Текст обращения:\n{text[:2000]}"
    )
    system = (
        "Ты — эксперт по делопроизводству. "
        "Отвечай строго на русском языке без вводных фраз."
    )
    try:
        result = await get_llm_response(prompt, system=system)
        variants = [v.strip() for v in result.strip().split("\n") if v.strip()]
        variants = [v[:77] + "..." if len(v) > 80 else v for v in variants[:3]]
        if current_summary and current_summary not in variants:
            variants = [
                (current_summary[:77] + "..." if len(current_summary) > 80 else current_summary)
            ] + variants[:2]
        return variants[:3]
    except Exception as exc:
        logger.warning("Failed to generate summary variants: %s", exc)
        return [current_summary] if current_summary else []


# ── FastMCP tool регистрация ─────────────────────────────────────────────────


def register_appeal_tools(mcp: FastMCP) -> None:
    """Регистрирует инструменты работы с обращениями."""

    @mcp.tool(
        description=(
            "Автоматически заполнить карточку обращения гражданина (APPEAL) "
            "через LLM-анализ вложенного файла. Извлекает ФИО, адрес, тему, "
            "тип заявителя, географию и другие поля. "
            "generate_summary_choices=True — вернуть 3 варианта заголовка для выбора."
        )
    )
    async def autofill_appeal_document(
        document_id: str,
        token: str,
        attachment_id: str | None = None,
        generate_summary_choices: bool = False,
    ) -> dict[str, Any]:
        """
        Автозаполнение карточки обращения из вложения.

        Args:
            document_id: UUID документа категории APPEAL.
            token: JWT-токен.
            attachment_id: UUID вложения (опционально).
            generate_summary_choices: Вернуть 3 варианта заголовка для выбора.
        """
        logger.info(
            "autofill_appeal_document: doc=%s att=%s choices=%s",
            document_id[:8],
            attachment_id,
            generate_summary_choices,
        )

        try:
            orchestrator = AppealAutofillOrchestrator(
                document_id, token, attachment_id
            )
            result = await orchestrator.execute()
            result_dict = result.to_dict()
            result_dict["requires_reload"] = True

            if generate_summary_choices and orchestrator._last_extracted_text:
                variants = await _generate_summary_variants(
                    orchestrator._last_extracted_text,
                    orchestrator._last_short_summary,
                )
                if variants:
                    result_dict["summary_choices"] = variants
                    result_dict["summary_choices_hint"] = (
                        "Выберите заголовок или предложите свой вариант. "
                        "Скажите 'Установи заголовок: <текст>' чтобы применить."
                    )

            return result_dict

        except ValueError as e:
            logger.error("Validation error in autofill: %s", e)
            return {"status": "error", "message": str(e)}
        except Exception as e:
            logger.error("autofill_appeal_document failed: %s", e, exc_info=True)
            return {"status": "error", "message": f"Ошибка автозаполнения: {e!s}"}

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
        cleaned = file_path.strip()
        if not cleaned or cleaned.lower() in {"", "none", "null", "<local_file_path>"}:
            return {
                "status": "error",
                "message": "file_path не может быть пустым. Используй значение из контекста агента.",
            }

        if UUID_RE.match(cleaned):
            return {
                "status": "error",
                "message": "Создание из UUID вложения EDMS не поддерживается. Загрузите файл локально.",
            }

        if not Path(cleaned).exists():
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
        effective_file_name = file_name or Path(cleaned).name
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
                if "document" in created
                else created
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
                orchestrator = AppealAutofillOrchestrator(document_id, token, None)
                await orchestrator.execute()
                autofill_status = "done"
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