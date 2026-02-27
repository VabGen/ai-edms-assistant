# src/ai_edms_assistant/infrastructure/nlp/processors/document_nlp_service.py
"""Document NLP processing service.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class DocumentNLPService:
    """Null-safe semantic processor for EDMS document data.

    Accepts both raw API dict and domain Document entity.
    All field access via ``_safe()`` — never raises on missing fields.
    """

    # ── Public interface ──────────────────────────────────────────────────────

    def process_document(self, doc: Any) -> dict[str, Any]:
        """Build full semantic context from document data.

        Args:
            doc: Raw API dict or domain ``Document`` entity.

        Returns:
            Structured analytics dict with Russian-language keys.
            Returns ``{}`` on None input, ``{"error": "..."}`` on failure.
        """
        if not doc:
            logger.warning("process_document called with None")
            return {}

        try:
            category = self._safe(doc, "docCategoryConstant") or self._safe(
                doc, "document_category"
            )
            if hasattr(category, "value"):
                category = category.value

            # ── 1. Базовая идентификация ──────────────────────────────────────
            base_info = self._build_base_info(doc, category)

            # ── 2. Участники ──────────────────────────────────────────────────
            participants = self._build_participants(doc)

            # ── 3. Статистика поручений ───────────────────────────────────────
            tasks_info = self._build_tasks_info(doc)

            # ── 4. Регистрационные данные ─────────────────────────────────────
            registration = self._build_registration(doc)

            # ── 5. Категориезависимые разделы ─────────────────────────────────
            specific: dict[str, Any] = {}
            cat_str = str(category or "").upper()

            if cat_str == "APPEAL":
                appeal_data = self._process_appeal(doc)
                if appeal_data:
                    specific["обращение"] = appeal_data

            elif cat_str in ("MEETING", "MEETING_QUESTION"):
                meeting_data = self._process_meeting(doc)
                if meeting_data:
                    specific["заседание"] = meeting_data

            elif cat_str == "CONTRACT":
                contract_data = self._process_contract(doc)
                if contract_data:
                    specific["договор"] = contract_data

            # ── 6. Вложения ───────────────────────────────────────────────────
            attachments_info = self._build_attachments(doc)

            # ── 7. Поручения (если загружены) ─────────────────────────────────
            tasks_list = self._build_tasks_list(doc)

            return {
                **base_info,
                "участники": participants,
                "поручения_статистика": tasks_info,
                "регистрация": registration,
                **specific,
                "вложения": attachments_info,
                "поручения": tasks_list,
            }

        except Exception as exc:
            logger.error(
                "process_document_failed",
                extra={"error": str(exc)},
                exc_info=True,
            )
            return {"error": str(exc)}

    # ── Section builders ──────────────────────────────────────────────────────

    def _build_base_info(self, doc: Any, category: str | None) -> dict[str, Any]:
        """Build base identification section.
        """
        doc_type_name = (
            self._safe(doc, "document_type_name")
            or self._safe(doc, "documentTypeName")
            or self._extract_nested(doc, "documentType", "typeName")
        )

        delivery_name = (
            self._safe(doc, "delivery_method_name")
            or self._extract_nested(doc, "deliveryMethod", "deliveryName")
        )

        return {
            "id": str(self._safe(doc, "id")) if self._safe(doc, "id") else None,
            "категория": category,
            "категория_название": self._category_label(category),
            "профиль": self._safe(doc, "profileName") or self._safe(doc, "profile_name"),
            "тип_документа": doc_type_name,
            "краткое_содержание": (
                self._safe(doc, "shortSummary") or self._safe(doc, "short_summary")
            ),
            "полный_текст": self._safe(doc, "summary"),
            "примечание": self._safe(doc, "note"),
            "дней_исполнения": (
                self._safe(doc, "daysExecution") or self._safe(doc, "days_execution")
            ),
            "способ_доставки": delivery_name,
            "текущий_этап": (
                self._safe(doc, "current_bpmn_task_name")
                or self._safe(doc, "currentBpmnTaskName")
            ),
            "гриф_дсп": (
                self._safe(doc, "dspFlag") or self._safe(doc, "dsp_flag") or False
            ),
            # FIX: гриф доступа
            "гриф_доступа": (
                self._safe(doc, "enable_access_grief")
                or self._safe(doc, "enableAccessGrief")
                or False
            ),
            "на_контроле": (
                self._safe(doc, "controlFlag") or self._safe(doc, "control_flag") or False
            ),
            "страниц": self._safe(doc, "pages") or self._safe(doc, "pages_count") or 0,
        }

    def _build_participants(self, doc: Any) -> dict[str, Any]:
        """Build participants section including who_signed (FIX)."""
        return {
            "автор": self._format_user(self._safe(doc, "author")),
            "ответственный_исполнитель": self._format_user(
                self._safe(doc, "responsibleExecutor")
                or self._safe(doc, "responsible_executor")
            ),
            "инициатор": self._format_user(
                self._safe(doc, "initiator")
            ),
            "кем_подписан": self._format_user(
                self._safe(doc, "whoSigned") or self._safe(doc, "who_signed")
            ),
            "корреспондент": (
                self._safe(doc, "correspondentName")
                or self._safe(doc, "correspondent_name")
            ),
        }

    def _build_tasks_info(self, doc: Any) -> dict[str, Any]:
        """Build tasks statistics section with all counters (FIX)."""
        return {
            "всего": self._safe(doc, "countTask") or self._safe(doc, "count_task") or 0,
            # FIX: ранее отсутствовали
            "в_работе": (
                self._safe(doc, "taskProjectCount")
                or self._safe(doc, "task_project_count")
                or 0
            ),
            "завершено": (
                self._safe(doc, "completedTaskCount")
                or self._safe(doc, "completed_task_count")
                or 0
            ),
            "вложений": (
                len(self._safe(doc, "attachmentDocument") or self._safe(doc, "attachments") or [])
            ),
            "ознакомлений": (
                self._safe(doc, "introductionCount") or self._safe(doc, "introduction_count") or 0
            ),
            "ознакомлений_завершено": (
                self._safe(doc, "introductionCompleteCount")
                or self._safe(doc, "introduction_complete_count")
                or 0
            ),
        }

    def _build_registration(self, doc: Any) -> dict[str, Any]:
        """Build registration data section."""
        status_raw = self._safe(doc, "status")
        status_val = status_raw.value if hasattr(status_raw, "value") else status_raw
        return {
            "статус": status_val,
            "статус_название": self._status_label(status_val),
            "рег_номер": (
                self._safe(doc, "regNumber") or self._safe(doc, "reg_number")
            ),
            "дата_регистрации": self._format_date(
                self._safe(doc, "regDate") or self._safe(doc, "reg_date")
            ),
            "дата_создания": self._format_date(
                self._safe(doc, "createDate") or self._safe(doc, "create_date")
            ),
            "исх_номер": (
                self._safe(doc, "outRegNumber") or self._safe(doc, "out_reg_number")
            ),
        }

    def _build_attachments(self, doc: Any) -> list[dict]:
        """Build attachments summary list."""
        atts = (
            self._safe(doc, "attachmentDocument")
            or self._safe(doc, "attachments")
            or []
        )
        result = []
        for a in (atts or []):
            if not a:
                continue
            att = self._safe(a, "attachment") or a
            name = (
                self._safe(att, "fileName")
                or self._safe(att, "file_name")
                or self._safe(att, "name")
                or "Без имени"
            )
            size = self._safe(att, "fileSize") or self._safe(att, "file_size")
            result.append({"имя": name, "размер": size})
        return result

    def _build_tasks_list(self, doc: Any) -> list[dict[str, Any]]:
        """Build tasks list section if tasks are loaded."""
        tasks = self._safe(doc, "taskList") or self._safe(doc, "tasks") or []
        result = []
        for t in (tasks or []):
            if t:
                result.append(self._format_task(t))
        return result

    # ── Category-specific processors ─────────────────────────────────────────

    def _process_appeal(self, doc: Any) -> dict[str, Any] | None:
        """Extract appeal-specific fields.

        Args:
            doc: Raw document data (dict or entity).

        Returns:
            Appeal analytics dict or None if no appeal data.
        """
        app = self._safe(doc, "documentAppeal") or self._safe(doc, "appeal")
        if not app:
            return None

        # ── Тип заявителя ────────────────────────────────────────
        declarant_raw = (
            self._safe(app, "declarantType") or self._safe(app, "declarant_type")
        )
        declarant_label: str | None = None
        if declarant_raw:
            val = declarant_raw.value if hasattr(declarant_raw, "value") else str(declarant_raw)
            declarant_label = (
                "Юридическое лицо" if val == "ENTITY" else "Физическое лицо"
            )

        # ── Вид обращения ──
        citizen_type_name = (
            self._safe(app, "citizen_type_name")
            or self._safe(app, "citizenTypeName")
            or self._extract_nested(app, "citizenType", "name")
        )

        # ── Тема обращения ──────
        subject_name = (
            self._safe(app, "subject_name")
            or self._extract_nested(app, "subject", "name")
        )
        subject_parent = (
            self._safe(app, "subject_parent_name")
            or self._extract_nested_deep(app, "subject", "parentSubject", "name")
        )
        subject_display = None
        if subject_parent and subject_name:
            subject_display = f"{subject_parent} → {subject_name}"
        elif subject_name:
            subject_display = subject_name

        # ── Страна ────────────────────────────────────────────────
        country_name = (
            self._safe(app, "country_appeal_name")
            or self._safe(app, "countryAppealName")
        )

        # ── Гео-данные ────
        geo = self._safe(app, "geo_location") or self._safe(app, "geoLocation")
        if geo:
            region = self._safe(geo, "region_name") or self._safe(geo, "regionName")
            district = self._safe(geo, "district_name") or self._safe(geo, "districtName")
            city = self._safe(geo, "city_name") or self._safe(geo, "cityName")
        else:
            region = self._safe(app, "regionName")
            district = self._safe(app, "districtName")
            city = self._safe(app, "cityName")

        return {
            "заявитель": (
                self._safe(app, "fioApplicant") or self._safe(app, "applicant_name")
            ),
            "организация": (
                self._safe(app, "organizationName")
                or self._safe(app, "organization_name")
            ),
            "тип_заявителя": declarant_label,
            "коллективное": self._safe(app, "collective"),
            "анонимное": self._safe(app, "anonymous"),
            "подписано": self._safe(app, "signed"),
            "вид_обращения": citizen_type_name,
            "тема": subject_display,
            "телефон": self._safe(app, "phone"),
            "email": self._safe(app, "email"),
            "страна": country_name,
            "регион": region,
            "район": district,
            "город": city,
            "почтовый_индекс": self._safe(app, "index"),
            "адрес": (
                self._safe(app, "fullAddress") or self._safe(app, "full_address")
            ),
            "дата_поступления": self._format_date(
                self._safe(app, "receiptDate") or self._safe(app, "receipt_date")
            ),
            "исх_номер_корреспондента": (
                self._safe(app, "correspondentOrgNumber")
                or self._safe(app, "correspondent_org_number")
            ),
            "ход_рассмотрения": (
                self._safe(app, "reviewProgress")
                or self._safe(app, "review_progress")
            ),
            "обоснованное": self._safe(app, "reasonably"),
        }

    def _process_meeting(self, doc: Any) -> dict[str, Any] | None:
        """Extract meeting-specific fields.

        Args:
            doc: Raw document data.

        Returns:
            Meeting analytics dict or None if no meeting data.
        """
        date_meeting = (
            self._safe(doc, "dateMeeting") or self._safe(doc, "date_meeting")
        )
        if not date_meeting:
            return None

        return {
            "дата": self._format_date(date_meeting),
            "время_начала": self._format_datetime(
                self._safe(doc, "startMeeting") or self._safe(doc, "start_meeting")
            ),
            "время_окончания": self._format_datetime(
                self._safe(doc, "endMeeting") or self._safe(doc, "end_meeting")
            ),
            "место": self._safe(doc, "placeMeeting") or self._safe(doc, "place_meeting"),
            "председатель": self._format_user(self._safe(doc, "chairperson")),
            "секретарь": self._format_user(self._safe(doc, "secretary")),
            "количество_приглашенных": (
                self._safe(doc, "inviteesCount") or self._safe(doc, "invitees_count")
            ),
            "оповещений_по_вопросу": (
                self._safe(doc, "meetingQuestionNotifyCount")
                or self._safe(doc, "meeting_question_notify_count")
                or 0
            ),
        }

    def _process_contract(self, doc: Any) -> dict[str, Any] | None:
        """Extract contract-specific fields from Java ``DocumentDto``.

        Changelog (2026-02 gap-fix):
          + дата_начала_действия (contractStartDate)
          + валюта (currencyId — пока только ID до справочника)

        Args:
            doc: Raw document data.

        Returns:
            Contract analytics dict or None if no contract data present.
        """
        contract_sum = (
            self._safe(doc, "contractSum") or self._safe(doc, "contract_sum")
        )
        contract_number = (
            self._safe(doc, "contractNumber") or self._safe(doc, "contract_number")
        )
        if not contract_sum and not contract_number:
            return None

        return {
            "номер_договора": contract_number,
            "дата_договора": self._format_date(
                self._safe(doc, "contractDate") or self._safe(doc, "contract_date")
            ),
            "сумма": str(contract_sum) if contract_sum else None,
            "дата_подписания": self._format_date(
                self._safe(doc, "contractSigningDate")
                or self._safe(doc, "contract_signing_date")
            ),
            "дата_начала_действия": self._format_date(
                self._safe(doc, "contractStartDate")
                or self._safe(doc, "contract_start_date")
            ),
            "дата_начала_срока": self._format_date(
                self._safe(doc, "contractDurationStart")
                or self._safe(doc, "contract_duration_start")
            ),
            "дата_окончания_срока": self._format_date(
                self._safe(doc, "contractDurationEnd")
                or self._safe(doc, "contract_duration_end")
            ),
            "согласован": (
                self._safe(doc, "contractAgreement")
                or self._safe(doc, "contract_agreement")
            ),
            "автопролонгация": (
                self._safe(doc, "contractAutoProlongation")
                or self._safe(doc, "contract_auto_prolongation")
            ),
            "типовой": (
                self._safe(doc, "contractTypical") or self._safe(doc, "contract_typical")
            ),
        }

    # ── Task formatting ───────────────────────────────────────────────────────

    def _format_task(self, t: Any) -> dict[str, Any]:
        """Format a single task record for LLM context.

        Mapping:
            - ``author`` → ``"постановщик"`` (who assigned, not executor)
            - ``taskExecutors[responsible=True]`` → ``"ответственный_исполнитель"``
            - all ``taskExecutors`` → ``"все_исполнители"``

        Args:
            t: Raw task dict or domain ``Task`` entity.

        Returns:
            Formatted task dict.
        """
        status_raw = self._safe(t, "taskStatus") or self._safe(t, "status")
        status_val = status_raw.value if hasattr(status_raw, "value") else status_raw

        # Маппинг исполнителей из taskExecutors (Java: List<TaskExecutorsDto>)
        responsible_name = None
        all_executors: list[str] = []

        task_executors = (
            self._safe(t, "taskExecutors") or self._safe(t, "executors") or []
        )
        for ex in (task_executors or []):
            emp_raw = self._safe(ex, "executor") or ex
            name = self._format_user(emp_raw)
            if name:
                all_executors.append(name)
                is_responsible = (
                    self._safe(ex, "responsible")
                    or self._safe(ex, "isResponsible")
                    or False
                )
                if is_responsible and not responsible_name:
                    responsible_name = name

        return {
            "номер": (
                self._safe(t, "taskNumber") or self._safe(t, "task_number")
            ),
            "текст": self._safe(t, "text"),
            "статус": status_val,
            "постановщик": self._format_user(self._safe(t, "author")),
            "ответственный_исполнитель": responsible_name,
            "все_исполнители": all_executors,
            "срок": self._format_date(
                self._safe(t, "planEndDate") or self._safe(t, "plan_end_date")
            ),
        }

    # ── Utility helpers ───────────────────────────────────────────────────────

    def _safe(self, obj: Any, key: str, default: Any = None) -> Any:
        """Null-safe field accessor for both dicts and objects.

        Args:
            obj: Any object — dict, domain entity, or None.
            key: Field name.
            default: Fallback value.

        Returns:
            Field value or ``default``.
        """
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _extract_nested(self, obj: Any, nested_key: str, field: str) -> Any:
        """Extract a field from a nested dict/object.

        Used for patterns like ``documentType.typeName`` where Java returns
        an object but Python entity has the flat field.

        Args:
            obj: Parent dict or object.
            nested_key: Key of the nested dict/object.
            field: Field inside the nested dict.

        Returns:
            Field value or None.
        """
        nested = self._safe(obj, nested_key)
        if not nested:
            return None
        return self._safe(nested, field)

    def _extract_nested_deep(
        self, obj: Any, key1: str, key2: str, field: str
    ) -> Any:
        """Extract a field two levels deep.

        Used for: ``subject.parentSubject.name``.

        Args:
            obj: Root object.
            key1: First level key.
            key2: Second level key.
            field: Target field name.

        Returns:
            Field value or None.
        """
        level1 = self._safe(obj, key1)
        if not level1:
            return None
        level2 = self._safe(level1, key2)
        if not level2:
            return None
        return self._safe(level2, field)

    def _format_user(self, user: Any) -> str | None:
        """Format UserInfoDto to full name string.

        Args:
            user: UserInfoDto dict, UserInfo entity, or None.

        Returns:
            'Фамилия Имя Отчество' or None if all parts are empty.
        """
        if not user:
            return None
        parts = [
            self._safe(user, "lastName") or self._safe(user, "last_name") or "",
            self._safe(user, "firstName") or self._safe(user, "first_name") or "",
            self._safe(user, "middleName") or self._safe(user, "middle_name") or "",
        ]
        full_name = " ".join(p for p in parts if p).strip()
        if not full_name:
            return None

        post = (
            self._safe(user, "authorPost")
            or self._safe(user, "post_name")
            or self._safe(user, "postName")
        )
        dept = (
            self._safe(user, "authorDepartmentName")
            or self._safe(user, "department_name")
            or self._safe(user, "departmentName")
        )
        if post or dept:
            details = ", ".join(p for p in [post, dept] if p)
            return f"{full_name} ({details})"
        return full_name

    def _format_date(self, value: Any) -> str | None:
        """Format datetime to DD.MM.YYYY string.

        Args:
            value: ``datetime`` object, ISO string, or None.

        Returns:
            Formatted date string or None.
        """
        if not value:
            return None
        try:
            from datetime import datetime

            if isinstance(value, str):
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return value.strftime("%d.%m.%Y")
        except (ValueError, AttributeError, TypeError):
            return str(value)

    def _format_datetime(self, value: Any) -> str | None:
        """Format datetime to DD.MM.YYYY HH:MM string.

        Args:
            value: ``datetime`` object, ISO string, or None.

        Returns:
            Formatted datetime string or None.
        """
        if not value:
            return None
        try:
            from datetime import datetime

            if isinstance(value, str):
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return value.strftime("%d.%m.%Y %H:%M")
        except (ValueError, AttributeError, TypeError):
            return str(value)

    def _status_label(self, status: str | None) -> str:
        """Returns Russian label for DocumentStatus value.

        Args:
            status: Status string like 'NEW', 'REGISTERED', etc.

        Returns:
            Russian label or original value if not found.
        """
        _labels: dict[str, str] = {
            "NEW": "Новый",
            "DRAFT": "Черновик",
            "STATEMENT": "На утверждении",
            "APPROVED": "Утверждён",
            "SIGNING": "На подписании",
            "SIGNED": "Подписан",
            "AGREEMENT": "На согласовании",
            "AGREED": "Согласован",
            "REVIEW": "На рассмотрении",
            "REVIEWED": "Рассмотрен",
            "REGISTRATION": "На регистрации",
            "REGISTERED": "Зарегистрирован",
            "EXECUTION": "На исполнении",
            "EXECUTED": "Исполнен",
            "DISPATCH": "На отправке",
            "SENT": "Отправлен",
            "REJECT": "Отклонён",
            "CANCEL": "Аннулирован",
            "PREPARATION": "Подготовка",
            "PAPERWORK": "На оформлении",
            "FORMALIZED": "Оформлен",
            "ACCEPTANCE": "На одобрении",
            "ACCEPTED": "Одобрен",
            "CONTRACT_EXECUTION": "Исполнение договора",
            "CONTRACT_CLOSED": "Закрыт",
            "ARCHIVE": "Архив",
            "DELETED": "Удалён",
        }
        if not status:
            return "Не указан"
        return _labels.get(str(status).upper(), str(status))

    def _category_label(self, category: str | None) -> str:
        """Returns Russian label for DocumentCategory value.

        Args:
            category: Category string like 'APPEAL', 'CONTRACT', etc.

        Returns:
            Russian label or original value if not found.
        """
        _labels: dict[str, str] = {
            "INTERN": "Внутренний",
            "INCOMING": "Входящий",
            "OUTGOING": "Исходящий",
            "MEETING": "Совещание",
            "QUESTION": "Вопрос повестки",
            "MEETING_QUESTION": "Повестка заседания",
            "APPEAL": "Обращение",
            "CONTRACT": "Договор",
            "CUSTOM": "Пользовательский",
        }
        if not category:
            return "Не указана"
        return _labels.get(str(category).upper(), str(category))