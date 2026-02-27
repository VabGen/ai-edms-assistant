# src/ai_edms_assistant/infrastructure/nlp/processors/document_nlp_service.py
"""Document NLP processing service.

Converts raw EDMS API dicts OR domain entity objects into a structured,
LLM-ready analytics context. All field access is null-safe.

Architecture:
    Infrastructure Layer → NLP processors.
    No external ML dependencies — pure Python.
    Used by: application/tools/document_tool.py.

Key design:
    All field access uses ``getattr(obj, key, default)`` or
    ``dict.get(key, default)`` — never direct indexing. This makes the
    service resilient to:
      - Java API returning null for any field.
      - Partial response DTOs (different endpoints return different field sets).
      - Domain entity objects vs raw dicts (both are supported).

Audit fixes (2025-06):
    CRITICAL — Task mapping:
        - ``author`` was incorrectly mapped as ``"исполнитель"``.
          In Java ``TaskDto``, ``author`` = задание поставил (кто создал поручение),
          NOT the executor. Fixed: mapped as ``"постановщик"``.
        - ``taskExecutors`` (``List<TaskExecutorsDto>``) was completely ignored.
          Fixed: ``_format_task_executors()`` resolves responsible executor +
          full executors list from ``TaskDto.taskExecutors``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class DocumentNLPService:
    """Null-safe semantic processor for EDMS document data.

    Accepts both raw API dict and domain Document entity. All field access
    is null-safe via ``_safe()`` helper — never raises on missing or null fields.
    """

    # ── Public interface ──────────────────────────────────────────────────────

    def process_document(self, doc: Any) -> dict[str, Any]:
        """Build full semantic context from document data.

        Args:
            doc: Raw API dict or domain Document entity.

        Returns:
            Structured analytics dict with Russian-language keys.
            Returns ``{}`` on None input, ``{"error": "..."}`` on failure.
        """
        if not doc:
            logger.warning("process_document called with None")
            return {}

        try:
            category = self._safe(doc, "docCategoryConstant")

            # ── 1. Базовая идентификация ──────────────────────────────────────
            base_info = {
                "id": str(self._safe(doc, "id")) if self._safe(doc, "id") else None,
                "категория": category,
                "профиль": self._safe(doc, "profileName"),
                "тип_документа": self._safe(doc, "documentTypeName"),
                "краткое_содержание": self._safe(doc, "shortSummary"),
                "полный_текст": self._safe(doc, "summary"),
                "примечание": self._safe(doc, "note"),
            }

            # ── 2. Регистрация и метаданные ───────────────────────────────────
            registration = {
                "рег_номер": (
                    self._safe(doc, "regNumber") or self._safe(doc, "reservedRegNumber")
                ),
                "дата_регистрации": self._format_date(self._safe(doc, "regDate")),
                "дата_создания": self._format_datetime(self._safe(doc, "createDate")),
                "зарезервированный_номер": self._safe(doc, "reservedRegNumber"),
                "исходящий_номер": self._safe(doc, "outRegNumber"),
                "исходящая_дата": self._format_date(self._safe(doc, "outRegDate")),
                "пропуск_регистрации": self._safe(doc, "skipRegistration"),
            }

            # ── 3. Участники ──────────────────────────────────────────────────
            who_addressed_raw = self._safe(doc, "whoAddressed") or []
            participants = {
                "автор": self._format_user(self._safe(doc, "author")),
                "инициатор": self._format_user(self._safe(doc, "initiator")),
                "ответственный_исполнитель": self._format_user(
                    self._safe(doc, "responsibleExecutor")
                ),
                "подписанты": [
                    self._format_user(u)
                    for u in who_addressed_raw
                    if self._format_user(u)
                ],
                "корреспондент": self._safe(doc, "correspondentName"),
                "количество_ответственных": self._safe(
                    doc, "responsibleExecutorsCount"
                ),
                "есть_ответственные": self._safe(doc, "hasResponsibleExecutor"),
            }

            # ── 4. Жизненный цикл ─────────────────────────────────────────────
            lifecycle = {
                "текущий_статус": (
                    self._safe(doc, "status") or self._safe(doc, "state")
                ),
                "предыдущий_статус": self._safe(doc, "prevStatus"),
            }

            # ── 5. Контроль и сроки ───────────────────────────────────────────
            control_info = {
                "на_контроле": self._safe(doc, "controlFlag"),
                "снят_с_контроля": self._safe(doc, "removeControl"),
                "дней_на_исполнение": self._safe(doc, "daysExecution"),
            }

            # ── 6. Специализированные данные по категории ─────────────────────
            specialized: dict[str, Any] = {}

            if category == "APPEAL":
                appeal_data = self._process_appeal(doc)
                if appeal_data:
                    specialized["обращение"] = appeal_data

            if category in ("MEETING", "MEETING_QUESTION", "QUESTION") or self._safe(
                doc, "dateMeeting"
            ):
                meeting_data = self._process_meeting(doc)
                if meeting_data:
                    specialized["совещание"] = meeting_data

            if category == "CONTRACT":
                contract_data = self._process_contract(doc)
                if contract_data:
                    specialized["договор"] = contract_data

            # ── 7. Поручения и задачи ─────────────────────────────────────────
            # FIX: Ранее task.author маппился как "исполнитель" — НЕВЕРНО.
            #   Java TaskDto.author = задание поставил (кто создал поручение).
            #   Реальные исполнители хранятся в TaskDto.taskExecutors.
            #   Исправлено: author → "постановщик", taskExecutors → "исполнители".
            tasks_raw = self._safe(doc, "taskList") or []
            tasks_info = {
                "общее_количество": (self._safe(doc, "countTask") or len(tasks_raw)),
                "список": [self._format_task(t) for t in tasks_raw],
            }

            # ── 8. Безопасность ───────────────────────────────────────────────
            security = {
                "гриф_ДСП": self._safe(doc, "dspFlag"),
            }

            # ── 9. Вложения ───────────────────────────────────────────────────
            attachments_raw = (
                self._safe(doc, "attachmentDocument")
                or self._safe(doc, "attachments")
                or []
            )
            relations = {
                "вложения": [
                    {
                        "название": (
                            self._safe(a, "fileName") or self._safe(a, "name")
                        ),
                        "id": (
                            str(self._safe(a, "id")) if self._safe(a, "id") else None
                        ),
                        "тип": (
                            self._safe(a, "type") or self._safe(a, "attachmentType")
                        ),
                        "размер_байт": (
                            self._safe(a, "fileSize") or self._safe(a, "size")
                        ),
                        "есть_ЭЦП": bool(
                            self._safe(a, "signs") and len(self._safe(a, "signs")) > 0
                        ),
                    }
                    for a in attachments_raw
                ],
                "количество_вложений": len(attachments_raw),
                "количество_связей": self._safe(doc, "documentLinksCount"),
            }

            # ── 10. Дополнительно ─────────────────────────────────────────────
            additional = {
                "страниц": (self._safe(doc, "pages") or self._safe(doc, "pages_count")),
                "листов_приложений": self._safe(doc, "additionalPages"),
                "ознакомление": {
                    "количество_визирующих": self._safe(doc, "introductionCount"),
                    "количество_завизировавших": self._safe(
                        doc, "introductionCompleteCount"
                    ),
                },
                "версионность": {
                    "включена": self._safe(doc, "versionFlag"),
                    "id_версии": (
                        str(self._safe(doc, "documentVersionId"))
                        if self._safe(doc, "documentVersionId")
                        else None
                    ),
                },
            }

            # ── Сборка результата ─────────────────────────────────────────────
            result: dict[str, Any] = {
                "базовая_информация": self._clean(base_info),
                "регистрация": self._clean(registration),
                "участники": self._clean(participants),
                "жизненный_цикл": self._clean(lifecycle),
                "контроль": self._clean(control_info),
                "задачи": self._clean(tasks_info),
                "безопасность": self._clean(security),
                "связи_и_вложения": self._clean(relations),
                "дополнительная_информация": self._clean(additional),
            }

            if specialized:
                result["специализированная_информация"] = self._clean(specialized)

            return self._clean(result)

        except Exception as exc:
            logger.error(
                "document_nlp_processing_failed",
                exc_info=True,
                extra={"error": str(exc)},
            )
            return {"error": "Ошибка обработки документа", "details": str(exc)}

    # ── Category-specific processors ─────────────────────────────────────────

    def _process_appeal(self, doc: Any) -> dict[str, Any] | None:
        """Extract appeal-specific fields from ``documentAppeal`` / ``appeal``.

        Args:
            doc: Raw document data (dict or entity).

        Returns:
            Appeal analytics dict or None if no appeal data.
        """
        app = self._safe(doc, "documentAppeal") or self._safe(doc, "appeal")
        if not app:
            return None

        return {
            "заявитель": (
                self._safe(app, "fioApplicant") or self._safe(app, "applicant_name")
            ),
            "организация": (
                self._safe(app, "organizationName")
                or self._safe(app, "organization_name")
            ),
            "тип": (
                "Коллективное" if self._safe(app, "collective") else "Индивидуальное"
            ),
            "телефон": self._safe(app, "phone"),
            "email": self._safe(app, "email"),
            "адрес": (
                self._safe(app, "fullAddress") or self._safe(app, "full_address")
            ),
            "дата_поступления": self._format_date(
                self._safe(app, "receiptDate") or self._safe(app, "receipt_date")
            ),
            "ход_рассмотрения": (
                self._safe(app, "reviewProgress") or self._safe(app, "review_progress")
            ),
        }

    def _process_meeting(self, doc: Any) -> dict[str, Any] | None:
        """Extract meeting-specific fields.

        Args:
            doc: Raw document data.

        Returns:
            Meeting analytics dict or None if no meeting data.
        """
        date_meeting = self._safe(doc, "dateMeeting") or self._safe(doc, "date_meeting")
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
            "место": (
                self._safe(doc, "placeMeeting") or self._safe(doc, "place_meeting")
            ),
            "председатель": self._format_user(self._safe(doc, "chairperson")),
            "секретарь": self._format_user(self._safe(doc, "secretary")),
            "количество_приглашенных": (
                self._safe(doc, "inviteesCount") or self._safe(doc, "invitees_count")
            ),
        }

    def _process_contract(self, doc: Any) -> dict[str, Any] | None:
        """Extract contract-specific fields from Java ``DocumentDto``.

        Java fields: contractSum, contractNumber, contractDate,
        contractSigningDate, contractDurationStart, contractDurationEnd,
        contractAgreement, contractAutoProlongation, contractTypical.

        Args:
            doc: Raw document data.

        Returns:
            Contract analytics dict or None if no contract data present.
        """
        # Проверяем наличие хоть одного контрактного поля
        contract_sum = self._safe(doc, "contractSum") or self._safe(doc, "contract_sum")
        contract_number = self._safe(doc, "contractNumber") or self._safe(
            doc, "contract_number"
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
            "дата_начала": self._format_date(
                self._safe(doc, "contractDurationStart")
                or self._safe(doc, "contract_duration_start")
            ),
            "дата_окончания": self._format_date(
                self._safe(doc, "contractDurationEnd")
                or self._safe(doc, "contract_duration_end")
            ),
            "согласован": self._safe(doc, "contractAgreement"),
            "автопролонгация": self._safe(doc, "contractAutoProlongation"),
            "типовой": self._safe(doc, "contractTypical"),
        }

    def _format_task(self, t: Any) -> dict[str, Any]:
        """Format a single task record for LLM context.

        FIX: Previously ``author`` was mapped as ``"исполнитель"`` — this is
        semantically wrong. In Java ``TaskDto``:
            - ``author`` = UserInfoDto of the person who CREATED / ASSIGNED the task.
            - ``taskExecutors`` = List<TaskExecutorsDto> of the people who must EXECUTE it.

        Correct mapping:
            - ``author`` → ``"постановщик"`` (who assigned the task)
            - ``taskExecutors[responsible=True].executor`` → ``"ответственный_исполнитель"``
            - all ``taskExecutors`` → ``"все_исполнители"``

        Args:
            t: Raw task dict or domain Task entity.

        Returns:
            Formatted task dict suitable for LLM prompt injection.
        """
        # ── Responsible executor (ответственный исполнитель) ──────────────────
        # Priority: domain entity pre-computes this; fallback to raw taskExecutors
        responsible_executor = self._format_user(
            self._safe(t, "responsible_executor")  # domain entity attr
            or self._safe(t, "responsibleExecutor")  # Java camelCase
        )

        if not responsible_executor:
            # Resolve from taskExecutors list (Java DTO path)
            responsible_executor = self._resolve_responsible_executor(t)

        # ── All executors list ────────────────────────────────────────────────
        all_executors = self._resolve_all_executors(t)

        # ── Deadline ──────────────────────────────────────────────────────────
        deadline_raw = self._safe(t, "planedDateEnd") or self._safe(  # Java DTO field
            t, "deadline"
        )  # domain entity field
        deadline_str = self._format_date(deadline_raw) if deadline_raw else "Бессрочно"

        # ── Status ────────────────────────────────────────────────────────────
        status_raw = self._safe(t, "taskStatus") or self._safe(t, "status")
        if hasattr(status_raw, "value"):
            status_str = status_raw.value
        else:
            status_str = str(status_raw) if status_raw else None

        return {
            "номер": self._safe(t, "taskNumber") or self._safe(t, "task_number"),
            "текст": (
                self._safe(t, "taskText")  # Java DTO field
                or self._safe(t, "text")  # domain entity field
            ),
            "постановщик": self._format_user(self._safe(t, "author")),
            "ответственный_исполнитель": responsible_executor,
            "все_исполнители": all_executors if all_executors else None,
            "срок": deadline_str,
            "статус": status_str,
            "бессрочное": bool(self._safe(t, "endless") or self._safe(t, "is_endless")),
        }

    def _resolve_responsible_executor(self, t: Any) -> str | None:
        """Resolve responsible executor from ``taskExecutors`` list.

        Finds the executor entry with ``responsible=True`` in
        ``TaskDto.taskExecutors`` (``List<TaskExecutorsDto>``).

        Args:
            t: Raw task dict or domain Task entity.

        Returns:
            Formatted full name of the responsible executor, or None.
        """
        executors_raw = (
            self._safe(t, "taskExecutors")  # Java DTO field
            or self._safe(t, "executors")  # domain entity field
            or []
        )
        if not executors_raw:
            return None

        for ex in executors_raw:
            is_responsible = self._safe(ex, "responsible") or self._safe(
                ex, "is_responsible"
            )
            if is_responsible:
                executor_user = self._safe(
                    ex, "executor"
                ) or self._safe(  # Java: TaskExecutorsDto.executor
                    ex, "employee"
                )
                return self._format_user(executor_user)

        # Fallback: return first executor if none is marked responsible
        first = executors_raw[0]
        executor_user = self._safe(first, "executor") or self._safe(first, "employee")
        return self._format_user(executor_user)

    def _resolve_all_executors(self, t: Any) -> list[str]:
        """Resolve all executors from ``taskExecutors`` list.

        Args:
            t: Raw task dict or domain Task entity.

        Returns:
            List of formatted full names. Empty list when no executors.
        """
        executors_raw = (
            self._safe(t, "taskExecutors") or self._safe(t, "executors") or []
        )
        result: list[str] = []
        for ex in executors_raw:
            executor_user = self._safe(ex, "executor") or self._safe(ex, "employee")
            name = self._format_user(executor_user)
            if name:
                is_responsible = self._safe(ex, "responsible") or self._safe(
                    ex, "is_responsible"
                )
                label = f"{name} (ответственный)" if is_responsible else name
                result.append(label)
        return result

    # ── Field access helpers ──────────────────────────────────────────────────

    @staticmethod
    def _safe(obj: Any, key: str, default: Any = None) -> Any:
        """Null-safe field access for both dict and object.

        Never raises — returns ``default`` on any access failure.

        Args:
            obj: Dict or object to access.
            key: Field name (supports both camelCase and snake_case).
            default: Value to return if field is missing or None.

        Returns:
            Field value or ``default``.
        """
        if obj is None:
            return default
        try:
            if isinstance(obj, dict):
                val = obj.get(key, default)
            else:
                val = getattr(obj, key, default)
            return val if val is not None else default
        except Exception:
            return default

    @staticmethod
    def _format_user(user: Any) -> str | None:
        """Format user dict or object to full name string.

        Tries camelCase fields (Java DTO) first, then snake_case (domain entity).
        Falls back to ``name`` / ``fullName`` / ``fullPostName`` field.

        Args:
            user: User dict or object with firstName/lastName fields.

        Returns:
            Full name string (``"Иванов Иван Иванович"``) or None.
        """
        if not user:
            return None
        try:
            if isinstance(user, dict):
                parts = [
                    user.get("lastName") or "",
                    user.get("firstName") or "",
                    user.get("middleName") or "",
                ]
                name = " ".join(p for p in parts if p).strip()
                return (
                    name
                    or user.get("name")
                    or user.get("fullName")
                    or user.get("fullPostName")
                )
            else:
                parts = [
                    getattr(user, "last_name", "")
                    or getattr(user, "lastName", "")
                    or "",
                    getattr(user, "first_name", "")
                    or getattr(user, "firstName", "")
                    or "",
                    getattr(user, "middle_name", "")
                    or getattr(user, "middleName", "")
                    or "",
                ]
                name = " ".join(p for p in parts if p).strip()
                return name or getattr(user, "name", None)
        except Exception:
            return None

    @staticmethod
    def _format_date(raw: Any) -> str | None:
        """Format date to ``DD.MM.YYYY`` string.

        Args:
            raw: datetime object, ISO string, or None.

        Returns:
            Formatted date string or None.
        """
        if not raw:
            return None
        try:
            if hasattr(raw, "strftime"):
                return raw.strftime("%d.%m.%Y")
            s = str(raw)
            if len(s) >= 10:
                parts = s[:10].split("-")
                if len(parts) == 3:
                    return f"{parts[2]}.{parts[1]}.{parts[0]}"
            return s[:10]
        except Exception:
            return None

    @staticmethod
    def _format_datetime(raw: Any) -> str | None:
        """Format datetime to ``DD.MM.YYYY HH:MM`` string.

        Args:
            raw: datetime object, ISO string, or None.

        Returns:
            Formatted datetime string or None.
        """
        if not raw:
            return None
        try:
            if hasattr(raw, "strftime"):
                return raw.strftime("%d.%m.%Y %H:%M")
            s = str(raw)
            if len(s) >= 16:
                parts = s[:10].split("-")
                time = s[11:16]
                if len(parts) == 3:
                    return f"{parts[2]}.{parts[1]}.{parts[0]} {time}"
            return s[:16]
        except Exception:
            return None

    @staticmethod
    def _clean(d: Any) -> Any:
        """Recursively remove None, empty dicts, lists, and strings.

        Args:
            d: Any nested data structure.

        Returns:
            Cleaned structure. None if everything was empty.
        """
        if isinstance(d, dict):
            cleaned = {k: DocumentNLPService._clean(v) for k, v in d.items()}
            cleaned = {k: v for k, v in cleaned.items() if v not in (None, {}, [], "")}
            return cleaned or None
        if isinstance(d, list):
            cleaned = [DocumentNLPService._clean(i) for i in d]
            cleaned = [i for i in cleaned if i not in (None, {}, [], "")]
            return cleaned or None
        return d
