"""
EDMS AI Assistant - Appeal Extraction Service
"""

import asyncio
import logging
import re
from datetime import datetime

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from edms_ai_assistant.llm_client import get_llm_client
from mcp_server.models.appeal_fields import AppealFields

logger = logging.getLogger(__name__)


_CITY_STOPWORDS: frozenset[str] = frozenset(
    {
        "республики",
        "республика",
        "беларуси",
        "беларусь",
        "беларуские",
        "области",
        "область",
        "района",
        "район",
        "министерства",
        "министерство",
        "исполнительного",
        "исполнительный",
        "комитета",
        "комитет",
        "совета",
        "совет",
        "центра",
        "центр",
        "лет",
        "годов",
        "года",
        "улицы",
        "улица",
        "проспекта",
        "проспект",
        "переулка",
        "переулок",
    }
)

# Таблица: первые 3 цифры почтового индекса → город
_POSTAL_PREFIX_CITY: dict[str, str] = {
    "210": "Витебск",
    "211": "Витебск",
    "212": "Могилёв",
    "213": "Могилёв",
    "220": "Минск",
    "221": "Минск",
    "222": "Молодечно",
    "223": "Борисов",
    "224": "Пинск",
    "225": "Брест",
    "230": "Гродно",
    "231": "Лида",
    "232": "Гродно",
    "236": "Барановичи",
    "246": "Гомель",
    "247": "Гомель",
    "248": "Гомель",
}

_PHONE_CODE_CITY: dict[str, str] = {
    "17": "Минск",
    "162": "Брест",
    "163": "Барановичи",
    "152": "Гродно",
    "154": "Лида",
    "232": "Гомель",
    "236": "Молодечно",
    "222": "Могилёв",
    "212": "Витебск",
    "174": "Борисов",
    "224": "Пинск",
}
_MINSK_SHORT_PHONE_FIRST_DIGITS: frozenset[str] = frozenset({"2", "3"})


class AppealExtractionService:
    """
    Сервис для извлечения структурированных данных из обращений граждан.
    """

    MIN_TEXT_LENGTH = 30
    MAX_TEXT_LENGTH = 12000
    DEFAULT_MAX_RETRIES = 3
    BASE_RETRY_DELAY = 2

    def __init__(self):
        base_llm = get_llm_client()
        self.extraction_llm = base_llm.with_config({"temperature": 0.0})
        self._last_raw_text: str | None = (
            None  # для city fallback в _post_process_fields
        )
        logger.info("AppealExtractionService initialized with temperature=0.0")

    async def extract_appeal_fields(self, text: str) -> AppealFields:
        if not self._validate_text_length(text):
            return AppealFields()

        self._last_raw_text = text

        try:
            parser = JsonOutputParser(pydantic_object=AppealFields)
            prompt = self._build_extraction_prompt()
            chain = prompt | self.extraction_llm | parser

            preprocessed_text = self._preprocess_text(text)
            truncated_text = self._truncate_text(preprocessed_text)

            logger.debug(
                "Invoking LLM for extraction",
                extra={"text_length": len(truncated_text)},
            )

            result = await chain.ainvoke(
                {
                    "text": truncated_text,
                    "format_instructions": parser.get_format_instructions(),
                }
            )

            if isinstance(result, dict):
                if result.get("shortSummary"):
                    ss = str(result["shortSummary"]).strip()
                    if len(ss) > 80:
                        result["shortSummary"] = ss[:80]
                        logger.warning(
                            "Pre-validate: shortSummary слишком длинный (%d символов) → обрезан до 80",
                            len(ss),
                        )
                    else:
                        result["shortSummary"] = ss
                if (
                    result.get("organizationName")
                    and len(str(result["organizationName"])) > 300
                ):
                    result["organizationName"] = str(result["organizationName"])[:300]
                if result.get("fullAddress") and len(str(result["fullAddress"])) > 500:
                    result["fullAddress"] = str(result["fullAddress"])[:500]
            appeal_data = AppealFields.model_validate(result)
            appeal_data = self._post_process_fields(appeal_data, truncated_text)

            logger.info(
                "Appeal data extracted successfully",
                extra={
                    "has_fio": bool(appeal_data.fioApplicant),
                    "has_org": bool(appeal_data.organizationName),
                    "declarant_type": appeal_data.declarantType,
                    "citizen_type": appeal_data.citizenType,
                    "has_city": bool(appeal_data.cityName),
                },
            )

            return appeal_data

        except Exception as e:
            logger.error(
                f"LLM extraction failed: {type(e).__name__}: {e}",
                exc_info=True,
            )
            return AppealFields()

    def _post_process_fields(self, fields: AppealFields, text: str) -> AppealFields:
        """
        Post-processing для извлечения данных, которые LLM мог пропустить.

        Fixes:
        1. Парсинг даты из correspondentOrgNumber если dateDocCorrespondentOrg пуст
        2. Извлечение города из fullAddress если cityName пуст
        """
        if fields.declarantType == "ENTITY":
            if not fields.dateDocCorrespondentOrg and fields.correspondentOrgNumber:
                parsed_date = self._parse_date_from_number(
                    fields.correspondentOrgNumber
                )
                if parsed_date:
                    fields.dateDocCorrespondentOrg = parsed_date
                    logger.info(
                        f"Parsed date from correspondentOrgNumber: {parsed_date}"
                    )

        if not fields.cityName and fields.fullAddress:
            extracted_city = self._extract_city_from_address(fields.fullAddress)
            if extracted_city:
                fields.cityName = extracted_city
                logger.info(f"Extracted city from address: {extracted_city}")

        # Оставляем в signed только ФИО, убираем должность
        if fields.signed and len(fields.signed) > 20:
            cleaned_signed = self._extract_fio_from_signed(fields.signed)
            if cleaned_signed != fields.signed:
                logger.info(
                    "signed trimmed: '%s' → '%s'", fields.signed[:60], cleaned_signed
                )
                fields.signed = cleaned_signed

        if fields.index and fields.fullAddress:
            if fields.index not in fields.fullAddress:
                logger.info(
                    "index '%s' not in fullAddress '%s' — clearing",
                    fields.index,
                    fields.fullAddress[:60],
                )
                fields.index = None
        elif fields.index and not fields.fullAddress:
            # Нет адреса заявителя — индекс невозможно верифицировать
            logger.info("index '%s' cleared: fullAddress is empty", fields.index)
            fields.index = None

        if fields.declarantType == "ENTITY" and not fields.organizationName:
            proximity = self._recover_org_from_address_proximity(text)
            if proximity:
                fields.organizationName = proximity
                logger.info(
                    "organizationName from address proximity: %s", proximity[:60]
                )
            else:
                recovered = self._recover_org_name_from_text(text)
                if recovered:
                    fields.organizationName = recovered
                    logger.info(
                        "Recovered organizationName from Russian text: %s",
                        recovered[:60],
                    )

        # Fallback: если city всё ещё None — ищем по всему тексту вокруг контактов
        if not fields.cityName and self._last_raw_text:
            fallback_city = self._extract_city_from_full_text(
                self._last_raw_text,
                phone=fields.phone,
                email=fields.email,
                index=fields.index,
            )
            if fallback_city:
                fields.cityName = fallback_city
                logger.info(
                    "City via full-text fallback: %s (phone=%s email=%s index=%s)",
                    fallback_city,
                    fields.phone,
                    fields.email,
                    fields.index,
                )

        return fields

    def _parse_date_from_number(self, number: str) -> str | None:
        """
        Парсит дату из исходящего номера документа.

        Patterns:
        - "№ 123 от 15.01.2025"
        - "№ 01-01/26" (26 = 2026 год, 01 = январь)
        - "исх. № 45/2 от 12.02.2024"
        """
        date_patterns = [
            r"от\s+(\d{1,2})\.(\d{1,2})\.(\d{4})",
            r"от\s+(\d{1,2})\s+([а-яА-Я]+)\s+(\d{4})",
        ]

        for pattern in date_patterns:
            match = re.search(pattern, number)
            if match:
                try:
                    day, month, year = match.groups()

                    if month.isdigit():
                        month_num = int(month)
                    else:
                        month_map = {
                            "января": 1,
                            "февраля": 2,
                            "марта": 3,
                            "апреля": 4,
                            "мая": 5,
                            "июня": 6,
                            "июля": 7,
                            "августа": 8,
                            "сентября": 9,
                            "октября": 10,
                            "ноября": 11,
                            "декабря": 12,
                        }
                        month_num = month_map.get(month.lower(), 1)

                    dt = datetime(int(year), month_num, int(day))
                    return dt.isoformat() + "Z"
                except (ValueError, KeyError):
                    pass

        slash_pattern = r"(\d{1,2})-(\d{1,2})/(\d{2})\b"
        match = re.search(slash_pattern, number)
        if match:
            try:
                month_num, day_num, year_short = match.groups()
                # Проверяем что это не просто ID-номер (месяц 1-12, день 1-31)
                m_int, d_int, y_short = int(month_num), int(day_num), int(year_short)
                if 1 <= m_int <= 12 and 1 <= d_int <= 31:
                    year_full = 2000 + y_short
                    dt = datetime(year_full, m_int, d_int)
                    return dt.isoformat() + "Z"
            except ValueError:
                pass

        return None

    def _extract_city_from_address(self, address: str) -> str | None:
        """
        Извлекает город из адреса.

        Patterns:
        - "г. Минск"
        - "Минск, ул. ..."
        - "220004, Минск"
        """
        # Паттерн 1: "г. Город" (с обязательным пробелом → не матчит "г.\nРеспублики")
        m = re.search(r"\bг\.\s+([А-ЯЁ][а-яё]{3,})", address)
        if m and m.group(1).lower() not in _CITY_STOPWORDS:
            return m.group(1).strip()

        # Паттерн 2: "220004, Минск" (индекс + город)
        m = re.search(r"\d{6},?\s*([А-ЯЁ][а-яё]{3,})", address)
        if m and m.group(1).lower() not in _CITY_STOPWORDS:
            return m.group(1).strip()

        # Паттерн 3: "Минск, ул." (город перед улицей)
        m = re.search(r"([А-ЯЁ][а-яё]{3,}),\s*ул\.", address)
        if m and m.group(1).lower() not in _CITY_STOPWORDS:
            return m.group(1).strip()

        return None

    @staticmethod
    def _extract_city_from_full_text(
        text: str,
        phone: str | None = None,
        email: str | None = None,
        index: str | None = None,
    ) -> str | None:
        """Fallback city extraction from the full document text.

        Used when LLM returned cityName=None but we still have contact data.
        Strategy (in priority order):
        1. Direct postal index param → _POSTAL_PREFIX_CITY lookup.
        2. "г. Город" near phone/email lines (safe: filtered by _CITY_STOPWORDS).
        3. Postal index near phone/email → _POSTAL_PREFIX_CITY.
        4. "г. Город" anywhere in last 30 lines (safe).
        5. Postal index in last 30 lines.

                Safety: "г.?" without point can match "г." from "2026 г." in text —
        results are filtered through _CITY_STOPWORDS and min-length check.

        Args:
            text: Full extracted document text.
            phone: Applicant phone (anchor for context search).
            email: Applicant email (anchor for context search).
            index: Postal index already extracted from AppealFields.

        Returns:
            City name string or None.
        """

        def _city_from_postal(fragment: str) -> str | None:
            """Find postal index in text and look up city name."""
            m = re.search(r"\b(2[012][0-9]{4})\b", fragment)
            if m:
                return _POSTAL_PREFIX_CITY.get(m.group(1)[:3])
            return None

        # 1. Прямой lookup по уже извлечённому полю index (AppealFields.index)
        if index and len(index) >= 3:
            city = _POSTAL_PREFIX_CITY.get(index[:3])
            if city:
                return city

        if not text:
            return None

        lines = text.split("\n")
        phone_digits = re.sub(r"[\s\-\(\)]", "", phone) if phone else ""
        email_lower = email.lower() if email else ""

        anchor_indices: list[int] = []
        for i, line in enumerate(lines):
            line_clean = re.sub(r"[\s\-\(\)]", "", line)
            if (phone_digits and phone_digits in line_clean) or (
                email_lower and email_lower in line.lower()
            ):
                anchor_indices.append(i)

        def _city_from_phone_code(phone_str: str) -> str | None:
            """Determine Belarusian city from phone number format.

            Handles formats:
            - "205-55-65"       → 7-digit Minsk (starts with 2 or 3)
            - "017-205-55-65"   → with city code 017 = Minsk
            - "+375-17-205-65"  → with country code +375, city code 17
            """
            digits = re.sub(r"[^\d]", "", phone_str)
            # Убираем код страны +375 в начале
            if digits.startswith("375"):
                digits = digits[3:]
            # Убираем ведущий ноль кода города: 017xx → 17xx
            if digits.startswith("0") and len(digits) >= 3:
                digits = digits[1:]

            # 7-значный минский (205-55-65 → first digit "2" or "3")
            if len(digits) == 7 and digits[0] in _MINSK_SHORT_PHONE_FIRST_DIGITS:
                return "Минск"

            # Длинный с кодом города: пробуем срезы начала
            for code_len in (3, 2):
                code = digits[:code_len]
                if code in _PHONE_CODE_CITY:
                    return _PHONE_CODE_CITY[code]

            return None

        # 2. Ищем почтовый индекс в ±5 строках от контактов заявителя
        for anchor in anchor_indices:
            context = "\n".join(lines[max(0, anchor - 5) : anchor + 3])
            city = _city_from_postal(context)
            if city:
                return city

        # 3. Почтовый индекс в последних 30 строках документа
        tail = "\n".join(lines[-30:])
        city = _city_from_postal(tail)
        if city:
            return city

        # 4. Определяем город по телефонному коду (fallback)
        if phone:
            city = _city_from_phone_code(phone)
            if city:
                logger.debug("City from phone code: %s → %s", phone, city)
                return city

        return None

    async def extract_with_retry(
        self,
        text: str,
        max_attempts: int | None = None,
    ) -> AppealFields:
        max_attempts = max_attempts or self.DEFAULT_MAX_RETRIES

        for attempt in range(1, max_attempts + 1):
            try:
                result = await self.extract_appeal_fields(text)

                if self._is_valid_extraction(result):
                    logger.info(
                        f"Extraction successful on attempt {attempt}/{max_attempts}",
                        extra={"attempt": attempt, "max_attempts": max_attempts},
                    )
                    return result

                logger.warning(
                    f"Attempt {attempt}/{max_attempts}: LLM returned insufficient data",
                    extra={"attempt": attempt},
                )

            except Exception as e:
                logger.error(
                    f"Attempt {attempt}/{max_attempts} failed: {type(e).__name__}: {e}",
                    extra={"attempt": attempt, "error": str(e)},
                )

            if attempt < max_attempts:
                wait_time = self._calculate_retry_delay(attempt)
                logger.info(
                    f"Waiting {wait_time}s before retry...",
                    extra={"wait_time": wait_time, "attempt": attempt},
                )
                await asyncio.sleep(wait_time)

        logger.error(
            f"Extraction failed after {max_attempts} attempts. Returning empty object.",
            extra={"max_attempts": max_attempts},
        )
        return AppealFields()

    def _validate_text_length(self, text: str) -> bool:
        if not text:
            logger.warning("Empty text provided for extraction")
            return False

        text_length = len(text.strip())

        if text_length < self.MIN_TEXT_LENGTH:
            logger.warning(
                f"Text too short for analysis (min: {self.MIN_TEXT_LENGTH}, got: {text_length})"
            )
            return False

        return True

    def _truncate_text(self, text: str) -> str:
        if len(text) <= self.MAX_TEXT_LENGTH:
            return text

        logger.debug(
            f"Truncating text for LLM: {len(text)} → {self.MAX_TEXT_LENGTH} chars"
        )
        return text[: self.MAX_TEXT_LENGTH]

    def _preprocess_text(self, text: str) -> str:
        """Remove Belarusian-language lines from bilingual appeal documents.

        Many official letters from Belarusian government organisations are
        formatted with parallel Belarusian and Russian columns.  The LLM
        often picks up the Belarusian variant of the organisation name,
        which is never found in the Russian-language EDMS reference books.

        Detection strategy: lines containing the characters ``ў`` or ``і``
        (unique to Belarusian orthography, absent from Russian) are dropped.
        Lines without these markers — including mixed or transliterated text —
        are kept, so Russian content is preserved intact.

        Args:
            text: Raw extracted text from the attachment.

        Returns:
            Text with Belarusian-only lines removed.
        """
        _BELARUSIAN_MARKERS = frozenset("ўЎіІ")
        lines = text.split("\n")
        filtered = [ln for ln in lines if not any(c in _BELARUSIAN_MARKERS for c in ln)]
        result = "\n".join(filtered)
        if len(result) < len(text):
            removed = len(lines) - len(filtered)
            logger.debug(
                "Bilingual preprocess: removed %d Belarusian line(s), %d → %d chars",
                removed,
                len(text),
                len(result),
            )
        return result

    @staticmethod
    def _is_valid_extraction(result: AppealFields) -> bool:
        return any(
            [
                result.fioApplicant,
                result.organizationName,
                result.shortSummary,
            ]
        )

    @classmethod
    def _calculate_retry_delay(cls, attempt: int) -> int:
        return cls.BASE_RETRY_DELAY**attempt

    @staticmethod
    def _recover_org_name_from_text(text: str) -> str | None:
        """Attempt to extract a Russian organisation name from raw text.

        Used as a fallback when the primary LLM extraction picked the
        Belarusian variant of the name (subsequently cleared by anti-
        hallucination logic).  Searches for common patterns that appear
        in Belarusian official correspondence headers:

        * Quoted names:  «Национальный центр электронных услуг»
        * RUE/SUE prefixes followed by quoted content
        * Abbreviations in parentheses: (ГП «НЦЭУ»)

        Args:
            text: Raw or preprocessed document text.

        Returns:
            Best candidate organisation name string, or None.
        """
        # Паттерн 1: «Название организации» в кавычках (любые скобки-кавычки)
        quoted_pattern = re.compile(
            r'[«""]([А-ЯЁа-яё][^«»""\n]{5,80})[»""]',
            re.MULTILINE,
        )
        # Паттерн 2: РУП / ГП / ОАО / РУПП / ГУ перед кавычкой или названием
        org_prefix_pattern = re.compile(
            r"(?:республиканское унитарное предприятие|"
            r"государственное предприятие|"
            r"открытое акционерное общество|"
            r"государственное учреждение|"
            r"\bруп\b|\bгп\b|\bгу\b|\bоао\b)"
            r'\s+[«""]([^«»""\n]{5,80})[»""]',
            re.IGNORECASE | re.MULTILINE,
        )

        # Приоритет: org_prefix_pattern → quoted_pattern
        for pattern in (org_prefix_pattern, quoted_pattern):
            for m in pattern.finditer(text):
                candidate = m.group(1).strip()
                # Исключаем явно белорусские слова
                if any(c in candidate for c in "ўЎіІ"):
                    continue
                # Минимальная длина и не похоже на адрес
                if len(candidate) >= 8 and not re.search(
                    r"\d{5,}|ул\.|пр\.", candidate
                ):
                    return candidate

        return None

    @staticmethod
    def _recover_org_from_address_proximity(text: str) -> str | None:
        """Extract the organisation name closest to the address/contacts block.

        In official Belarusian correspondence, letterheads follow this structure:

            [Superior authority (line 1)]   ← NOT the author
            «Actual author organisation»     ← THIS is organizationName
            (abbreviation)
            address, phone, e-mail           ← contact block

        This method finds the quoted name on the line immediately preceding
        the first contact block (address/phone/e-mail) in the text.

        Args:
            text: Preprocessed (Russian-only) document text.

        Returns:
            Organisation name string, or None.
        """
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        contact_pattern = re.compile(
            r"(?:ул\.|пр\.|пер\.|бул\.|e-mail|тел\.|факс|@|\d{6})",
            re.IGNORECASE,
        )
        quoted_pattern = re.compile(r'[«"\""]([А-ЯЁа-яё][^«»"\""\'\n]{4,70})[»"\""]')

        for i, line in enumerate(lines):
            if contact_pattern.search(line):
                # Ищем строку с кавычками в 1–4 строках выше
                for j in range(i - 1, max(i - 5, -1), -1):
                    m = quoted_pattern.search(lines[j])
                    if m:
                        candidate = m.group(1).strip()
                        # Исключаем белорусский текст
                        if not any(c in candidate for c in "ўЎіІ"):
                            return candidate
                break
        return None

    @staticmethod
    def _extract_fio_from_signed(text: str) -> str:
        """Extract FIO (name only) from a 'signed' field that may include a job title.

        Handles both compact (``Д.Д.``) and spaced (``Д. Д.``) initial formats,
        as LLMs sometimes insert spaces between initials.

        Supported FIO formats (matched from the end of the string):
        - ``"И.О. Фамилия"`` / ``"И. О. Фамилия"``   — initials + surname
        - ``"Фамилия И.О."`` / ``"Фамилия И. О."``   — surname + initials
        - ``"Фамилия Имя Отчество"`` — full name (3 capitalised words)

        Args:
            text: Raw signer string from the letter.

        Returns:
            Extracted FIO string, or the original string if no pattern matched.
        """
        if not text or not text.strip():
            return text

        # Паттерн 1а: "И.О. Фамилия" (без пробелов в инициалах)
        m = re.search(r"([А-ЯЁ]\.[А-ЯЁ]\.\s+[А-ЯЁ][а-яё]+)\s*$", text)
        if m:
            return m.group(1).strip()

        # Паттерн 1б: "И. О. Фамилия" (с пробелами в инициалах — LLM-вариант)
        m = re.search(r"([А-ЯЁ]\.\s+[А-ЯЁ]\.\s+[А-ЯЁ][а-яё]+)\s*$", text)
        if m:
            return m.group(1).strip()

        # Паттерн 2а: "Фамилия И.О." в конце
        m = re.search(r"([А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.[А-ЯЁ]\.)\s*$", text)
        if m:
            return m.group(1).strip()

        # Паттерн 2б: "Фамилия И. О." в конце
        m = re.search(r"([А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s+[А-ЯЁ]\.)\s*$", text)
        if m:
            return m.group(1).strip()

        # Паттерн 3: три слова с заглавными буквами (Фамилия Имя Отчество)
        m = re.search(r"([А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+)\s*$", text)
        if m:
            return m.group(1).strip()

        return text

    def _build_extraction_prompt(self) -> ChatPromptTemplate:
        system_message = """Ты — эксперт-аналитик системы электронного документооборота (СЭД).
Твоя задача: проанализировать текст официального обращения и извлечь факты для заполнения регистрационной карточки.

═══════════════════════════════════════════════════════════════════════════════
ПРАВИЛА ИЗВЛЕЧЕНИЯ И КЛАССИФИКАЦИИ
═══════════════════════════════════════════════════════════════════════════════

1️⃣ **declarantType (Тип заявителя)** — КРИТИЧЕСКИ ВАЖНО:

   ✅ INDIVIDUAL (Физическое лицо):
   - Обращение написано от имени гражданина
   - Подпись в формате "ФИО" без должности
   - Личные вопросы (ЖКХ, пенсия, медицина)
   - Примеры: "Иванов И.И.", "гражданин Петров", "житель г. Минска"

   ✅ ENTITY (Юридическое лицо):
   - Обращение от организации, предприятия, госоргана
   - Указана должность подписавшего (Директор, Председатель)
   - Есть исходящий номер документа
   - Примеры: "ОАО 'Завод'", "Минздрав РБ", "Администрация района", "МБОО"

2️⃣ **Персональные данные**:

   - **fioApplicant**: ФИО заявителя
     * Для INDIVIDUAL: автор обращения
     * Для ENTITY: подписант (если указан)

   - **organizationName**: Название организации-АВТОРА письма (той, чей бланк)
     * ТОЛЬКО для ENTITY
     * ❗ КРИТИЧНО: В шапке белорусских официальных писем ЧАСТО указаны ДВЕ организации:
       - Строка 1: вышестоящий орган (учредитель, регулятор)
       - Строка 2: сама организация-автор (чьи адрес, тел., email указаны ниже)
     * Правило: organizationName = организация, рядом с которой стоят адрес/телефон/email.
       Вышестоящий орган из первой строки — НЕ является автором письма.
     * Примеры:
       - Шапка: "ОАЦ при Президенте РБ / РУП «НЦЭУ» / ул. Притыцкого, тел..." → organizationName = "НЦЭУ"
       - Шапка: "Министерство финансов / ГУ «Центр» / ул. ... тел..." → organizationName = "Центр"
       - Обычное письмо от "ООО Стройком" → organizationName = "ООО Стройком"

3️⃣ **Даты и Номера** (для ENTITY):

   - **correspondentOrgNumber**: Исходящий номер документа
     * Примеры: "№ 123 от 15.01.2025", "исх. № 45/2", "№ 01-01/26"
     * ВАЖНО: Извлекай ПОЛНОСТЬЮ "как есть" вместе с датой!

   - **dateDocCorrespondentOrg**: Дата исходящего номера
     * Формат ISO 8601: "2025-01-15T00:00:00Z"
     * Извлекается из:
       1. Явной даты в тексте: "26 марта 2018 года" → "2018-03-26T00:00:00Z"
       2. Даты в номере: "№ 123 от 15.01.2025" → "2025-01-15T00:00:00Z"
       3. Закодированной даты: "№ 01-01/26" → месяц=01, день=01, год=2026 → "2026-01-01T00:00:00Z"

   - **receiptDate**: Дата НАПИСАНИЯ/ПОДПИСАНИЯ обращения (из текста письма).
     * Это дата самого письма, НЕ дата сегодня, НЕ дата регистрации в системе.
     * Ищи в начале письма: "26 января 2026 года", "15.03.2025", "март 2025"
     * Если дата письма не указана явно → null (НЕ придумывай текущую дату!)
     * Для ENTITY: если в тексте есть "26 января 2026 года № 01-01/26" → receiptDate="2026-01-26T00:00:00Z"

4️⃣ **Логические признаки**:

   - **reasonably (Обоснованность)**: true, если есть:
     * Ссылки на законы/нормативные акты
     * Конкретные факты, даты, суммы
     * Расчеты или доказательства

   - **collective**: true, если подписано группой лиц
     * Примеры: "жители дома", "группа родителей", "Мы, родители детей"

   - **anonymous**: true, если автор скрыт
     * Нет подписи или "Аноним"

5️⃣ **Контактная информация заявителя**:

   - **fullAddress**: ЛЮБОЕ упоминание адреса рядом с контактами заявителя.
     Даже неполный адрес: "Маяковского 5", "ул. Ленина 12, кв. 3" — извлекай как есть.
     Не требуй наличия города — адрес может быть без города.
   - **phone**: любой номер телефона/факса в тексте
   - **email**: любой email (должен содержать '@')
   - **signed**: ФИО подписанта — ТОЛЬКО имя, БЕЗ должности.
     Примеры: "Директор Иванов И.И." → signed="Иванов И.И."
              "Заместитель ... Д. Д. Жуков" → signed="Д. Д. Жуков"
     Если подписи нет — signed=null

6️⃣ **Краткое содержание (shortSummary)**:

   - Сформулируй СУТЬ обращения — ЗАКОНЧЕННАЯ мысль, СТРОГО не более 80 символов.
   - Это заголовок документа — он должен быть ПОНЯТНЫМ и ЗАВЕРШЁННЫМ.
   - ПРИМЕРЫ (считай символы!):
     ✅ "Право детей с аутизмом на образование" — 38 симв.
     ✅ "Жалоба на качество услуг ЖКХ" — 30 симв.
     ✅ "Обеспечение прав детей-аутистов на образование" — 47 симв.
     ✅ "Нарушение прав инвалидов на персональное сопровождение" — 55 симв.
     ❌ "Обеспечение права детей-инвалидов-аутистов на персональное сопровождение в" — 78 симв., обрывается на "в"!
   - ПРАВИЛО: заголовок должен заканчиваться на СУЩЕСТВИТЕЛЬНОМ или ГЛАГОЛЕ, не на предлоге.
   - Если суть не умещается в 80 — выбери ГЛАВНУЮ тему, остальное опусти.
   - НЕ включай название организации — оно уже есть в отдельном поле.

7️⃣ **Пересылка (correspondentAppeal)**:

   correspondentAppeal — это орган, который ПЕРЕСЛАЛ нам данное письмо.
   Это НЕ адресат (кому направлено), НЕ список копий.

   ✅ Заполняй ТОЛЬКО если есть явные признаки пересылки:
   - "Перенаправлено из ...", "Переслано от ...", "По поручению ..."
   - "Поступило из ...", "Направлено из ..."
   - В шапке письма явно указан орган-пересыльщик (а не адресат)

   ❌ НЕ заполняй correspondentAppeal если:
   - В шапке указаны адресаты ("Министерство образования РБ") — это получатели, не пересылающий орган
   - Строки "Копии:" / "Копия:" содержат органы — это рассылка копий, не пересылка
   - Письмо написано заявителем самим и отправлено напрямую
   - МБОО написала Министерству образования → correspondentAppeal = null
   - Гражданин написал в горисполком → correspondentAppeal = null

   Примеры ПРАВИЛЬНОГО заполнения:
   - "Направляем обращение, поступившее из Администрации Президента РБ" → "Администрация Президента РБ"
   - "Переслано по компетенции из Минздрава РБ" → "Минздрав РБ"
   - Шапка: "МБОО → Министерству образования" → correspondentAppeal = null

8️⃣ **citizenType (Вид обращения)** — КРИТИЧЕСКИ ВАЖНО:

   ✅ "Жалоба" если: "жалоб", "претензи", "недовольн", "нарушен"
   ✅ "Заявление" если: "заявлен", "прошу", "просьба", "рассмотреть"
   ✅ "Предложение" если: "предлага", "инициатив", "улучшить"
   ✅ "Запрос" если: "запрос информаци", "просим предоставить"
   ✅ "Благодарность" если: "благодар", "спасибо"

9️⃣ **География (country, regionName, districtName, cityName)**:

   **country** (страна):
   - Извлекай ТОЛЬКО если явно упомянута: "Беларусь", "Республика Беларусь", "Россия"
   - Если не упомянута → null

   **cityName** (ВЫСШИЙ ПРИОРИТЕТ — извлекай всегда):
   - Ищи "г. Город" или "город Город" в адресе заявителя ИЛИ рядом с его контактами (телефон, email):
     * "г. Минск" → "Минск", "город Гомель" → "Гомель"
     * "220004, Минск, ул. Ленина" → "Минск"
     * "Маяковского 5, г. Минск" → "Минск"
   - Если адрес неполный (только улица без города), ищи город:
     * В той же строке или ±2 строки от адреса заявителя
     * В строках рядом с телефоном/email организации
     * В реквизитах организации-отправителя (не в шапке адресата!)
   - Если индекс известен (например 220xxx = Минск) — определи город по индексу:
     220xxx = Минск, 225xxx = Брест, 230xxx/232xxx = Гродно, 246xxx-248xxx = Гомель,
     212xxx = Могилёв, 210xxx = Витебск
   - ВАЖНО: НЕ бери город из шапки АДРЕСАТА письма (Министерство, Акимат, ...)
   - Убирай префиксы "г.", "город"

   **regionName** (область):
   - Извлекай ТОЛЬКО если явно указана в адресе заявителя:
     * "Минская область" → regionName="Минская область"
     * "Гомельская обл." → regionName="Гомельская область"
   - Если не указана → null (система определит по городу)

   **districtName** (административный район):
   - Извлекай ТОЛЬКО если район явно указан в АДРЕСЕ ЗАЯВИТЕЛЯ
     (шапка письма, строка "Адрес:", "проживающего по адресу:")
   - ✅ "проживающего: Гомельский район, г. Гомель" → districtName="Гомельский"
   - ✅ "220030, г. Минск, Центральный район, ул. Ленина" → districtName="Центральный"
   - ❌ "суд Октябрьского района г. Гомель" → null (адрес суда, не заявителя)
   - ❌ "администрация Советского района" → null (название организации)
   - ❌ "В производстве суда Октябрьского района..." → null (содержание письма)
   - Если не уверен → null. Система определит по городу автоматически.

   ПРИМЕРЫ:

   📝 "пр. Победителей, 23, к.2, 220004, Минск"
   → cityName="Минск", regionName=null, districtName=null
   (система определит область и район по городу)

   📝 "г. Молодечно, Минская область"
   → cityName="Молодечно", regionName="Минская область", districtName=null

   📝 "проживающего по адресу: Гомельский район, г. Гомель, ул. Пушкина 1"
   → cityName="Гомель", regionName=null, districtName="Гомельский"

   📝 "В производстве суда Октябрьского района г. Гомель находится дело..."
   → cityName="Гомель", regionName=null, districtName=null
   (Октябрьский — район суда в тексте, не адрес заявителя)

═══════════════════════════════════════════════════════════════════════════════
ВАЖНО
═══════════════════════════════════════════════════════════════════════════════

❗ Если информации по полю НЕТ — оставляй **null** (НЕ строку "None", "N/A")
❗ НЕ ПРИДУМЫВАЙ данные — извлекай ТОЛЬКО то, что явно написано
❗ Ответ ТОЛЬКО в формате JSON (без текста, комментариев, markdown)
❗ Все даты в формате ISO 8601 с суффиксом Z
❗ Для ENTITY обязательно пытайся извлечь дату из correspondentOrgNumber!

ПРИМЕРЫ ПРАВИЛЬНОГО ЗАПОЛНЕНИЯ:
- "fioApplicant": null (НЕ "None")
- "cityName": "Гомель" (извлечено из "Октябрьского района г. Гомель")
- "districtName": null (район суда в тексте — не адрес заявителя)
- "regionName": null (если не указана явно — EDMS заполняет по городу автоматически)
- "dateDocCorrespondentOrg": "2026-01-01T00:00:00Z" (извлечено из "№ 01-01/26")
- "correspondentOrgNumber": "№ 01-01/26" (полностью с датой)
"""

        user_message = """Текст обращения для анализа:
───────────────────────────────────────────────────────────────────────────────
{text}
───────────────────────────────────────────────────────────────────────────────

Инструкции по формату JSON-ответа:
{format_instructions}
"""

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("user", user_message),
            ]
        )
