# edms_ai_assistant/orchestrator/services/appeal_extraction_service.py
"""
Сервис LLM-извлечения данных из обращений граждан.

ИЗМЕНЕНИЯ (исправление архитектурного расхождения):
  - Импорт get_chat_model теперь из edms_ai_assistant.orchestrator.llm
    (ранее из edms_ai_assistant.llm — несуществующего модуля)
  - with_config() теперь поддерживается через _LangChainCompatWrapper
"""

import asyncio
import logging
import re
from datetime import datetime

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

# ИСПРАВЛЕНО: правильный импорт get_chat_model
from edms_ai_assistant.orchestrator.llm import get_chat_model
from edms_ai_assistant.orchestrator.models.appeal_fields import AppealFields

logger = logging.getLogger(__name__)


_CITY_STOPWORDS: frozenset[str] = frozenset(
    {
        "республики", "республика", "беларуси", "беларусь", "беларуские",
        "области", "область", "района", "район", "министерства", "министерство",
        "исполнительного", "исполнительный", "комитета", "комитет",
        "совета", "совет", "центра", "центр", "лет", "годов", "года",
        "улицы", "улица", "проспекта", "проспект", "переулка", "переулок",
    }
)

_POSTAL_PREFIX_CITY: dict[str, str] = {
    "210": "Витебск", "211": "Витебск", "212": "Могилёв", "213": "Могилёв",
    "220": "Минск", "221": "Минск", "222": "Молодечно", "223": "Борисов",
    "224": "Пинск", "225": "Брест", "230": "Гродно", "231": "Лида",
    "232": "Гродно", "236": "Барановичи", "246": "Гомель",
    "247": "Гомель", "248": "Гомель",
}

_PHONE_CODE_CITY: dict[str, str] = {
    "17": "Минск", "162": "Брест", "163": "Барановичи", "152": "Гродно",
    "154": "Лида", "232": "Гомель", "236": "Молодечно", "222": "Могилёв",
    "212": "Витебск", "174": "Борисов", "224": "Пинск",
}
_MINSK_SHORT_PHONE_FIRST_DIGITS: frozenset[str] = frozenset({"2", "3"})


class AppealExtractionService:
    """Сервис для извлечения структурированных данных из обращений граждан."""

    MIN_TEXT_LENGTH = 30
    MAX_TEXT_LENGTH = 12000
    DEFAULT_MAX_RETRIES = 3
    BASE_RETRY_DELAY = 2

    def __init__(self) -> None:
        # ИСПРАВЛЕНО: используем get_chat_model() из orchestrator.llm,
        # который возвращает _LangChainCompatWrapper с поддержкой with_config()
        base_llm = get_chat_model()
        self.extraction_llm = base_llm.with_config({"temperature": 0.0})
        self._last_raw_text: str | None = None
        logger.info("AppealExtractionService инициализирован с temperature=0.0")

    async def extract_appeal_fields(self, text: str) -> AppealFields:
        """Извлекает поля карточки обращения из текста через LLM.

        Args:
            text: Текст обращения гражданина.

        Returns:
            AppealFields с заполненными полями (пустые поля = None).
        """
        if not self._validate_text_length(text):
            return AppealFields()

        self._last_raw_text = text

        try:
            parser = JsonOutputParser(pydantic_object=AppealFields)
            prompt = self._build_extraction_prompt()
            chain = prompt | self.extraction_llm | parser

            preprocessed_text = self._preprocess_text(text)
            truncated_text = self._truncate_text(preprocessed_text)

            logger.debug("Вызов LLM для извлечения данных", extra={"text_length": len(truncated_text)})

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
                if result.get("organizationName") and len(str(result["organizationName"])) > 300:
                    result["organizationName"] = str(result["organizationName"])[:300]
                if result.get("fullAddress") and len(str(result["fullAddress"])) > 500:
                    result["fullAddress"] = str(result["fullAddress"])[:500]

            appeal_data = AppealFields.model_validate(result)
            appeal_data = self._post_process_fields(appeal_data, truncated_text)

            logger.info(
                "Данные обращения извлечены",
                extra={
                    "has_fio": bool(appeal_data.fioApplicant),
                    "has_org": bool(appeal_data.organizationName),
                    "declarant_type": appeal_data.declarantType,
                    "has_city": bool(appeal_data.cityName),
                },
            )
            return appeal_data

        except Exception as e:
            logger.error("Ошибка LLM-извлечения: %s: %s", type(e).__name__, e, exc_info=True)
            return AppealFields()

    def _post_process_fields(self, fields: AppealFields, text: str) -> AppealFields:
        """Post-processing: заполняем пропущенные поля эвристиками."""
        if fields.declarantType == "ENTITY":
            if not fields.dateDocCorrespondentOrg and fields.correspondentOrgNumber:
                parsed_date = self._parse_date_from_number(fields.correspondentOrgNumber)
                if parsed_date:
                    fields.dateDocCorrespondentOrg = parsed_date

        if not fields.cityName and fields.fullAddress:
            extracted_city = self._extract_city_from_address(fields.fullAddress)
            if extracted_city:
                fields.cityName = extracted_city

        if fields.signed and len(fields.signed) > 20:
            cleaned_signed = self._extract_fio_from_signed(fields.signed)
            if cleaned_signed != fields.signed:
                fields.signed = cleaned_signed

        if fields.index and fields.fullAddress:
            if fields.index not in fields.fullAddress:
                fields.index = None
        elif fields.index and not fields.fullAddress:
            fields.index = None

        if fields.declarantType == "ENTITY" and not fields.organizationName:
            proximity = self._recover_org_from_address_proximity(text)
            if proximity:
                fields.organizationName = proximity
            else:
                recovered = self._recover_org_name_from_text(text)
                if recovered:
                    fields.organizationName = recovered

        if not fields.cityName and self._last_raw_text:
            fallback_city = self._extract_city_from_full_text(
                self._last_raw_text,
                phone=fields.phone,
                email=fields.email,
                index=fields.index,
            )
            if fallback_city:
                fields.cityName = fallback_city

        return fields

    def _parse_date_from_number(self, number: str) -> str | None:
        """Парсит дату из исходящего номера документа."""
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
                            "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
                            "мая": 5, "июня": 6, "июля": 7, "августа": 8,
                            "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
                        }
                        month_num = month_map.get(month.lower(), 1)
                    dt = datetime(int(year), month_num, int(day))
                    return dt.isoformat() + "Z"
                except (ValueError, KeyError):
                    pass
        return None

    def _extract_city_from_address(self, address: str) -> str | None:
        """Извлекает город из строки адреса."""
        m = re.search(r"\bг\.\s+([А-ЯЁ][а-яё]{3,})", address)
        if m and m.group(1).lower() not in _CITY_STOPWORDS:
            return m.group(1).strip()
        m = re.search(r"\d{6},?\s*([А-ЯЁ][а-яё]{3,})", address)
        if m and m.group(1).lower() not in _CITY_STOPWORDS:
            return m.group(1).strip()
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
        """Fallback-извлечение города из полного текста документа."""
        def _city_from_postal(fragment: str) -> str | None:
            m = re.search(r"\b(2[012][0-9]{4})\b", fragment)
            if m:
                return _POSTAL_PREFIX_CITY.get(m.group(1)[:3])
            return None

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
            digits = re.sub(r"[^\d]", "", phone_str)
            if digits.startswith("375"):
                digits = digits[3:]
            if digits.startswith("0") and len(digits) >= 3:
                digits = digits[1:]
            if len(digits) == 7 and digits[0] in _MINSK_SHORT_PHONE_FIRST_DIGITS:
                return "Минск"
            for code_len in (3, 2):
                code = digits[:code_len]
                if code in _PHONE_CODE_CITY:
                    return _PHONE_CODE_CITY[code]
            return None

        for anchor in anchor_indices:
            context = "\n".join(lines[max(0, anchor - 5): anchor + 3])
            city = _city_from_postal(context)
            if city:
                return city

        tail = "\n".join(lines[-30:])
        city = _city_from_postal(tail)
        if city:
            return city

        if phone:
            city = _city_from_phone_code(phone)
            if city:
                return city

        return None

    async def extract_with_retry(
        self, text: str, max_attempts: int | None = None
    ) -> AppealFields:
        """Извлечение с повторными попытками при недостаточном результате."""
        max_attempts = max_attempts or self.DEFAULT_MAX_RETRIES
        for attempt in range(1, max_attempts + 1):
            try:
                result = await self.extract_appeal_fields(text)
                if self._is_valid_extraction(result):
                    return result
                logger.warning("Попытка %d/%d: LLM вернул недостаточно данных", attempt, max_attempts)
            except Exception as e:
                logger.error("Попытка %d/%d провалилась: %s", attempt, max_attempts, e)
            if attempt < max_attempts:
                wait_time = self._calculate_retry_delay(attempt)
                await asyncio.sleep(wait_time)
        logger.error("Извлечение не удалось после %d попыток", max_attempts)
        return AppealFields()

    def _validate_text_length(self, text: str) -> bool:
        if not text:
            return False
        return len(text.strip()) >= self.MIN_TEXT_LENGTH

    def _truncate_text(self, text: str) -> str:
        return text[: self.MAX_TEXT_LENGTH] if len(text) > self.MAX_TEXT_LENGTH else text

    def _preprocess_text(self, text: str) -> str:
        """Удаляет строки на белорусском языке из двуязычных документов."""
        _BELARUSIAN_MARKERS = frozenset("ўЎіІ")
        lines = text.split("\n")
        filtered = [ln for ln in lines if not any(c in _BELARUSIAN_MARKERS for c in ln)]
        return "\n".join(filtered)

    @staticmethod
    def _is_valid_extraction(result: AppealFields) -> bool:
        return any([result.fioApplicant, result.organizationName, result.shortSummary])

    @classmethod
    def _calculate_retry_delay(cls, attempt: int) -> int:
        return cls.BASE_RETRY_DELAY ** attempt

    @staticmethod
    def _recover_org_name_from_text(text: str) -> str | None:
        """Пытается извлечь название организации из сырого текста."""
        quoted_pattern = re.compile(r'[«""]([А-ЯЁа-яё][^«»""\n]{5,80})[»""]', re.MULTILINE)
        org_prefix_pattern = re.compile(
            r"(?:республиканское унитарное предприятие|государственное предприятие|"
            r"открытое акционерное общество|государственное учреждение|"
            r"\bруп\b|\bгп\b|\bгу\b|\bоао\b)"
            r'\s+[«""]([^«»""\n]{5,80})[»""]',
            re.IGNORECASE | re.MULTILINE,
        )
        for pattern in (org_prefix_pattern, quoted_pattern):
            for m in pattern.finditer(text):
                candidate = m.group(1).strip()
                if any(c in candidate for c in "ўЎіІ"):
                    continue
                if len(candidate) >= 8 and not re.search(r"\d{5,}|ул\.|пр\.", candidate):
                    return candidate
        return None

    @staticmethod
    def _recover_org_from_address_proximity(text: str) -> str | None:
        """Извлекает название организации рядом с контактным блоком."""
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        contact_pattern = re.compile(
            r"(?:ул\.|пр\.|пер\.|бул\.|e-mail|тел\.|факс|@|\d{6})", re.IGNORECASE
        )
        quoted_pattern = re.compile(r'[«"\""]([А-ЯЁа-яё][^«»"\""\'\n]{4,70})[»"\""]')
        for i, line in enumerate(lines):
            if contact_pattern.search(line):
                for j in range(i - 1, max(i - 5, -1), -1):
                    m = quoted_pattern.search(lines[j])
                    if m:
                        candidate = m.group(1).strip()
                        if not any(c in candidate for c in "ўЎіІ"):
                            return candidate
                break
        return None

    @staticmethod
    def _extract_fio_from_signed(text: str) -> str:
        """Извлекает ФИО из поля подписи, убирая должность."""
        if not text or not text.strip():
            return text
        for pattern in [
            r"([А-ЯЁ]\.[А-ЯЁ]\.\s+[А-ЯЁ][а-яё]+)\s*$",
            r"([А-ЯЁ]\.\s+[А-ЯЁ]\.\s+[А-ЯЁ][а-яё]+)\s*$",
            r"([А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.[А-ЯЁ]\.)\s*$",
            r"([А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s+[А-ЯЁ]\.)\s*$",
            r"([А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+)\s*$",
        ]:
            m = re.search(pattern, text)
            if m:
                return m.group(1).strip()
        return text

    def _build_extraction_prompt(self) -> "ChatPromptTemplate":
        """Строит промпт для LLM-извлечения данных обращения."""
        system_message = """Ты — эксперт-аналитик системы электронного документооборота (СЭД).
Извлеки данные из текста официального обращения для заполнения регистрационной карточки.

КЛЮЧЕВЫЕ ПРАВИЛА:
1. declarantType: INDIVIDUAL (физлицо) или ENTITY (юрлицо/организация)
2. shortSummary: до 80 символов, законченная мысль
3. dateDocCorrespondentOrg: ISO 8601 с суффиксом Z
4. correspondentAppeal: ТОЛЬКО если есть явные признаки пересылки
5. cityName: приоритет — всегда извлекать
6. Пустые поля → null (не строку "None")

Ответ строго в JSON формате."""

        user_message = """Текст обращения:
{text}

{format_instructions}"""

        return ChatPromptTemplate.from_messages(
            [("system", system_message), ("user", user_message)]
        )