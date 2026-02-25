# src/ai_edms_assistant/domain/value_objects/meeting.py
"""Meeting-related value objects (questions, speakers)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field

from ..entities.base import DomainModel
from ..entities.employee import UserInfo

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SpeakerType(StrEnum):
    """Type of speaker in a meeting question.

    Attributes:
        MAIN: Primary speaker/presenter (основной докладчик).
        ADDITIONAL: Additional/co-speaker (содокладчик).
    """

    MAIN = "MAIN"
    ADDITIONAL = "ADDITIONAL"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


class Speaker(DomainModel):
    """Speaker/presenter for a meeting question.

    Maps to SpeakerDto from Java. Immutable because speaker assignments
    are set when the meeting agenda is created and should not change.

    Attributes:
        id: Speaker record UUID.
        employee: UserInfo reference to the speaking employee.
        speaker_type: Whether this is the main or additional speaker.
        order: Presentation order (for multiple speakers).

    Example:
        >>> from uuid import uuid4
        >>> speaker = Speaker(
        ...     id=uuid4(),
        ...     employee=UserInfo(...),
        ...     speaker_type=SpeakerType.MAIN,
        ... )
        >>> speaker.is_main_speaker
        True
    """

    id: UUID
    employee: UserInfo
    speaker_type: SpeakerType = Field(default=SpeakerType.MAIN, alias="speakerType")
    order: int = 0

    @property
    def is_main_speaker(self) -> bool:
        """Returns True for the primary speaker.

        Returns:
            ``True`` when ``speaker_type == SpeakerType.MAIN``.
        """
        return self.speaker_type == SpeakerType.MAIN

    def __str__(self) -> str:
        """Returns human-readable summary."""
        role = "Докладчик" if self.is_main_speaker else "Содокладчик"
        return f"{role}: {self.employee.name}"


class DocumentQuestion(DomainModel):
    """Question (agenda item) in a meeting document.

    Maps to DocumentQuestionDto from Java. Represents a single item
    on the meeting agenda with speakers, related documents, and decisions.

    Immutable because meeting agendas are typically frozen before the meeting.

    Attributes:
        id: Question record UUID.
        number_question: Sequence number in the agenda (1, 2, 3...).
        question_text: Full text of the agenda item.
        short_text: Brief summary for display in lists.
        speakers: List of speakers presenting this question.
        related_documents: UUIDs of documents referenced in this question.
        decision_text: Text of the decision made on this question.
        create_date: When this question was added to the agenda.
        author: Employee who created/added this question.
    """

    id: UUID
    number_question: int = Field(alias="numberQuestion")
    question_text: str = Field(alias="questionText")
    short_text: str | None = Field(default=None, alias="shortText")

    speakers: list[Speaker] = Field(default_factory=list)
    related_documents: list[UUID] = Field(default_factory=list, alias="documents")

    decision_text: str | None = Field(default=None, alias="decisionText")
    create_date: datetime | None = Field(default=None, alias="createDate")
    author: UserInfo | None = None

    @property
    def has_speakers(self) -> bool:
        """Returns True when at least one speaker is assigned.

        Returns:
            ``True`` when the ``speakers`` list is non-empty.
        """
        return bool(self.speakers)

    @property
    def main_speaker(self) -> Speaker | None:
        """Returns the main speaker for this question.

        Returns:
            The first ``Speaker`` with ``speaker_type=MAIN``, or ``None``.
        """
        for speaker in self.speakers:
            if speaker.is_main_speaker:
                return speaker
        return None

    @property
    def has_decision(self) -> bool:
        """Returns True when a decision has been recorded.

        Returns:
            ``True`` when ``decision_text`` is not empty.
        """
        return bool(self.decision_text)

    def __str__(self) -> str:
        """Returns human-readable summary."""
        return f"Вопрос №{self.number_question}: {self.short_text or self.question_text[:50]}"
