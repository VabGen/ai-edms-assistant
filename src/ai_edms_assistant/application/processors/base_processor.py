# src/ai_edms_assistant/application/processors/base_processor.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class AbstractProcessor(ABC, Generic[InputT, OutputT]):
    """Abstract processor interface.

    Args:
        InputT: Input type (e.g. FileMetadata, Document).
        OutputT: Output type (e.g. str, dict).
    """

    @abstractmethod
    async def process(self, input_data: InputT) -> OutputT:
        """Process input and return output.

        Args:
            input_data: Input data to process.

        Returns:
            Processed output.
        """
