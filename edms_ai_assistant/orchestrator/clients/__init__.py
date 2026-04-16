# edms_ai_assistant/orchestrator/clients/__init__.py
"""
Публичный API пакета clients.

Все клиенты экспортируются отсюда — импортируй только через этот модуль:
    from edms_ai_assistant.orchestrator.clients import DocumentClient, EmployeeClient

Добавляя новый клиент:
1. Создай файл <name>_client.py
2. Добавь его в __all__ и импорт ниже
"""

from edms_ai_assistant.orchestrator.clients.attachment_client import EdmsAttachmentClient
from edms_ai_assistant.orchestrator.clients.base_client import EdmsBaseClient, EdmsHttpClient
from edms_ai_assistant.orchestrator.clients.department_client import (
    BaseDepartmentClient,
    DepartmentClient,
)
from edms_ai_assistant.orchestrator.clients.document_client import (
    FULL_DOC_INCLUDES,
    SEARCH_DOC_INCLUDES,
    DocumentClient,
    EdmsDocumentClient,
)
from edms_ai_assistant.orchestrator.clients.document_creator_client import DocumentCreatorClient
from edms_ai_assistant.orchestrator.clients.employee_client import (
    BaseEmployeeClient,
    EmployeeClient,
)
from edms_ai_assistant.orchestrator.clients.group_client import BaseGroupClient, GroupClient
from edms_ai_assistant.orchestrator.clients.reference_client import ReferenceClient
from edms_ai_assistant.orchestrator.clients.task_client import BaseTaskClient, TaskClient

__all__ = [
    # Base
    "EdmsBaseClient",
    "EdmsHttpClient",
    # Document
    "EdmsDocumentClient",
    "DocumentClient",
    "FULL_DOC_INCLUDES",
    "SEARCH_DOC_INCLUDES",
    "DocumentCreatorClient",
    # Employee
    "BaseEmployeeClient",
    "EmployeeClient",
    # Department
    "BaseDepartmentClient",
    "DepartmentClient",
    # Group
    "BaseGroupClient",
    "GroupClient",
    # Task
    "BaseTaskClient",
    "TaskClient",
    # Attachment
    "EdmsAttachmentClient",
    # Reference
    "ReferenceClient",
]