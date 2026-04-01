# edms_ai_assistant/tools/__init__.py
"""
EDMS AI Assistant — Tool Registry.
"""

from .appeal_autofill import autofill_appeal_document
from .attachment import doc_get_file_content
from .create_document_from_file import create_document_from_file
from .doc_notification import doc_send_notification
from .doc_search import doc_search_tool
from .doc_update_field import doc_update_field
from .document import doc_get_details
from .document_comparison import doc_compare_documents
from .document_versions import doc_get_versions
from .employee_search import employee_search_tool
from .file_compare_tool import doc_compare_attachment_with_local
from .introduction import introduction_create_tool
from .local_file_tool import read_local_file_content
from .summarization import doc_summarize_text
from .task import task_create_tool

all_tools = [
    # Documents
    doc_get_details,
    doc_get_versions,
    doc_compare_documents,
    doc_search_tool,
    create_document_from_file,
    # Content
    doc_get_file_content,
    read_local_file_content,
    doc_compare_attachment_with_local,
    # Analysis
    doc_summarize_text,
    # Workflow
    introduction_create_tool,
    task_create_tool,
    autofill_appeal_document,
    doc_update_field,
    # People
    employee_search_tool,
    # Notifications
    doc_send_notification,
]

__all__ = [
    "all_tools",
    "autofill_appeal_document",
    "doc_compare_attachment_with_local",
    "doc_compare_documents",
    "doc_get_details",
    "doc_get_file_content",
    "doc_get_versions",
    "doc_search_tool",
    "doc_send_notification",
    "doc_summarize_text",
    "doc_update_field",
    "employee_search_tool",
    "introduction_create_tool",
    "read_local_file_content",
    "task_create_tool",
    "create_document_from_file"
]
