# import logging
# from typing import Dict, Any, Optional
# from langchain_core.tools import tool
# from pydantic import BaseModel, Field
#
# from edms_ai_assistant.clients.employee_client import EmployeeClient
#
# from edms_ai_assistant.generated.resources_openapi import (
#     Include1,
#     EmployeeDto,
# )
# from edms_ai_assistant.services.nlp_service import EDMSNaturalLanguageService
#
# logger = logging.getLogger(__name__)
#
#
# class EmployeeSearchInput(BaseModel):
#     """Схема входных данных для инструмента."""
#
#     token: str = Field(..., description="Токен авторизации пользователя (JWT)")
#     employee_id: Optional[str] = Field(
#         None,
#         description="UUID конкретного сотрудника. Используется для получения детальной карточки после выбора из списка.",
#     )
#     lastName: Optional[str] = Field(None, description="Фамилия")
#     firstName: Optional[str] = Field(None, description="Имя")
#     middleName: Optional[str] = Field(None, description="Отчество")
#     fullPostName: Optional[str] = Field(None, description="Полное название должности")
#
#
# @tool("employee_search_tool", args_schema=EmployeeSearchInput)
# async def employee_search_tool(
#     token: str, employee_id: Optional[str] = None, **filter_params
# ) -> Dict[str, Any]:
#     """
#     Поиск сотрудников и получение их детальных карточек.
#     Если найдено несколько человек, возвращает список для выбора.
#     """
#     nlp = EDMSNaturalLanguageService()
#
#     try:
#         async with EmployeeClient() as client:
#             if employee_id:
#                 raw_data = await client.get_employee(token, employee_id)
#                 if not raw_data:
#                     return {"status": "error", "message": "Сотрудник не найден."}
#
#                 emp = EmployeeDto.model_validate(raw_data)
#                 return {
#                     "status": "found",
#                     "employee_card": nlp.process_employee_info(emp),
#                 }
#
#             search_payload = {k: v for k, v in filter_params.items() if v}
#             if "includes" not in search_payload:
#                 search_payload["includes"] = [Include1.POST, Include1.DEPARTMENT]
#
#             if len(search_payload) <= 1:
#                 return {
#                     "status": "error",
#                     "message": "Укажите критерии поиска (например, lastName).",
#                 }
#
#             results = await client.search_employees(token, search_payload)
#
#             if not results:
#                 return {
#                     "status": "not_found",
#                     "message": "Сотрудники не найдены по данным критериям.",
#                 }
#
#             if len(results) == 1:
#                 emp = EmployeeDto.model_validate(results[0])
#                 return {
#                     "status": "found",
#                     "employee_card": nlp.process_employee_info(emp),
#                 }
#
#             choices = []
#             for r in results:
#                 emp_item = EmployeeDto.model_validate(r)
#                 choices.append(
#                     {
#                         "id": str(emp_item.id),
#                         "full_name": f"{emp_item.lastName} {emp_item.firstName} {emp_item.middleName or ''}".strip(),
#                         "post": nlp.get_safe(emp_item, "post.postName", "Не указана"),
#                         "dept": nlp.get_safe(emp_item, "department.name", "Не указан"),
#                     }
#                 )
#
#             return {
#                 "status": "requires_action",
#                 "action_type": "select_employee",
#                 "message": f"Найдено сотрудников: {len(results)}. Выберите нужного:",
#                 "choices": choices,
#             }
#
#     except Exception as e:
#         logger.error(f"[EMPLOYEE-TOOL] Error: {e}", exc_info=True)
#         return {
#             "status": "error",
#             "message": f"Ошибка при работе с реестром сотрудников: {str(e)}",
#         }
