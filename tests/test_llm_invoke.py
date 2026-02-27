# src\ai_edms_assistant\tests\test_llm_invoke.py
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Импорт конфигурации и LLM...")
from ai_edms_assistant.shared.config.settings import settings
from ai_edms_assistant.infrastructure.llm.providers.openai_provider import get_chat_model

print("--- Конфигурация ---")
print(f"LLM Endpoint: {settings.LLM_ENDPOINT}")
print(f"LLM Model: {settings.LLM_MODEL_NAME}")
print(f"Temperature: {settings.LLM_TEMPERATURE}")

print("\n--- Инициализация и вызов ChatModel ---")

try:
    print("Инициализация ChatModel...")
    llm = get_chat_model()
    print(f"ChatModel успешно инициализирован: {type(llm).__name__}")

    test_message = "Привет! кто ты?"
    print(f"\nОтправка сообщения: '{test_message}'")

    print("Выполняется вызов LLM...")
    response = llm.invoke(test_message)

    print(f"\nПолучен ответ от LLM:")
    print(f"Тип ответа: {type(response)}")
    print(f"Содержимое: {response}")

    if hasattr(response, 'content'):
        print(f"Текст ответа: {response.content}")
    elif hasattr(response, 'text'):
        print(f"Текст ответа: {response.text}")
    else:
        print(f"Текст ответа (str): {str(response)}")

except Exception as e:
    print(f"Ошибка при инициализации или вызове ChatModel: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Тест вызова LLM завершен ---")