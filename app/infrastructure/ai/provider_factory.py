from app.core.config import AI_PROVIDER

from app.infrastructure.ai.ollama_provider import (
    OllamaProvider,
)

from app.infrastructure.ai.groq_provider import (
    GroqProvider,
)


def create_chat_provider():
    """
    สร้าง Chat/LLM provider ตามค่า AI_PROVIDER ใน .env

    รองรับ:
    - ollama
    - groq
    """

    if AI_PROVIDER == "groq":
        return GroqProvider()

    if AI_PROVIDER == "ollama":
        return OllamaProvider()

    raise ValueError(
        f"Unsupported AI_PROVIDER: {AI_PROVIDER}"
    )


def create_embedding_provider():
    """
    Embedding provider ของระบบ

    ปัจจุบันยังใช้ Ollama + nomic-embed-text
    ไม่ขึ้นกับ AI_PROVIDER
    """

    return OllamaProvider()