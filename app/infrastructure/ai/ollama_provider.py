from ollama import Client

from app.core.config import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    EMBEDDING_MODEL,
)


class OllamaProvider:
    """
    Provider สำหรับติดต่อ Ollama

    มีหน้าที่หลัก 2 อย่าง:
    1. สร้างข้อความด้วย Language Model
    2. สร้าง Embedding จากข้อความ
    """

    def __init__(self):
        self.client = Client(
            host=OLLAMA_HOST
        )

    def generate(
        self,
        messages: list,
        model: str | None = None,
    ) -> str:
        """
        ส่งข้อความไปยัง Ollama Chat API
        แล้วคืนข้อความที่ AI สร้างขึ้น
        """

        response = self.client.chat(
            model=model or OLLAMA_MODEL,
            messages=messages,
        )

        return response["message"]["content"]

    def embed(
        self,
        text: str,
    ) -> list[float]:
        """
        แปลงข้อความเป็น embedding vector
        สำหรับใช้กับ RAG / Vector Search
        """

        response = self.client.embeddings(
            model=EMBEDDING_MODEL,
            prompt=text,
        )

        return response["embedding"]