import os
from dotenv import load_dotenv


load_dotenv()

ACTIVE_COURSE = os.getenv(
    "ACTIVE_COURSE",
    "electronics",
)

AI_PROVIDER = os.getenv(
    "AI_PROVIDER",
    "ollama"
).strip().lower()

SUPPORTED_AI_PROVIDERS = {
    "ollama",
    "groq",
}

if AI_PROVIDER not in SUPPORTED_AI_PROVIDERS:
    raise ValueError(
        f"Unsupported AI_PROVIDER: {AI_PROVIDER}"
    )

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY",
    ""
).strip()

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)


OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:8b"
)

if AI_PROVIDER == "groq":

    ACTIVE_CHAT_MODEL = GROQ_MODEL

elif AI_PROVIDER == "ollama":

    ACTIVE_CHAT_MODEL = OLLAMA_MODEL

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)

APP_NAME = os.getenv(
    "APP_NAME",
    "AI Tutor"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text"
)

CHROMA_PATH = os.getenv(
    "CHROMA_PATH",
    "./data/chroma"
)

DOCUMENT_PATH = os.getenv(
    "DOCUMENT_PATH",
    "./docs"
)