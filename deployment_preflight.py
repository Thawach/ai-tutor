import sys


EXPECTED_PYTHON_MAJOR = 3
EXPECTED_PYTHON_MINOR = 11


def emit(label: str, value) -> None:
    print(f"{label:<28} {value}")


def fail(message: str) -> None:
    emit("PREFLIGHT", "FAIL")
    emit("Reason", message)
    raise SystemExit(1)


def main() -> None:
    print("AI Tutor 16.22 Deployment Preflight")
    print("-" * 48)

    # ---------------------------------------------------------
    # Python runtime
    # ---------------------------------------------------------

    version = sys.version_info

    if (
        version.major != EXPECTED_PYTHON_MAJOR
        or version.minor != EXPECTED_PYTHON_MINOR
    ):
        fail(
            "Python 3.11.x is required for the "
            "tested 16.22 deployment baseline."
        )

    emit(
        "Python",
        f"PASS ({version.major}.{version.minor}.{version.micro})",
    )

    # ---------------------------------------------------------
    # Configuration
    #
    # Import is deliberately delayed so invalid configuration
    # can be reported as a preflight failure.
    # ---------------------------------------------------------

    try:
        from app.core.config import (
            ACTIVE_COURSE,
            AI_PROVIDER,
            ACTIVE_CHAT_MODEL,
            GROQ_API_KEY,
            OLLAMA_HOST,
            EMBEDDING_MODEL,
        )
    except Exception as error:
        fail(
            f"Configuration invalid: "
            f"{type(error).__name__}: {error}"
        )

    emit("Configuration", "PASS")
    emit("AI provider", AI_PROVIDER)
    emit("Active chat model", ACTIVE_CHAT_MODEL)
    emit("Active course", ACTIVE_COURSE)

    # Never print the actual secret.
    if AI_PROVIDER == "groq":
        emit(
            "Groq API key",
            "CONFIGURED" if GROQ_API_KEY else "MISSING",
        )

    if not ACTIVE_CHAT_MODEL.strip():
        fail("Active chat model is empty.")

    if not OLLAMA_HOST.strip():
        fail("OLLAMA_HOST is empty.")

    if not EMBEDDING_MODEL.strip():
        fail("EMBEDDING_MODEL is empty.")

    emit("Ollama host", OLLAMA_HOST)
    emit("Embedding model", EMBEDDING_MODEL)

    # ---------------------------------------------------------
    # Course profile
    # ---------------------------------------------------------

    try:
        from app.courses.loader import CourseProfileLoader

        profile = CourseProfileLoader().load(
            ACTIVE_COURSE
        )
    except Exception as error:
        fail(
            f"Course profile invalid: "
            f"{type(error).__name__}: {error}"
        )

    emit("Course profile", "PASS")
    emit("Course ID", profile.course_id)
    emit("Chroma path", profile.chroma_path)

    # document_path is part of the course profile contract,
    # but source PDFs are not required by runtime deployment.
    emit("Document path", profile.document_path)

    # ---------------------------------------------------------
    # Runtime knowledge store
    #
    # IMPORTANT:
    # validate_existing_store() is existing-only.
    # It must not create a missing store or collection.
    # ---------------------------------------------------------

    try:
        from app.vector_store import validate_existing_store

        count = validate_existing_store(
            chroma_path=profile.chroma_path
        )
    except Exception as error:
        fail(
            f"Knowledge store invalid: "
            f"{type(error).__name__}: {error}"
        )

    emit("Knowledge store", "PASS")
    emit("Collection", "course_documents")
    emit("Indexed records", count)

    # ---------------------------------------------------------
    # Provider construction
    #
    # Constructors are checked, but no generate/embed method
    # is called. Therefore preflight performs no LLM/API request.
    # ---------------------------------------------------------

    try:
        from app.infrastructure.ai.provider_factory import (
            create_chat_provider,
            create_embedding_provider,
        )

        chat_provider = create_chat_provider()
        embedding_provider = create_embedding_provider()

    except Exception as error:
        fail(
            f"Provider configuration invalid: "
            f"{type(error).__name__}: {error}"
        )

    emit(
        "Chat provider",
        f"PASS ({type(chat_provider).__name__})",
    )

    emit(
        "Embedding provider",
        f"PASS ({type(embedding_provider).__name__})",
    )

    print("-" * 48)
    emit("PREFLIGHT", "PASS")


if __name__ == "__main__":
    main()
