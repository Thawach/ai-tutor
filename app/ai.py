from app.infrastructure.ai.provider_factory import (
    create_chat_provider,
    create_embedding_provider,
)

from app.infrastructure.ai.usage_collector import (
    AIUsageCollector,
)

from app.infrastructure.ai.task_profiles import (
    get_task_profile,
)

_chat_provider = create_chat_provider()
_embedding_provider = create_embedding_provider()

_usage_collector = AIUsageCollector()


def chat_with_ai(
    messages: list,
    task_name: str = "chat",
) -> str:
    """
    เรียก learner-facing/general text generation
    โดยเลือก token budget ตาม task_name
    """

    profile = get_task_profile(
        task_name=task_name,
        structured=False,
    )

    response = _chat_provider.generate(
        messages=messages,
        max_completion_tokens=(
            profile.max_completion_tokens
        ),
        reasoning_effort=(
            profile.reasoning_effort
        ),
    )

    usage = getattr(
        _chat_provider,
        "last_usage",
        None,
    )

    _usage_collector.add(
        component=task_name,
        usage=usage,
    )

    return response


def chat_with_structured_ai(
    messages: list,
    schema_name: str,
    schema: dict,
    task_name: str = "structured",
) -> str:
    """
    เรียก Structured Output
    โดยเลือก token budget ตาม task_name
    """

    generate_structured = getattr(
        _chat_provider,
        "generate_structured",
        None,
    )

    if generate_structured is None:
        raise RuntimeError(
            "Current chat provider does not support "
            "structured generation."
        )

    profile = get_task_profile(
        task_name=task_name,
        structured=True,
    )

    response = generate_structured(
        messages=messages,
        schema_name=schema_name,
        schema=schema,
        max_completion_tokens=(
            profile.max_completion_tokens
        ),
        reasoning_effort=(
            profile.reasoning_effort
        ),
    )

    usage = getattr(
        _chat_provider,
        "last_usage",
        None,
    )

    _usage_collector.add(
        component=task_name,
        usage=usage,
    )

    return response


def create_embedding(text: str) -> list[float]:
    """
    Compatibility function สำหรับสร้าง embedding

    Embedding ยังคงใช้ OllamaProvider
    และ EMBEDDING_MODEL ที่กำหนดไว้
    """

    return _embedding_provider.embed(
        text
    )
   

def reset_ai_usage():
    """
    เริ่ม usage collection สำหรับ Tutor turn ใหม่
    """

    _usage_collector.reset()


def get_ai_usage_summary() -> dict:
    """
    คืน usage summary ของ Tutor turn ปัจจุบัน
    """

    return _usage_collector.get_summary()


def get_ai_usage_records():
    """
    คืน raw usage records ของ turn ปัจจุบัน
    """

    return _usage_collector.get_records()


    return _embedding_provider.embed(
        text
    )