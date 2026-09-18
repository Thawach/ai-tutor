from dataclasses import dataclass


@dataclass(frozen=True)
class AITaskProfile:
    """
    กำหนด runtime profile ของ LLM task แต่ละประเภท
    """

    max_completion_tokens: int
    reasoning_effort: str = "low"


TASK_PROFILES = {

    # -------------------------
    # Learner-facing generation
    # -------------------------

    "tutor": AITaskProfile(
        max_completion_tokens=400,
        reasoning_effort="low",
    ),

    "guiding_question_validator": AITaskProfile(
        max_completion_tokens=1200,
        reasoning_effort="low",
    ),

    "response_repair": AITaskProfile(
        max_completion_tokens=400,
        reasoning_effort="low",
    ),

    "evidence_safe_translation": AITaskProfile(
        max_completion_tokens=800,
        reasoning_effort="low",
    ),

    # -------------------------
    # Internal structured tasks
    # -------------------------

    "evaluator": AITaskProfile(
        max_completion_tokens=600,
        reasoning_effort="low",
    ),

    "relevance_gate": AITaskProfile(
        max_completion_tokens=600,
        reasoning_effort="low",
    ),

    "repair_evidence_selector": AITaskProfile(
        max_completion_tokens=1200,
        reasoning_effort="low",
    ),

    "response_validator": AITaskProfile(
        max_completion_tokens=1200,
        reasoning_effort="low",
    ),

    "response_validator_repair": AITaskProfile(
        max_completion_tokens=1600,
        reasoning_effort="low",
    ),

    "pedagogical_validator": AITaskProfile(
        max_completion_tokens=2400,
        reasoning_effort="low",
    ),

    "pedagogical_validator_repair": AITaskProfile(
        max_completion_tokens=2400,
        reasoning_effort="low",
    ),
}


DEFAULT_TEXT_PROFILE = AITaskProfile(
    max_completion_tokens=400,
    reasoning_effort="low",
)


DEFAULT_STRUCTURED_PROFILE = AITaskProfile(
    max_completion_tokens=600,
    reasoning_effort="low",
)


def get_task_profile(
    task_name: str,
    structured: bool = False,
) -> AITaskProfile:

    profile = TASK_PROFILES.get(
        task_name
    )

    if profile is not None:
        return profile

    if structured:
        return DEFAULT_STRUCTURED_PROFILE

    return DEFAULT_TEXT_PROFILE