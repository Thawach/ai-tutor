from dataclasses import dataclass, field


@dataclass(frozen=True)
class ResponseQualityResult:
    """
    Deterministic response-form quality assessment.

    This result is telemetry-only in Step 16.17.3A.

    status:
    - acceptable
    - warning
    - uncertain
    """

    status: str

    reason: str

    issues: list[str] = field(
        default_factory=list
    )

    character_count: int = 0

    token_like_count: int = 0

    question_count: int = 0

    sentence_count: int = 0

    too_long: bool = False

    too_short: bool = False

    too_many_questions: bool = False

    repetitive: bool = False