from dataclasses import dataclass


@dataclass(frozen=True)
class ResponseQualityTurnRecord:
    """
    Immutable snapshot of response-quality telemetry
    for one completed Tutor turn.

    This record is analytics-only.
    """

    turn_number: int

    quality_status: str
    policy_status: str

    issue_count: int
    requires_attention: bool

    character_count: int
    token_like_count: int
    question_count: int
    sentence_count: int

    too_long: bool
    too_short: bool
    too_many_questions: bool
    repetitive: bool

    issues: tuple[str, ...] = ()

    max_characters: int = 1200
    min_characters: int = 3
    max_questions: int = 2
    detect_repetition: bool = True