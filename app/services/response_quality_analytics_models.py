from dataclasses import dataclass


@dataclass(frozen=True)
class ResponseQualityAnalyticsResult:
    """
    Deterministic session-level response-quality analytics.

    Analytics are observational only.
    """

    total_turns: int = 0

    acceptable_turns: int = 0
    advisory_turns: int = 0
    attention_turns: int = 0

    issue_turns: int = 0
    total_issues: int = 0

    attention_rate: float = 0.0

    average_character_count: float = 0.0
    average_question_count: float = 0.0

    latest_quality_status: str | None = None
    latest_policy_status: str | None = None