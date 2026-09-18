from dataclasses import dataclass


@dataclass(frozen=True)
class LearnerTurnIntentResult:
    """
    Deterministic classification of one learner turn.

    Intent:
    - answer
    - follow_up_question
    - clarification_question
    - topic_change
    - uncertain
    """

    intent: str
    reason: str

    confidence: float = 0.0

    signals: tuple[str, ...] = ()

    is_question: bool = False