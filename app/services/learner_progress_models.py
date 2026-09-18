from dataclasses import dataclass


@dataclass(frozen=True)
class LearnerProgressSnapshot:
    """
    Read-only snapshot of learner progress state.

    The snapshot is observational only.
    It must not mutate learner state or alter
    tutoring behavior.
    """

    # ---------------------------------------------
    # Session-level state
    # ---------------------------------------------

    session_turn: int

    # ---------------------------------------------
    # Current learning-sequence state
    # ---------------------------------------------

    scaffolding_level: int

    attempt_count: int
    hint_count: int

    correct_streak: int
    partial_streak: int
    failure_streak: int

    # ---------------------------------------------
    # Evaluation state
    # ---------------------------------------------

    sequence_last_evaluation: str | None

    current_turn_evaluation: str | None

    evaluation_performed: bool

    # ---------------------------------------------
    # Learner-turn context
    # ---------------------------------------------

    learner_turn_intent: str | None
    learner_turn_route: str | None

    # ---------------------------------------------
    # Learner misconception state
    # ---------------------------------------------

    active_misconception_count: int

    active_misconceptions: tuple[str, ...] = ()