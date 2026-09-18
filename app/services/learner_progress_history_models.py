from dataclasses import dataclass

from app.services.learner_progress_models import (
    LearnerProgressSnapshot,
)


@dataclass(frozen=True)
class LearnerProgressHistoryRecord:
    """
    Immutable observation record for one completed
    learner/tutor turn.

    The history record is observational only and
    must not influence tutoring behavior.
    """

    history_index: int

    learning_sequence_id: int

    starts_new_sequence: bool

    snapshot: LearnerProgressSnapshot