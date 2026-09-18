from app.services.learner_progress_models import (
    LearnerProgressSnapshot,
)

from app.services.learner_progress_history_models import (
    LearnerProgressHistoryRecord,
)


class LearnerProgressHistoryStore:
    """
    In-memory history of learner progress snapshots.

    Observation-only responsibilities:
    - record snapshots
    - preserve turn order
    - preserve learning-sequence boundaries
    - expose immutable history views
    - clear history on explicit session reset

    This store does not:
    - call an LLM
    - mutate LearnerState
    - mutate LearnerProgressSnapshot
    - change scaffolding
    - change routing
    - change evaluation
    - change Tutor generation
    """

    def __init__(self):

        self._records: list[
            LearnerProgressHistoryRecord
        ] = []

        self._current_sequence_id = 0

    # =========================================================
    # RECORD
    # =========================================================

    def record(
        self,
        snapshot: LearnerProgressSnapshot,
        *,
        starts_new_sequence: bool = False,
    ) -> LearnerProgressHistoryRecord:

        if not isinstance(
            snapshot,
            LearnerProgressSnapshot,
        ):
            raise TypeError(
                "snapshot must be a "
                "LearnerProgressSnapshot."
            )

        if not isinstance(
            starts_new_sequence,
            bool,
        ):
            raise TypeError(
                "starts_new_sequence must be bool."
            )

        if snapshot.session_turn < 1:
            raise ValueError(
                "snapshot.session_turn must be "
                "at least 1 before recording."
            )

        # -----------------------------------------------------
        # Turn-order integrity
        # -----------------------------------------------------

        if self._records:

            previous_turn = (
                self._records[-1]
                .snapshot
                .session_turn
            )

            if (
                snapshot.session_turn
                <= previous_turn
            ):
                raise ValueError(
                    "snapshot.session_turn must be "
                    "greater than the previously "
                    "recorded turn."
                )

        # -----------------------------------------------------
        # Learning-sequence identity
        # -----------------------------------------------------

        actual_sequence_start = (
            starts_new_sequence
        )

        if not self._records:

            self._current_sequence_id = 1

            actual_sequence_start = True

        elif starts_new_sequence:

            self._current_sequence_id += 1

        record = (
            LearnerProgressHistoryRecord(
                history_index=(
                    len(self._records) + 1
                ),
                learning_sequence_id=(
                    self._current_sequence_id
                ),
                starts_new_sequence=(
                    actual_sequence_start
                ),
                snapshot=snapshot,
            )
        )

        self._records.append(
            record
        )

        return record

    # =========================================================
    # READ
    # =========================================================

    def get_records(
        self,
    ) -> tuple[
        LearnerProgressHistoryRecord,
        ...
    ]:

        return tuple(
            self._records
        )

    def get_latest(
        self,
    ) -> LearnerProgressHistoryRecord | None:

        if not self._records:
            return None

        return self._records[-1]

    def count(
        self,
    ) -> int:

        return len(
            self._records
        )

    def current_sequence_id(
        self,
    ) -> int:

        return self._current_sequence_id

    # =========================================================
    # RESET
    # =========================================================

    def clear(
        self,
    ) -> None:

        self._records.clear()

        self._current_sequence_id = 0