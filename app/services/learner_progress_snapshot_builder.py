from app.domain.learner.state import (
    LearnerState,
)

from app.services.learner_progress_models import (
    LearnerProgressSnapshot,
)


class LearnerProgressSnapshotBuilder:
    """
    Deterministically builds a read-only learner
    progress snapshot from the active LearnerState.

    This builder does not:
    - call an LLM
    - mutate LearnerState
    - change scaffolding
    - change routing
    - change evaluation
    - generate Tutor responses
    """

    def build(
        self,
        *,
        state: LearnerState,
        current_turn_evaluation: str | None = None,
        learner_turn_intent: str | None = None,
        learner_turn_route: str | None = None,
    ) -> LearnerProgressSnapshot:

        if not isinstance(
            state,
            LearnerState,
        ):
            raise TypeError(
                "state must be a LearnerState."
            )

        self._validate_optional_string(
            current_turn_evaluation,
            "current_turn_evaluation",
        )

        self._validate_optional_string(
            learner_turn_intent,
            "learner_turn_intent",
        )

        self._validate_optional_string(
            learner_turn_route,
            "learner_turn_route",
        )

        active_records = (
            state.get_active_misconceptions()
        )

        active_misconceptions = tuple(
            item.description
            for item in active_records
            if (
                getattr(
                    item,
                    "description",
                    None,
                )
            )
        )

        return LearnerProgressSnapshot(
            session_turn=(
                state.turn_count
            ),
            scaffolding_level=(
                state.scaffolding_level
            ),
            attempt_count=(
                state.attempt_count
            ),
            hint_count=(
                state.hint_count
            ),
            correct_streak=(
                state.correct_streak
            ),
            partial_streak=(
                state.partial_streak
            ),
            failure_streak=(
                state.failure_streak
            ),
            sequence_last_evaluation=(
                state.last_evaluation
            ),
            current_turn_evaluation=(
                current_turn_evaluation
            ),
            evaluation_performed=(
                current_turn_evaluation
                is not None
            ),
            learner_turn_intent=(
                learner_turn_intent
            ),
            learner_turn_route=(
                learner_turn_route
            ),
            active_misconception_count=(
                len(active_misconceptions)
            ),
            active_misconceptions=(
                active_misconceptions
            ),
        )

    @staticmethod
    def _validate_optional_string(
        value: str | None,
        field_name: str,
    ) -> None:

        if (
            value is not None
            and
            not isinstance(
                value,
                str,
            )
        ):
            raise TypeError(
                f"{field_name} must be "
                "a string or None."
            )