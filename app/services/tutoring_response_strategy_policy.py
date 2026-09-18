from app.services.learner_turn_intent_models import (
    LearnerTurnIntentResult,
)

from app.services.tutoring_response_mode_models import (
    TutoringResponseModeDecision,
)


class TutoringResponseStrategyPolicy:
    """
    Deterministically maps learner-turn intent to the
    response behavior that the Tutor should use.

    This policy:
    - does not call an LLM
    - does not generate Tutor text
    - does not mutate learner state
    - does not modify scaffolding level
    """

    def decide(
        self,
        learner_turn_intent: (
            LearnerTurnIntentResult | None
        ),
    ) -> TutoringResponseModeDecision:

        # -------------------------------------------------
        # Initial learner turn
        # -------------------------------------------------

        if learner_turn_intent is None:

            return TutoringResponseModeDecision(
                mode="new_topic_scaffold",
                reason=(
                    "No prior learner-turn intent exists; "
                    "start the learning sequence with the "
                    "normal scaffolding strategy."
                ),
                direct_answer_required=False,
                guiding_question_policy="optional",
                preserve_scaffolding_strategy=True,
                max_guiding_questions=None,
            )

        if not isinstance(
            learner_turn_intent,
            LearnerTurnIntentResult,
        ):
            raise TypeError(
                "learner_turn_intent must be "
                "LearnerTurnIntentResult or None."
            )

        intent = learner_turn_intent.intent

        # -------------------------------------------------
        # Normal learner answer
        # -------------------------------------------------

        if intent == "answer":

            return TutoringResponseModeDecision(
                mode="scaffolded",
                reason=(
                    "Learner provided an answer; "
                    "preserve the active scaffolding "
                    "strategy."
                ),
                direct_answer_required=False,
                guiding_question_policy="optional",
                preserve_scaffolding_strategy=True,
                max_guiding_questions=None,
            )

        # -------------------------------------------------
        # Follow-up question
        # -------------------------------------------------

        if intent == "follow_up_question":

            return TutoringResponseModeDecision(
                mode="answer_then_guide",
                reason=(
                    "Learner asked a follow-up question; "
                    "answer the question before optionally "
                    "continuing with guided learning."
                ),
                direct_answer_required=True,
                guiding_question_policy="optional",
                preserve_scaffolding_strategy=False,
                max_guiding_questions=1,
            )

        # -------------------------------------------------
        # Clarification request
        # -------------------------------------------------

        if intent == "clarification_question":

            return TutoringResponseModeDecision(
                mode="direct_clarification",
                reason=(
                    "Learner explicitly requested "
                    "clarification; provide a direct "
                    "explanation before any optional "
                    "guiding question."
                ),
                direct_answer_required=True,
                guiding_question_policy="optional",
                preserve_scaffolding_strategy=False,
                max_guiding_questions=1,
            )

        # -------------------------------------------------
        # Explicit topic change
        # -------------------------------------------------

        if intent == "topic_change":

            return TutoringResponseModeDecision(
                mode="new_topic_scaffold",
                reason=(
                    "Learner explicitly changed topic; "
                    "start the new learning sequence with "
                    "the normal scaffolding strategy."
                ),
                direct_answer_required=False,
                guiding_question_policy="optional",
                preserve_scaffolding_strategy=True,
                max_guiding_questions=None,
            )

        # -------------------------------------------------
        # Uncertain learner turn
        # -------------------------------------------------

        if intent == "uncertain":

            return TutoringResponseModeDecision(
                mode="scaffolded",
                reason=(
                    "Learner intent is uncertain; "
                    "preserve the existing scaffolding "
                    "behavior conservatively."
                ),
                direct_answer_required=False,
                guiding_question_policy="optional",
                preserve_scaffolding_strategy=True,
                max_guiding_questions=None,
            )

        # -------------------------------------------------
        # Unknown future intent — fail-safe
        # -------------------------------------------------

        return TutoringResponseModeDecision(
            mode="scaffolded",
            reason=(
                "Unknown learner intent encountered; "
                "fall back conservatively to the "
                "existing scaffolding strategy."
            ),
            direct_answer_required=False,
            guiding_question_policy="optional",
            preserve_scaffolding_strategy=True,
            max_guiding_questions=None,
        )