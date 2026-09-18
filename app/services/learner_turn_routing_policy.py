from app.services.learner_turn_intent_models import (
    LearnerTurnIntentResult,
)

from app.services.learner_turn_routing_models import (
    LearnerTurnRoutingDecision,
)


class LearnerTurnRoutingPolicy:
    """
    Deterministic routing policy for learner turns.

    The policy does not:
    - call an LLM
    - perform evaluation
    - perform retrieval
    - modify TutorState
    - modify learner input
    - generate Tutor responses
    """

    ROUTE_EVALUATE_ANSWER = (
        "evaluate_answer"
    )

    ROUTE_FOLLOW_UP_QUESTION = (
        "follow_up_question"
    )

    ROUTE_CLARIFICATION_QUESTION = (
        "clarification_question"
    )

    ROUTE_TOPIC_CHANGE = (
        "topic_change"
    )

    ROUTE_EVALUATE_UNCERTAIN = (
        "evaluate_uncertain"
    )

    def decide(
        self,
        intent_result: LearnerTurnIntentResult,
    ) -> LearnerTurnRoutingDecision:

        if not isinstance(
            intent_result,
            LearnerTurnIntentResult,
        ):
            raise TypeError(
                "intent_result must be a "
                "LearnerTurnIntentResult."
            )

        intent = intent_result.intent

        # =====================================================
        # Normal learner answer
        # =====================================================

        if intent == "answer":

            return LearnerTurnRoutingDecision(
                route=(
                    self.ROUTE_EVALUATE_ANSWER
                ),
                reason=(
                    "Learner turn is answer-like and "
                    "should be evaluated against the "
                    "previous Tutor question."
                ),
                should_evaluate_response=True,
                use_current_message_for_retrieval=False,
                include_previous_tutor_context=True,
                replace_active_question=False,
            )

        # =====================================================
        # Follow-up learner question
        # =====================================================

        if intent == "follow_up_question":

            return LearnerTurnRoutingDecision(
                route=(
                    self.ROUTE_FOLLOW_UP_QUESTION
                ),
                reason=(
                    "Learner turn is a follow-up question "
                    "and must not be scored as an answer."
                ),
                should_evaluate_response=False,
                use_current_message_for_retrieval=True,
                include_previous_tutor_context=True,
                replace_active_question=True,
            )

        # =====================================================
        # Clarification question
        # =====================================================

        if intent == "clarification_question":

            return LearnerTurnRoutingDecision(
                route=(
                    self.ROUTE_CLARIFICATION_QUESTION
                ),
                reason=(
                    "Learner requests clarification of the "
                    "current tutoring context and must not "
                    "be scored as an answer."
                ),
                should_evaluate_response=False,
                use_current_message_for_retrieval=True,
                include_previous_tutor_context=True,
                replace_active_question=False,
            )

        # =====================================================
        # Explicit topic change
        # =====================================================

        if intent == "topic_change":

            return LearnerTurnRoutingDecision(
                route=(
                    self.ROUTE_TOPIC_CHANGE
                ),
                reason=(
                    "Learner explicitly starts a new topic; "
                    "previous Tutor context must not drive "
                    "the new retrieval query."
                ),
                should_evaluate_response=False,
                use_current_message_for_retrieval=True,
                include_previous_tutor_context=False,
                replace_active_question=True,
            )

        # =====================================================
        # Uncertain intent — conservative fallback
        # =====================================================

        if intent == "uncertain":

            return LearnerTurnRoutingDecision(
                route=(
                    self.ROUTE_EVALUATE_UNCERTAIN
                ),
                reason=(
                    "Learner-turn intent is uncertain; "
                    "existing evaluation behavior is "
                    "preserved as a conservative fallback."
                ),
                should_evaluate_response=True,
                use_current_message_for_retrieval=False,
                include_previous_tutor_context=True,
                replace_active_question=False,
            )

        # =====================================================
        # Unknown intent — fail safely
        # =====================================================

        return LearnerTurnRoutingDecision(
            route=(
                self.ROUTE_EVALUATE_UNCERTAIN
            ),
            reason=(
                "Unknown learner-turn intent; existing "
                "evaluation behavior is preserved "
                "fail-safely."
            ),
            should_evaluate_response=True,
            use_current_message_for_retrieval=False,
            include_previous_tutor_context=True,
            replace_active_question=False,
        )