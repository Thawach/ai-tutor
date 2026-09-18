from dataclasses import dataclass


@dataclass(frozen=True)
class LearnerTurnRoutingDecision:
    """
    Deterministic routing decision derived from
    LearnerTurnIntentResult.

    This decision describes routing behavior only.
    It does not execute the route.
    """

    route: str
    reason: str

    should_evaluate_response: bool

    use_current_message_for_retrieval: bool

    include_previous_tutor_context: bool

    replace_active_question: bool