from app.services.tutoring_response_mode_models import (
    TutoringResponseModeDecision,
)


class TutoringResponseModeInstructionBuilder:
    """
    Converts a deterministic tutoring response-mode
    decision into prompt instructions.

    This builder:
    - does not call an LLM
    - does not mutate learner state
    - does not select learner intent
    - does not select scaffolding level
    """

    def build(
        self,
        decision: TutoringResponseModeDecision,
    ) -> str:

        if not isinstance(
            decision,
            TutoringResponseModeDecision,
        ):
            raise TypeError(
                "decision must be "
                "TutoringResponseModeDecision."
            )

        # -------------------------------------------------
        # Existing scaffolding behavior remains authoritative.
        # -------------------------------------------------

        if decision.preserve_scaffolding_strategy:
            return ""

        # -------------------------------------------------
        # Follow-up question:
        # answer first, then optionally guide.
        # -------------------------------------------------

        if decision.mode == "answer_then_guide":

            return (
                "LEARNER RESPONSE MODE:\n"
                "The learner asked a follow-up question.\n"
                "\n"
                "RESPONSE REQUIREMENTS:\n"
                "- Answer the learner's current question "
                "directly before asking any guiding question.\n"
                "- Do not respond only with another question.\n"
                "- Use the supplied course knowledge as the "
                "basis of the explanation.\n"
                "- Keep the direct explanation concise and "
                "appropriate to the learner's current level.\n"
                "- After answering, a guiding question is "
                "optional.\n"
                "- If a guiding question is used, ask no more "
                "than one.\n"
            )

        # -------------------------------------------------
        # Clarification:
        # directly explain the requested concept first.
        # -------------------------------------------------

        if decision.mode == "direct_clarification":

            return (
                "LEARNER RESPONSE MODE:\n"
                "The learner explicitly requested "
                "clarification.\n"
                "\n"
                "RESPONSE REQUIREMENTS:\n"
                "- Directly explain the term, concept, or "
                "point the learner asked to clarify.\n"
                "- Give the clarification before asking any "
                "question.\n"
                "- Do not respond only with a Socratic or "
                "guiding question.\n"
                "- Base factual explanations only on the "
                "supplied course knowledge.\n"
                "- Keep the clarification concise, clear, "
                "and appropriate to the learner's level.\n"
                "- A guiding question after the explanation "
                "is optional.\n"
                "- If a guiding question is used, ask no more "
                "than one.\n"
            )

        # -------------------------------------------------
        # Fail-safe
        #
        # Unknown non-preserving modes must not silently
        # invent new Tutor behavior.
        # -------------------------------------------------

        raise ValueError(
            "Unsupported non-preserving tutoring "
            f"response mode: {decision.mode}"
        )