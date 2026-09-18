from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationEscalationDecision:
    """
    Final escalation decision after deterministic and
    semantic response validation.

    action:
    - accept
    - deterministic_repair
    - llm_repair
    """

    action: str
    reason: str

    grounding_issue: bool = False
    terminology_issue: bool = False
    pedagogical_issue: bool = False


class ValidationEscalationPolicy:
    """
    Decide how the tutor pipeline should handle a validated
    response.

    This class performs NO LLM calls and does not modify the
    response itself.

    Its responsibility is only to choose the cheapest safe
    next action.
    """

    ACCEPT = "accept"
    DETERMINISTIC_REPAIR = "deterministic_repair"
    LLM_REPAIR = "llm_repair"

    def decide(
        self,
        grounding_validation,
        embedded_claim_term_guard,
        pedagogical_validation,
    ) -> ValidationEscalationDecision:

        # =====================================================
        # Normalize statuses
        # =====================================================

        grounding_status = getattr(
            grounding_validation,
            "status",
            None,
        )

        term_status = getattr(
            embedded_claim_term_guard,
            "status",
            None,
        )

        pedagogical_status = getattr(
            pedagogical_validation,
            "status",
            None,
        )

        # =====================================================
        # Detect issue classes
        # =====================================================

        terminology_issue = (
            term_status == "terminology_issue"
        )

        grounding_issue = grounding_status in {
            "partially_supported",
            "unsupported",
        }

        pedagogical_issue = (
            pedagogical_status == "violation"
        )

        # =====================================================
        # 1. Terminology-only problem
        #
        # A configured terminology mapping can be repaired
        # deterministically without another LLM call.
        # =====================================================

        if (
            terminology_issue
            and not pedagogical_issue
        ):

            return ValidationEscalationDecision(
                action=self.DETERMINISTIC_REPAIR,
                reason=(
                    "Configured terminology issue can be "
                    "repaired deterministically."
                ),
                grounding_issue=grounding_issue,
                terminology_issue=True,
                pedagogical_issue=False,
            )

        # =====================================================
        # 2. Grounding or pedagogical problem
        #
        # These may require semantic rewriting and therefore
        # must use the LLM repair service.
        # =====================================================

        if (
            grounding_issue
            or pedagogical_issue
        ):

            return ValidationEscalationDecision(
                action=self.LLM_REPAIR,
                reason=(
                    "Semantic grounding or pedagogical "
                    "repair is required."
                ),
                grounding_issue=grounding_issue,
                terminology_issue=terminology_issue,
                pedagogical_issue=pedagogical_issue,
            )

        # =====================================================
        # 3. Safe response
        # =====================================================

        return ValidationEscalationDecision(
            action=self.ACCEPT,
            reason=(
                "No grounding, terminology, or pedagogical "
                "issue requires repair."
            ),
            grounding_issue=False,
            terminology_issue=False,
            pedagogical_issue=False,
        )