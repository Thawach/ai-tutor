from app.services.response_quality_models import (
    ResponseQualityResult,
)

from app.services.response_quality_policy_models import (
    ResponseQualityPolicyResult,
)


class ResponseQualityPolicy:
    """
    Deterministic interpretation policy for
    ResponseQualityResult.

    This policy does not:
    - call an LLM
    - rewrite responses
    - repair responses
    - regenerate responses
    - change validation escalation
    """

    STATUS_ACCEPTABLE = "acceptable"

    STATUS_ADVISORY = "advisory"

    STATUS_ATTENTION = "attention"

    def evaluate(
        self,
        quality_result: ResponseQualityResult,
    ) -> ResponseQualityPolicyResult:

        if not isinstance(
            quality_result,
            ResponseQualityResult,
        ):
            raise TypeError(
                "quality_result must be "
                "a ResponseQualityResult."
            )

        issues = list(
            quality_result.issues
        )

        issue_count = len(
            issues
        )

        # =====================================================
        # ACCEPTABLE
        # =====================================================

        if (
            quality_result.status
            == "acceptable"
        ):

            return ResponseQualityPolicyResult(
                status=self.STATUS_ACCEPTABLE,
                reason=(
                    "No response-quality issue "
                    "requires policy attention."
                ),
                issues=issues,
                source_status=(
                    quality_result.status
                ),
                issue_count=issue_count,
                requires_attention=False,
            )

        # =====================================================
        # UNCERTAIN
        # =====================================================

        if (
            quality_result.status
            == "uncertain"
        ):

            return ResponseQualityPolicyResult(
                status=self.STATUS_ATTENTION,
                reason=(
                    "Response quality is uncertain "
                    "and requires attention."
                ),
                issues=issues,
                source_status=(
                    quality_result.status
                ),
                issue_count=issue_count,
                requires_attention=True,
            )

        # =====================================================
        # WARNING
        # =====================================================

        if (
            quality_result.status
            == "warning"
        ):

            active_flag_count = sum(
                (
                    quality_result.too_long,
                    quality_result.too_short,
                    quality_result.too_many_questions,
                    quality_result.repetitive,
                )
            )

            if (
                active_flag_count >= 2
                or issue_count >= 2
            ):

                return ResponseQualityPolicyResult(
                    status=self.STATUS_ATTENTION,
                    reason=(
                        "Multiple response-quality "
                        "issues require attention."
                    ),
                    issues=issues,
                    source_status=(
                        quality_result.status
                    ),
                    issue_count=issue_count,
                    requires_attention=True,
                )

            return ResponseQualityPolicyResult(
                status=self.STATUS_ADVISORY,
                reason=(
                    "A response-quality issue "
                    "was detected for advisory "
                    "telemetry."
                ),
                issues=issues,
                source_status=(
                    quality_result.status
                ),
                issue_count=issue_count,
                requires_attention=False,
            )

        # =====================================================
        # UNKNOWN / FAIL-SAFE
        # =====================================================

        return ResponseQualityPolicyResult(
            status=self.STATUS_ATTENTION,
            reason=(
                "Unknown response-quality status "
                "requires conservative attention."
            ),
            issues=issues,
            source_status=(
                quality_result.status
            ),
            issue_count=issue_count,
            requires_attention=True,
        )