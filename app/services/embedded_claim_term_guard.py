import re
from dataclasses import dataclass

from app.courses.models import (
    CourseProfile,
)


@dataclass
class EmbeddedClaimTermGuardResult:
    """
    Deterministic result for checking:

    1. suspicious technical terminology
    2. factual claims embedded in tutor responses

    status:
    - clear
    - needs_validation
    - terminology_issue
    """

    status: str
    reason: str
    issues: list[str]


class EmbeddedClaimTermGuard:
    """
    Course-aware deterministic guard.

    Domain-specific terminology and claim patterns
    are provided through CourseProfile.

    This guard does NOT decide whether a factual
    statement is true or false.

    It only determines whether semantic grounding
    validation should be performed.
    """

    def __init__(
        self,
        course_profile: CourseProfile | None = None,
    ):

        self.course_profile = (
            course_profile
        )

        if course_profile is None:

            self.technical_terms = {}
            self.embedded_claim_patterns = []

        else:

            self.technical_terms = dict(
                course_profile.technical_terms
            )

            self.embedded_claim_patterns = list(
                course_profile.embedded_claim_patterns
            )

    def evaluate(
        self,
        response: str,
    ) -> EmbeddedClaimTermGuardResult:

        text = response.strip()

        # -------------------------
        # Empty response
        # -------------------------

        if not text:

            return EmbeddedClaimTermGuardResult(
                status="needs_validation",
                reason=(
                    "Tutor response is empty and cannot "
                    "be safely prevalidated."
                ),
                issues=[
                    "Empty tutor response.",
                ],
            )

        lowered = text.lower()

        # =====================================================
        # 1. Technical terminology check
        # =====================================================

        terminology_issues = []

        for (
            suspicious_term,
            preferred_term,
        ) in self.technical_terms.items():

            if (
                suspicious_term.lower()
                in lowered
            ):

                terminology_issues.append(
                    (
                        "Suspicious technical term "
                        f"'{suspicious_term}' detected; "
                        "expected terminology is "
                        f"'{preferred_term}'."
                    )
                )

        if terminology_issues:

            return EmbeddedClaimTermGuardResult(
                status="terminology_issue",
                reason=(
                    "A potentially incorrect technical "
                    "term was detected for the active course."
                ),
                issues=terminology_issues,
            )

        # =====================================================
        # 2. Embedded factual claim check
        # =====================================================

        claim_issues = []

        for pattern in self.embedded_claim_patterns:

            normalized_pattern = (
                pattern.strip()
            )

            if not normalized_pattern:
                continue

            match = re.search(
                normalized_pattern,
                text,
                re.IGNORECASE,
            )

            if match:

                claim_issues.append(
                    (
                        "Possible factual assertion embedded "
                        "inside the tutor response: "
                        f"'{match.group(0)}'."
                    )
                )

        if claim_issues:

            return EmbeddedClaimTermGuardResult(
                status="needs_validation",
                reason=(
                    "A possible course-specific factual "
                    "claim is embedded inside the response "
                    "and requires semantic grounding "
                    "validation."
                ),
                issues=claim_issues,
            )

        # =====================================================
        # 3. Clear
        # =====================================================

        return EmbeddedClaimTermGuardResult(
            status="clear",
            reason=(
                "No suspicious technical terminology or "
                "configured embedded factual claim was "
                "detected for the active course."
            ),
            issues=[],
        )