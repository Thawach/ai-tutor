from app.services.grounding_validation_models import (
    GroundingValidationResult,
)

from app.services.grounding_precision_guard import (
    GroundingPrecisionGuardResult,
)


class GroundingPrecisionVetoPolicy:
    """
    Combines semantic grounding validation with the
    deterministic grounding precision guard.

    The precision guard acts as a veto:
    a semantic 'supported' result cannot remain supported
    when unsupported high-risk precision is detected.

    This policy:
    - does not call an LLM
    - does not inspect course knowledge directly
    - does not mutate the supplied results
    """

    def apply(
        self,
        validation: GroundingValidationResult,
        precision: GroundingPrecisionGuardResult,
    ) -> GroundingValidationResult:

        if not isinstance(
            validation,
            GroundingValidationResult,
        ):
            raise TypeError(
                "validation must be "
                "GroundingValidationResult."
            )

        if not isinstance(
            precision,
            GroundingPrecisionGuardResult,
        ):
            raise TypeError(
                "precision must be "
                "GroundingPrecisionGuardResult."
            )

        # -------------------------------------------------
        # No deterministic precision concern.
        #
        # Preserve the semantic validator result exactly.
        # -------------------------------------------------

        if precision.status == "clear":
            return validation

        # -------------------------------------------------
        # Precision veto
        #
        # A response cannot remain fully supported when
        # deterministic evidence checks found unsupported
        # precision.
        # -------------------------------------------------

        if validation.status == "supported":
            status = "partially_supported"

        else:
            status = validation.status

        # -------------------------------------------------
        # Preserve existing semantic issues and append
        # deterministic precision issues without duplicates.
        # -------------------------------------------------

        issues = list(
            validation.issues
        )

        for issue in precision.issues:
            if issue not in issues:
                issues.append(issue)

        reason = (
            f"{validation.reason} "
            "Deterministic precision evidence checks "
            "found unsupported or insufficiently "
            "grounded technical detail."
        )

        return GroundingValidationResult(
            status=status,
            confidence=validation.confidence,
            reason=reason,
            issues=issues,
        )