import json

from app.ai import (
    chat_with_structured_ai,
)

from app.services.pedagogical_response_validator import (
    PedagogicalResponseValidator,
    PEDAGOGICAL_VALIDATION_SCHEMA,
)

from app.services.pedagogical_validation_models import (
    PedagogicalValidationResult,
)

from app.services.tutoring_response_mode_models import (
    TutoringResponseModeDecision,
)


RESPONSE_MODE_PEDAGOGICAL_VALIDATION_PROMPT = """
You are a pedagogical-response validator in an AI tutoring system.

Evaluate PEDAGOGICAL RESPONSE FORM only.

Do NOT evaluate factual correctness.
Factual grounding is handled by a separate validator.

The Tutor has both:

1. a normal scaffolding strategy, and
2. a turn-specific learner response mode.

IMPORTANT AUTHORITY RULE:

If PRESERVE SCAFFOLDING STRATEGY is false,
the CURRENT RESPONSE MODE governs the response form for this turn.

In that case, do NOT reject a direct explanation merely because
the underlying scaffolding strategy is normally a guiding-question
strategy.

RESPONSE-MODE RULES:

If DIRECT ANSWER REQUIRED is true:

- The Tutor must answer, explain, or clarify the learner's current
  question before any optional guiding question.
- A question-only response is a pedagogical violation.
- A direct explanation with no guiding question is allowed.
- A direct explanation followed by one optional guiding question
  is allowed if the configured question limit permits it.
- Do not require the Tutor to hide the requested clarification
  merely to preserve Socratic form.

GUIDING-QUESTION POLICY:

- "required":
  A guiding question must be present.

- "optional":
  A guiding question may be present but is not required.

- "forbidden":
  A guiding question must not be present.

MAXIMUM GUIDING QUESTIONS:

If a maximum is supplied, the Tutor response must not exceed it.

INTERVENTION:

The current intervention may still provide corrective or supportive
feedback, but it must not override the current response-mode rules.

EVIDENCE RULES:

- Judge only what is explicitly present in the AI Tutor response.
- Do not use outside knowledge.
- Do not evaluate factual truth.
- Do not infer an explanation that is not actually present.
- Do not classify a question-only response as a direct answer.
- Do not penalize a concise direct explanation merely because the
  normal strategy name is "guiding_question".

Return the result using the required structured format.

Confidence must be between 0.0 and 1.0.

If there are no pedagogical issues, return an empty issues list.
"""


class ResponseModePedagogicalValidator:
    """
    Semantic pedagogical validator that understands
    turn-specific tutoring response modes.

    Preserving modes delegate unchanged to the existing
    PedagogicalResponseValidator.

    Non-preserving modes use response-mode-aware semantic
    validation.

    Factual grounding remains outside this service.
    """

    ALLOWED_STATUS = {
        "valid",
        "violation",
    }

    def __init__(
        self,
        base_validator: PedagogicalResponseValidator | None = None,
    ) -> None:

        self.base_validator = (
            base_validator
            if base_validator is not None
            else PedagogicalResponseValidator()
        )

    def validate(
        self,
        response: str,
        scaffolding_level: int,
        strategy_name: str,
        strategy_instruction: str,
        intervention_name: str,
        intervention_instruction: str,
        response_mode: TutoringResponseModeDecision,
        response_mode_instruction: str = "",
        task_name: str = (
            "response_mode_pedagogical_validator"
        ),
    ) -> PedagogicalValidationResult:

        if not isinstance(
            response_mode,
            TutoringResponseModeDecision,
        ):
            raise TypeError(
                "response_mode must be "
                "TutoringResponseModeDecision."
            )

        # -------------------------------------------------
        # Preserve legacy validation exactly.
        # -------------------------------------------------

        if response_mode.preserve_scaffolding_strategy:

            return self.base_validator.validate(
                response=response,
                scaffolding_level=scaffolding_level,
                strategy_name=strategy_name,
                strategy_instruction=strategy_instruction,
                intervention_name=intervention_name,
                intervention_instruction=(
                    intervention_instruction
                ),
                task_name=task_name,
            )

        # -------------------------------------------------
        # Response-mode-aware semantic validation.
        # -------------------------------------------------

        messages = [
            {
                "role": "system",
                "content": (
                    RESPONSE_MODE_PEDAGOGICAL_VALIDATION_PROMPT
                ),
            },
            {
                "role": "user",
                "content": f"""
CURRENT SCAFFOLDING LEVEL:
{scaffolding_level}

CURRENT SCAFFOLDING STRATEGY:
{strategy_name}

STRATEGY INSTRUCTION:
{strategy_instruction}

CURRENT INTERVENTION:
{intervention_name}

INTERVENTION INSTRUCTION:
{intervention_instruction}

CURRENT RESPONSE MODE:
{response_mode.mode}

DIRECT ANSWER REQUIRED:
{response_mode.direct_answer_required}

GUIDING QUESTION POLICY:
{response_mode.guiding_question_policy}

MAX GUIDING QUESTIONS:
{response_mode.max_guiding_questions}

PRESERVE SCAFFOLDING STRATEGY:
{response_mode.preserve_scaffolding_strategy}

RESPONSE MODE INSTRUCTION:
{response_mode_instruction}

AI TUTOR RESPONSE:
{response}

Determine whether the AI Tutor response follows
the current turn-specific pedagogical constraints.
""",
            },
        ]

        try:
            raw_result = chat_with_structured_ai(
                messages=messages,
                schema_name=(
                    "response_mode_pedagogical_validation"
                ),
                schema=PEDAGOGICAL_VALIDATION_SCHEMA,
                task_name=task_name,
            )

        except Exception as exc:

            provider_error = str(exc).strip()

            if len(provider_error) > 1200:
                provider_error = (
                    provider_error[:1200]
                    + "..."
                )

            return PedagogicalValidationResult(
                status="violation",
                confidence=0.0,
                reason=(
                    "Response-mode pedagogical "
                    "validator provider failed."
                ),
                issues=[
                    (
                        "Response-mode pedagogical validator "
                        "service failure: "
                        f"{type(exc).__name__}: "
                        f"{provider_error}"
                    )
                ],
            )

        return self._parse_result(
            raw_result
        )

    def _parse_result(
        self,
        raw_result: str,
    ) -> PedagogicalValidationResult:

        try:
            data = json.loads(
                raw_result
            )

        except (
            json.JSONDecodeError,
            TypeError,
        ):
            return PedagogicalValidationResult(
                status="violation",
                confidence=0.0,
                reason=(
                    "Response-mode pedagogical "
                    "validator returned invalid JSON."
                ),
                issues=[
                    (
                        "Invalid response-mode "
                        "pedagogical validator output."
                    ),
                ],
            )

        status = data.get(
            "status",
            "violation",
        )

        if status not in self.ALLOWED_STATUS:
            status = "violation"

        confidence = data.get(
            "confidence",
            0.0,
        )

        try:
            confidence = float(
                confidence
            )

        except (
            TypeError,
            ValueError,
        ):
            confidence = 0.0

        confidence = max(
            0.0,
            min(
                confidence,
                1.0,
            ),
        )

        reason = data.get(
            "reason",
            "No reason provided.",
        )

        issues = data.get(
            "issues",
            [],
        )

        if not isinstance(
            issues,
            list,
        ):
            issues = []

        issues = [
            str(item)
            for item in issues
        ]

        return PedagogicalValidationResult(
            status=status,
            confidence=confidence,
            reason=str(reason),
            issues=issues,
        )