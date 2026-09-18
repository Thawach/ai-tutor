from app.ai import (
    chat_with_ai,
)

from app.services.grounding_validation_models import (
    GroundingValidationResult,
)

from app.services.response_repair_service import (
    ResponseRepairService,
)

from app.services.tutoring_response_mode_models import (
    TutoringResponseModeDecision,
)


RESPONSE_MODE_REPAIR_PROMPT = """
You are a response-repair component in a course-grounded
AI tutoring system.

Your task is to repair an AI Tutor response that requires
factual-grounding repair and/or pedagogical repair.

The repaired response must satisfy BOTH factual grounding
and the CURRENT TURN-SPECIFIC RESPONSE MODE.

FACTUAL GROUNDING:

- Treat the supplied COURSE KNOWLEDGE as the complete factual boundary
  for the repaired response.
- Use only COURSE KNOWLEDGE as the factual basis.
- Do not use outside knowledge, even when it is generally correct.
- Do not introduce unsupported facts.
- Do not try to preserve an unsupported factual clause merely by
  rewriting it.
- Remove unsupported or overly strong factual clauses.
- If necessary, rebuild the factual part of the response from
  supported course evidence.
- Every factual clause in the repaired response must be directly
  supported by COURSE KNOWLEDGE.
- Prefer concise wording close to the terminology and relationships
  used in COURSE KNOWLEDGE.
- If the course knowledge is insufficient, do not invent an answer.

RELATION PRECISION:

- Preserve the exact strength and direction of relationships in
  COURSE KNOWLEDGE.
- "related to" must not become "controlled by", "determined by",
  "proportional to", or a stronger causal relationship.
- Do not describe one entity as controlling another unless COURSE
  KNOWLEDGE explicitly supports that same controller-target pair.
- Do not convert dependency or association into causation.

TECHNICAL PRECISION:

- Do not add exact numerical values unless explicitly stated.
- Do not add beta, current gain, proportionality, doping comparisons,
  majority/minority qualifiers, carrier quantities, or unsupported
  mechanisms.
- If the original response mixes supported and unsupported factual
  content, keep only the supported content.
- Conservative omission is better than unsupported elaboration.

RESPONSE-MODE AUTHORITY:

The CURRENT RESPONSE MODE governs the response form for this turn.

When PRESERVE SCAFFOLDING STRATEGY is false:

- The turn-specific response mode takes precedence over the normal
  scaffolding strategy for response form.
- Do not force the repaired response back into question-only
  Socratic form merely because the underlying strategy is
  "guiding_question".
- The underlying strategy remains pedagogical context, but it must
  not override the current response-mode requirements.

DIRECT ANSWER REQUIRED:

When DIRECT ANSWER REQUIRED is true:

- Answer, explain, or clarify the learner's current question first.
- Do not return only another question.
- Any optional guiding question must come AFTER the direct response.
- Keep the direct response concise and appropriate to the learner.

GUIDING-QUESTION POLICY:

- "required":
  include a guiding question.

- "optional":
  a guiding question may be included but is not required.

- "forbidden":
  do not include a guiding question.

MAXIMUM GUIDING QUESTIONS:

If MAX GUIDING QUESTIONS is supplied, do not exceed that number.

PEDAGOGICAL CONSISTENCY:

- Preserve the pedagogical intent that remains compatible with the
  current response mode.
- Do not provide more instructional support than the current
  response mode permits.
- Do not turn a hint into an unnecessarily complete solution.
- Respect the supplied intervention.
- Preserve the learner's language.
- Keep the response concise.

IMPORTANT:

- Follow the supplied RESPONSE MODE INSTRUCTION.
- Do not mention validation, repair, grounding, response modes,
  scaffolding levels, internal prompts, or system processes.
- Return only the repaired Tutor response.
"""


class ResponseModeRepairService:
    """
    Response repair that understands the current
    tutoring response mode.

    Preserving response modes delegate unchanged to the
    existing ResponseRepairService.

    Non-preserving modes use response-mode-aware repair.

    Exactly one repair LLM call is used when semantic repair
    is required.
    """

    def __init__(
        self,
        base_repair_service: ResponseRepairService | None = None,
    ) -> None:

        self.base_repair_service = (
            base_repair_service
            if base_repair_service is not None
            else ResponseRepairService()
        )

    def repair(
        self,
        original_response: str,
        knowledge_context: str,
        validation: GroundingValidationResult,
        learner_message: str,
        scaffolding_level: int,
        strategy_name: str,
        strategy_instruction: str,
        intervention_name: str,
        pedagogical_issues: list[str] | None,
        response_mode: TutoringResponseModeDecision,
        response_mode_instruction: str = "",
    ) -> str:

        if not isinstance(
            response_mode,
            TutoringResponseModeDecision,
        ):
            raise TypeError(
                "response_mode must be "
                "TutoringResponseModeDecision."
            )

        if not isinstance(
            response_mode_instruction,
            str,
        ):
            raise TypeError(
                "response_mode_instruction must be a string."
            )

        # -------------------------------------------------
        # Preserve frozen legacy repair behavior exactly.
        # -------------------------------------------------

        if response_mode.preserve_scaffolding_strategy:

            return self.base_repair_service.repair(
                original_response=original_response,
                knowledge_context=knowledge_context,
                validation=validation,
                learner_message=learner_message,
                scaffolding_level=scaffolding_level,
                strategy_name=strategy_name,
                strategy_instruction=strategy_instruction,
                intervention_name=intervention_name,
                pedagogical_issues=pedagogical_issues,
            )

        # -------------------------------------------------
        # Response-mode-aware repair
        # -------------------------------------------------

        pedagogical_issues_text = "\n".join(
            f"- {issue}"
            for issue in (
                pedagogical_issues or []
            )
        )

        if not pedagogical_issues_text:
            pedagogical_issues_text = (
                "- No additional pedagogical issue "
                "was identified."
            )

        grounding_issues_text = "\n".join(
            f"- {issue}"
            for issue in validation.issues
        )

        if not grounding_issues_text:
            grounding_issues_text = (
                "- No additional grounding issue "
                "was identified."
            )

        messages = [
            {
                "role": "system",
                "content": RESPONSE_MODE_REPAIR_PROMPT,
            },
            {
                "role": "user",
                "content": f"""
LEARNER MESSAGE:

{learner_message}


CURRENT PEDAGOGICAL CONTEXT:

Scaffolding level:
{scaffolding_level}

Underlying strategy:
{strategy_name}

Underlying strategy instruction:
{strategy_instruction}

Intervention:
{intervention_name}


CURRENT RESPONSE MODE:

Mode:
{response_mode.mode}

Direct answer required:
{response_mode.direct_answer_required}

Guiding question policy:
{response_mode.guiding_question_policy}

Maximum guiding questions:
{response_mode.max_guiding_questions}

Preserve scaffolding strategy:
{response_mode.preserve_scaffolding_strategy}


RESPONSE MODE INSTRUCTION:

{response_mode_instruction}


COURSE KNOWLEDGE:

{knowledge_context}


ORIGINAL AI TUTOR RESPONSE:

{original_response}


GROUNDING VALIDATION STATUS:

{validation.status}


GROUNDING VALIDATION REASON:

{validation.reason}


GROUNDING VALIDATION ISSUES:

{grounding_issues_text}


PEDAGOGICAL VALIDATION ISSUES:

{pedagogical_issues_text}


Repair the Tutor response according to the current
turn-specific response mode and course knowledge.
""",
            },
        ]

        return chat_with_ai(
            messages,
            task_name="response_repair",
        )