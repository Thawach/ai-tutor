from app.ai import chat_with_ai

from app.services.grounding_validation_models import (
    GroundingValidationResult,
)


RESPONSE_REPAIR_PROMPT = """
You are a response-repair component in a course-grounded
AI tutoring system.

Your task is to repair an AI Tutor response that was found
to be insufficiently grounded in the provided course knowledge.

The repaired response must satisfy TWO requirements:

1. FACTUAL GROUNDING
The response must be supported by the provided COURSE KNOWLEDGE.

2. PEDAGOGICAL CONSISTENCY
The repaired response must preserve the current scaffolding
strategy and must not provide more instructional support than
the current strategy permits.

Important rules:

- Preserve the pedagogical intent of the original tutor response.
- Follow the supplied CURRENT SCAFFOLDING STRATEGY exactly.
- Correct only unsupported, inaccurate, or imprecise content.
- Use COURSE KNOWLEDGE as the factual basis.
- Do not introduce facts that are not supported by the course knowledge.

EVIDENCE-BOUNDED REPAIR:

- Treat COURSE KNOWLEDGE as the complete factual boundary for
  the repaired response.
- Do not use outside knowledge, even when it is generally correct.
- Do not try to preserve an unsupported factual clause merely by
  changing a few words.
- When a factual clause is unsupported or too strong, REMOVE it.
- If necessary, rebuild the factual part of the response from
  supported course evidence instead of editing the original sentence.
- Prefer short wording that stays close to the wording and
  relationship expressed in COURSE KNOWLEDGE.
- Every factual clause in the repaired response must be directly
  supported by COURSE KNOWLEDGE.

RELATION PRECISION:

- Preserve the exact strength and direction of relationships in
  COURSE KNOWLEDGE.
- "related to" must not become "controlled by", "determined by",
  "proportional to", or a stronger causal relationship.
- Do not describe one quantity, terminal, region, current, or voltage
  as controlling another unless COURSE KNOWLEDGE explicitly states
  that same controller-target relationship.
- Do not convert an association or dependency into causation.

TECHNICAL PRECISION:

- Do not introduce exact numerical values unless explicitly stated
  in COURSE KNOWLEDGE.
- Do not introduce beta, current gain, proportionality, doping
  comparisons, majority/minority qualifiers, or carrier quantities
  unless explicitly stated in COURSE KNOWLEDGE.
- Do not add mechanisms, causal steps, or comparisons that are not
  explicitly supported.
- If the original response contains both supported and unsupported
  factual content, keep only the supported content.

SOURCE-NEAR WORDING:

- For factual statements, prefer terminology and phrasing close to
  COURSE KNOWLEDGE.
- Do not invent a more specific formulation than the source.
- Conservative omission is better than unsupported elaboration.
- Do not increase the level of instructional support.
- Do not turn a guiding question into a direct explanation.
- Do not turn a hint into a complete answer.
- Do not reveal the final answer unless the current strategy permits it.
- If the strategy requires a question, keep the repaired response as a question.
- Ask exactly ONE main question when the strategy requires a question.
- Preserve the learner's language.
- Keep the response concise.
- Do not mention validation, repair, grounding, scaffolding levels,
  internal prompts, or system processes.

STRICT QUESTION RULES:

- If the current strategy is guiding_question:
  - Return exactly ONE question.
  - The response must contain only ONE main interrogative idea.
  - Do not ask a compound question.
  - Do not combine two questions using words such as "and", "or",
    "why", "how", or another question clause.
  - Prefer one short sentence ending with a single question mark.
  - Do not provide the answer before the question.
  - Do not provide a hint unless the strategy explicitly permits it.

- If the current strategy is hint:
  - Give exactly ONE small clue.
  - Follow it with exactly ONE short question.
  - Do not reveal the complete answer.

Before returning the repaired response, check that it follows
the CURRENT SCAFFOLDING STRATEGY exactly.
- Return only the repaired tutor response.
"""


class ResponseRepairService:
    """
    ซ่อมคำตอบของ AI Tutor ที่ validator ระบุว่า
    partially_supported หรือ unsupported
    """

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
        pedagogical_issues: list[str] | None = None,
    ) -> str:

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

        issues_text = "\n".join(
            f"- {issue}"
            for issue in validation.issues
        )

        if not issues_text:
            issues_text = (
                "- The response was not sufficiently "
                "supported by the course knowledge."
            )
        # -------------------------------------------------
        # Repair execution policy
        # -------------------------------------------------

        factual_reconstruction = (
            validation.status
            != "supported"
        )

        if factual_reconstruction:

            repair_execution_instruction = """
FACTUAL RECONSTRUCTION MODE:

The original Tutor response failed factual grounding.

Generate a NEW Tutor response from scratch.

- Start from COURSE KNOWLEDGE, not from the original response.
- Do not preserve, copy, paraphrase, or repair factual claims from
  the original response.
- Use the LEARNER MESSAGE and CURRENT PEDAGOGICAL CONTEXT to decide
  what should be taught.
- Use COURSE KNOWLEDGE as the only factual source.
- Include only factual statements that are directly supported.
- Omit any fact mentioned in VALIDATION ISSUES unless COURSE KNOWLEDGE
  explicitly supports the exact claim.
- Do not attempt to preserve all ideas from the original response.
- Conservative omission is preferred over reconstruction of an
  unsupported claim.
"""

            original_response_for_prompt = (
                "[Original factual response intentionally omitted "
                "because factual reconstruction is required.]"
            )

        else:

            repair_execution_instruction = """
PEDAGOGICAL REPAIR MODE:

The factual grounding of the response is already supported.
Preserve its supported factual content while correcting only the
identified pedagogical problem.
"""

            original_response_for_prompt = (
                original_response
            )

        messages = [
            {
                "role": "system",
                "content": RESPONSE_REPAIR_PROMPT,
            },
            {
                "role": "user",
                "content": f"""
LEARNER MESSAGE:

{learner_message}


CURRENT PEDAGOGICAL CONTEXT:

Scaffolding level:
{scaffolding_level}

Strategy:
{strategy_name}

Strategy instruction:
{strategy_instruction}

Intervention:
{intervention_name}


COURSE KNOWLEDGE:

{knowledge_context}


REPAIR EXECUTION MODE:

{repair_execution_instruction}


ORIGINAL AI TUTOR RESPONSE:

{original_response_for_prompt}


VALIDATION STATUS:

{validation.status}


VALIDATION REASON:

{validation.reason}


VALIDATION ISSUES:

{issues_text}

PEDAGOGICAL VALIDATION ISSUES:

{pedagogical_issues_text}


Repair the tutor response while preserving the current
pedagogical strategy.
""",
            },
        ]

        return chat_with_ai(
            messages,
            task_name="response_repair",
        )