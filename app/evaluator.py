import json

from app.ai import (
    chat_with_structured_ai,
)

from app.domain.evaluation.result import EvaluationResult


EVALUATOR_PROMPT = """
You are a learner-response evaluator in an educational AI tutoring system.

Evaluate the learner's response in relation to the tutor's MOST RECENT question.

Classify it into exactly ONE category:

correct
partial
misconception
dont_know

Definitions:

correct:
The learner correctly and sufficiently answers the tutor's most recent question.

partial:
The learner gives relevant and partly correct information,
but does not fully answer the tutor's most recent question.

misconception:
The learner demonstrates an incorrect understanding,
confuses important concepts, or gives an incorrect explanation.

dont_know:
The learner explicitly says they do not know,
cannot answer, or provides no meaningful attempt.

Important rules:

- Judge the response against the tutor's MOST RECENT question.
- Do not mark an answer as correct merely because it contains some correct facts.
- If the learner gives correct but incomplete information, classify it as partial.
- Be conservative when assigning correct.
- confidence must be a number between 0.0 and 1.0.
- reason must briefly explain why the classification was selected.
- misconception must contain a short description only when a misconception is detected.
- If there is no misconception, misconception must be null.

Evaluate the learner response carefully.

Return the evaluation using the required structured format.

Confidence must be between 0.0 and 1.0.

If there is no misconception, set misconception to null.
"""

EVALUATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "classification": {
            "type": "string",
            "enum": [
                "correct",
                "partial",
                "misconception",
                "dont_know",
            ],
        },
        "confidence": {
            "type": "number",
        },
        "reason": {
            "type": "string",
        },
        "misconception": {
            "type": [
                "string",
                "null",
            ],
        },
    },
    "required": [
        "classification",
        "confidence",
        "reason",
        "misconception",
    ],
    "additionalProperties": False,
}

def evaluate_response(
    original_question: str,
    tutor_question: str,
    learner_response: str,
) -> EvaluationResult:

    messages = [
        {
            "role": "system",
            "content": EVALUATOR_PROMPT,
        },
        {
            "role": "user",
            "content": f"""
Original learning question:
{original_question}

Tutor's most recent question:
{tutor_question}

Learner response:
{learner_response}

Evaluate the learner response.
""",
        },
    ]

    raw_result = chat_with_structured_ai(
        messages=messages,
        schema_name="learner_evaluation",
        schema=EVALUATOR_SCHEMA,
        task_name="evaluator",
    )

    try:
        data = json.loads(raw_result.strip())

    except json.JSONDecodeError:
        return EvaluationResult(
            classification="partial",
            confidence=0.0,
            reason="Evaluator returned invalid JSON.",
            misconception=None,
        )

    classification = data.get(
        "classification",
        "partial",
    )

    allowed = {
        "correct",
        "partial",
        "misconception",
        "dont_know",
    }

    if classification not in allowed:
        classification = "partial"

    confidence = data.get(
        "confidence",
        0.0,
    )

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(
        0.0,
        min(confidence, 1.0),
    )

    reason = data.get(
        "reason",
        None,
    )

    misconception = data.get(
        "misconception",
        None,
    )

    return EvaluationResult(
        classification=classification,
        confidence=confidence,
        reason=reason,
        misconception=misconception,
    )