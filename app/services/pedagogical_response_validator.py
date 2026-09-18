import json

from app.ai import (
    chat_with_structured_ai,
)

from app.services.pedagogical_validation_models import (
    PedagogicalValidationResult,
)


PEDAGOGICAL_VALIDATION_PROMPT = """
You are a pedagogical-response validator in an AI tutoring system.

Your task is to determine whether the AI Tutor response follows
the CURRENT SCAFFOLDING STRATEGY and CURRENT INTERVENTION.

Evaluate pedagogical behavior only.

Do NOT evaluate factual correctness.
Factual grounding is handled by a separate validator.

GENERAL RULES:

- Judge the response against the supplied strategy instruction.
- The response must not provide more instructional support
  than the strategy permits.
- Preserve the expected scaffolding style.
- Do not penalize wording differences that preserve the same
  pedagogical intent.
- Do not use outside knowledge to judge factual correctness.

GUIDING-QUESTION STRATEGY:

- The tutor should primarily activate prior knowledge or reasoning.
- The tutor should ask one main guiding question.
- It should not reveal the answer.
- It should not provide a direct hint unless the strategy permits it.

HINT STRATEGY RULES:

A valid hint must give the learner ONE small piece of useful
guidance before asking the learner to continue reasoning.

A hint may take several forms, including:

- directing attention to one relevant feature,
- pointing out one relationship to inspect,
- reminding the learner of one previously known idea,
- narrowing the learner's attention to one part of the problem,
- giving one partial cue without completing the answer.

VALID HINT:

"Hint: ในทรานซิสเตอร์ NPN ชั้นที่อยู่ด้านบนและด้านล่าง
เป็น n-type ส่วนชั้นตรงกลางเป็น p-type.
คุณคิดว่าชั้นใดควรเป็น emitter, base, และ collector?"

This is a VALID hint because:

1. The first sentence provides useful partial guidance.
2. The second sentence asks the learner to reason from that guidance.
3. The response does not directly assign emitter, base, and collector
   for the learner.

Do NOT classify this pattern as "question only".

IMPORTANT:

An attention-directing phrase COUNTS as a hint.

For example:

"ลองสังเกตตัวอักษรที่อยู่ตรงกลางของ N-P-N
แล้วคิดว่าชั้น Base ควรเป็นชนิดใด?"

is a VALID hint because the first part directs attention to
a relevant feature and the second part asks the learner to
reason from that clue.

Do NOT classify such a response as "question only".

A follow-up question is allowed and is normally expected after
the small clue.

Examples:

VALID HINT:
"ลองสังเกตตัวอักษรที่อยู่ตรงกลางของ N-P-N
แล้วคิดว่าชั้น Base ควรเป็นชนิดใด?"

INVALID — QUESTION ONLY:
"ชั้น Base ของ NPN เป็นชนิดใด?"

INVALID — TOO MUCH HELP:
"NPN คือ N-P-N โดย Base เป็น P-type และ Emitter กับ
Collector เป็น N-type."

CONCEPTUAL-SUPPORT STRATEGY:

- The tutor may briefly explain the key missing concept.
- It should not unnecessarily solve the complete learning problem.
- The learner should still be asked to continue reasoning or try again.

INTERVENTION RULES:

- If misconception correction is active, corrective feedback is allowed.
- Correction must still respect the current scaffolding strategy.
- Intervention must not automatically justify revealing the entire answer.

EVIDENCE RULES:

- Evaluate ONLY what is explicitly present in the AI TUTOR RESPONSE.
- Do not infer that the tutor revealed information merely because
  that information appears in the course topic, strategy instruction,
  intervention instruction, or surrounding context.
- Never attribute a factual statement to the tutor unless that
  statement actually appears in the AI TUTOR RESPONSE.
- Before marking "full answer", verify that the response itself
  explicitly reveals the answer.
- A question that asks the learner to identify an answer is not the
  same as the tutor stating that answer.
- If the response asks the learner to determine emitter, base,
  collector, material type, or another concept, do not claim that
  those values were already revealed unless they are explicitly stated.
- Base the reason and issues only on observable features of the
  supplied AI TUTOR RESPONSE.

Return the result using the required structured format.

Confidence must be between 0.0 and 1.0.

If there are no pedagogical issues, return an empty issues list.
"""


PEDAGOGICAL_VALIDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": [
                "valid",
                "violation",
            ],
        },
        "confidence": {
            "type": "number",
        },
        "reason": {
            "type": "string",
        },
        "issues": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
    },
    "required": [
        "status",
        "confidence",
        "reason",
        "issues",
    ],
    "additionalProperties": False,
}


class PedagogicalResponseValidator:
    """
    ตรวจว่าคำตอบ Tutor สอดคล้องกับ
    scaffolding strategy และ intervention ปัจจุบันหรือไม่
    """

    ALLOWED_STATUS = {
        "valid",
        "violation",
    }

    def validate(
        self,
        response: str,
        scaffolding_level: int,
        strategy_name: str,
        strategy_instruction: str,
        intervention_name: str,
        intervention_instruction: str,
        task_name: str = "pedagogical_validator",
    ) -> PedagogicalValidationResult:

        question_mark_count = (
            response.count("?")
            + response.count("？")
        )

        has_single_question_mark = (
            question_mark_count == 1
        )

        normalized_response = (
            response
            .replace("？", "?")
            .strip()
        )

        response_lines = [
            line.strip()
            for line in normalized_response.splitlines()
            if line.strip()
        ]

        hint_candidate = ""
        question_text = ""

        if response_lines:

            if (
                len(response_lines) >= 2
                and response_lines[-1].endswith("?")
            ):

                hint_candidate = " ".join(
                    response_lines[:-1]
                )

                question_text = (
                    response_lines[-1]
                )

            elif normalized_response.endswith("?"):

                question_text = (
                    normalized_response
                )

        messages = [
            {
                "role": "system",
                "content": PEDAGOGICAL_VALIDATION_PROMPT,
            },
            {
                "role": "user",
                "content": f"""
CURRENT PEDAGOGICAL CONTEXT:

Scaffolding level:
{scaffolding_level}

Strategy:
{strategy_name}

Strategy instruction:
{strategy_instruction}

Intervention:
{intervention_name}

Intervention instruction:
{intervention_instruction}


AI TUTOR RESPONSE — START

{response}

AI TUTOR RESPONSE — END

Evaluate only the text between
AI TUTOR RESPONSE — START
and
AI TUTOR RESPONSE — END.


STRUCTURAL OBSERVATIONS COMPUTED BY THE SYSTEM:

Question-mark count:
{question_mark_count}

Contains exactly one explicit question mark:
{has_single_question_mark}

Text appearing before the follow-up question:
{hint_candidate if hint_candidate else "(none)"}

Detected follow-up question:
{question_text if question_text else "(none)"}

IMPORTANT INTERPRETATION RULES:

- Treat the structural observations above as reliable.
- When strategy is "hint", explicitly inspect the text shown under
  "Text appearing before the follow-up question".
- If that text gives useful guidance, directs attention, narrows the
  problem, reminds the learner of relevant information, or provides
  a partial clue, it COUNTS as a hint.
- Do not ignore a declarative sentence simply because a question
  follows it.
- A response consisting of a clue followed by one question is NOT
  "question only".
- The literal label "Hint:" is strong evidence that the preceding
  statement is intended as a hint, but still judge whether the
  content actually gives useful guidance.

Important:

- Treat the structural observations above as reliable.
- Do not claim that the response contains multiple explicit questions
  when the question-mark count is exactly 1.
- A phrase such as "ส่วนใดบ้าง?" is ONE question, not multiple questions.
- A plural or open-ended request within one interrogative sentence
  does not automatically count as multiple guiding questions.
- Judge whether there is one MAIN pedagogical question, not how many
  possible pieces of information the learner could include in an answer.

Determine whether the tutor response follows
the current pedagogical constraints.

Examples:

VALID — ONE GUIDING QUESTION:
"คุณคิดว่าทรานซิสเตอร์ NPN ประกอบด้วยส่วนใดบ้าง?"

This is ONE guiding question.
The phrase "ส่วนใดบ้าง" does not make it multiple questions.

VALID — ONE GUIDING QUESTION:
"คุณคิดว่าชั้นตรงกลางของโครงสร้าง NPN ควรเป็นชนิดใด?"

INVALID — MULTIPLE QUESTIONS:
"ทรานซิสเตอร์ NPN มีส่วนประกอบอะไรบ้าง?
และแต่ละส่วนทำหน้าที่อย่างไร?"

This contains two separate questions.
""",
            },
        ]

        try:
            raw_result = chat_with_structured_ai(
                messages=messages,
                schema_name=(
                    "pedagogical_response_validation"
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
                    "Pedagogical validator provider failed."
                ),
                issues=[
                    (
                        "Pedagogical validator service failure: "
                        f"{type(exc).__name__}: "
                        f"{provider_error}"
                    )
                ],
            )

        try:
            data = json.loads(
                raw_result
            )

        except json.JSONDecodeError:

            return PedagogicalValidationResult(
                status="violation",
                confidence=0.0,
                reason=(
                    "Pedagogical validator returned "
                    "invalid JSON."
                ),
                issues=[
                    "Invalid pedagogical validator output."
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

        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(
            0.0,
            min(confidence, 1.0),
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
            reason=reason,
            issues=issues,
        )