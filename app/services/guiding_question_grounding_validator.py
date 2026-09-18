import json
import re

from app.ai import (
    chat_with_structured_ai,
)

from app.services.guiding_question_grounding_models import (
    GuidingQuestionGroundingResult,
)


GUIDING_QUESTION_GROUNDING_PROMPT = """
You validate guiding questions in a course-grounded
AI tutoring system.

Your task is NOT to validate factual statements in
the Tutor response.

Your task is ONLY to determine whether each guiding
question asks the learner for factual information
that is answerable from the supplied COURSE KNOWLEDGE.

COURSE KNOWLEDGE IS THE COMPLETE FACTUAL BOUNDARY.

Do not use outside knowledge.
Do not use general technical knowledge.
Do not infer missing facts.
Do not repair damaged source text.

QUESTION TYPES

1. factual
A question that expects a factual, technical,
quantitative, causal, relational, classificatory,
or definitional answer.

Examples:
- What voltage should VBE be?
- What value of beta is used?
- What is collector current related to?
- What range should the voltage be in?
- Which terminal controls the current?

A factual question is supported ONLY when the course
knowledge contains sufficient evidence to answer it.

2. reflective
A question that asks the learner to observe, compare,
reason from information already presented, or express
their interpretation without requiring a new factual
detail that is absent from the course evidence.

Examples:
- What relationship do you notice?
- How would you describe this relationship in your
  own words?
- What do you observe from the information above?

Reflective questions do not require an evidence quote
unless they themselves demand a specific factual fact.

STRICT ANSWERABILITY RULES

A question is unsupported when it asks for information
that the course evidence does not provide.

This includes unsupported:
- exact numerical values;
- numerical ranges;
- thresholds;
- efficiency or optimum criteria;
- gain or beta values;
- quantities or majority claims;
- stronger causal or control relationships;
- mechanisms absent from the evidence;
- comparisons absent from the evidence.

Do not treat related information as sufficient evidence
for a more specific question.

Examples:

Course evidence:
"The collector current is related to the emitter
current which is in turn a function of the B-E voltage."

Question:
"What is collector current related to?"
→ factual + answerable.

Question:
"What exact B-E voltage is required?"
→ factual + unsupported unless an exact value exists.

Question:
"What B-E voltage range gives efficient operation?"
→ factual + unsupported unless both the range and the
efficiency criterion exist explicitly.

EVIDENCE RULES

For every factual question classified as answerable:
- evidence_quote must be an exact excerpt from COURSE
  KNOWLEDGE;
- the evidence must actually answer the question;
- weaker evidence cannot support a stronger question.

For unsupported factual questions:
- evidence_quote must be empty;
- issue must explain what requested information is
  absent.

For reflective questions:
- evidence_quote must be empty;
- issue must be empty.

QUESTION QUOTE RULE

question_quote must be copied exactly from the Tutor
response.

Return every guiding question found in the Tutor
response.

Do not omit a question merely because it would be
unsupported.
""".strip()


GUIDING_QUESTION_GROUNDING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "question_quote": {
                        "type": "string",
                    },
                    "question_type": {
                        "type": "string",
                        "enum": [
                            "factual",
                            "reflective",
                        ],
                    },
                    "status": {
                        "type": "string",
                        "enum": [
                            "answerable",
                            "unsupported",
                            "reflective",
                        ],
                    },
                    "evidence_quote": {
                        "type": "string",
                    },
                    "issue": {
                        "type": "string",
                    },
                },
                "required": [
                    "question_quote",
                    "question_type",
                    "status",
                    "evidence_quote",
                    "issue",
                ],
            },
        },
    },
    "required": [
        "questions",
    ],
}


class GuidingQuestionGroundingValidator:

    _QUESTION_HINTS = (
        "?",
        "？",
        "คุณคิดว่า",
        "คุณคิดอย่างไร",
        "ลองคิดว่า",
        "ลองพิจารณาว่า",
        "what ",
        "why ",
        "how ",
        "which ",
    )

    _EXACT_VALUE_QUESTION_HINTS = (
        "เท่าไร",
        "เท่าไหร่",
        "กี่โวลต์",
        "กี่โวลท์",
        "ค่าเท่าไร",
        "ค่าเท่าไหร่",
        "what value",
        "what exact ",
        "exact value",
        "exact voltage",
        "what voltage should",
        "how much",
        "how many",
    )

    _RANGE_QUESTION_HINTS = (
        "ช่วงใด",
        "ช่วงไหน",
        "ช่วงเท่าไร",
        "ช่วงเท่าไหร่",
        "อยู่ในช่วง",
        "what range",
        "which range",
        "voltage range",
        "current range",
    )

    _THRESHOLD_QUESTION_HINTS = (
        # Thai
        "มากพอ",
        "พอเพียง",
        "เพียงพอแค่ไหน",
        "เพียงพอเท่าไร",
        "เพียงพอเท่าไหร่",
        "อย่างน้อยเท่าไร",
        "อย่างน้อยเท่าไหร่",
        "ขั้นต่ำเท่าไร",
        "ขั้นต่ำเท่าไหร่",

        # English
        "threshold",
        "minimum value",
        "minimum voltage",
        "at least how much",
        "how much is enough",
        "sufficient voltage",
        "sufficient current",
    )

    _EFFICIENCY_QUESTION_HINTS = (
        "มีประสิทธิภาพ",
        "ประสิทธิภาพ",
        "เหมาะสมที่สุด",
        "ค่าที่เหมาะสม",
        "ดีที่สุด",
        "efficient",
        "efficiency",
        "optimal",
        "optimum",
    )

    _EFFICIENCY_EVIDENCE_HINTS = (
        "มีประสิทธิภาพ",
        "ประสิทธิภาพ",
        "เหมาะสมที่สุด",
        "ค่าที่เหมาะสม",
        "ดีที่สุด",
        "efficient",
        "efficiency",
        "optimal",
        "optimum",
    )

    def validate(
        self,
        response: str,
        knowledge_context: str,
    ) -> GuidingQuestionGroundingResult:

        if not isinstance(
            response,
            str,
        ):
            raise TypeError(
                "response must be str."
            )

        if not isinstance(
            knowledge_context,
            str,
        ):
            raise TypeError(
                "knowledge_context must be str."
            )

        response_text = response.strip()
        knowledge_text = knowledge_context.strip()

        if not self._may_contain_question(
            response_text
        ):
            return GuidingQuestionGroundingResult(
                status="no_questions",
                reason=(
                    "Tutor response contains no "
                    "detectable guiding question."
                ),
                issues=(),
            )

        if not knowledge_text:
            return GuidingQuestionGroundingResult(
                status="unsupported",
                reason=(
                    "A guiding question is present "
                    "but no course knowledge is "
                    "available to bound factual "
                    "answerability."
                ),
                issues=(
                    (
                        "Guiding question cannot be "
                        "verified against course "
                        "knowledge."
                    ),
                ),
            )

        messages = [
            {
                "role": "system",
                "content": (
                    GUIDING_QUESTION_GROUNDING_PROMPT
                ),
            },
            {
                "role": "user",
                "content": f"""
TUTOR RESPONSE:

{response_text}


COURSE KNOWLEDGE:

{knowledge_text}


Evaluate only the guiding questions in the Tutor
response.
""".strip(),
            },
        ]

        try:
            raw_result = chat_with_structured_ai(
                messages=messages,
                schema_name=(
                    "guiding_question_grounding"
                ),
                schema=(
                    GUIDING_QUESTION_GROUNDING_SCHEMA
                ),
                task_name=(
                    "guiding_question_validator"
                ),
            )

        except Exception as exc:
            return GuidingQuestionGroundingResult(
                status="invalid",
                reason=(
                    "Guiding-question grounding "
                    "validator provider failed."
                ),
                issues=(
                    (
                        "Guiding-question validator "
                        f"failure: "
                        f"{type(exc).__name__}."
                    ),
                ),
            )

        try:
            data = json.loads(
                raw_result.strip()
            )

        except (
            json.JSONDecodeError,
            AttributeError,
        ):
            return GuidingQuestionGroundingResult(
                status="invalid",
                reason=(
                    "Guiding-question validator "
                    "returned invalid JSON."
                ),
                issues=(
                    (
                        "Invalid guiding-question "
                        "validation output."
                    ),
                ),
            )

        questions = data.get(
            "questions"
        )

        if not isinstance(
            questions,
            list,
        ):
            return self._invalid(
                "questions was not an array."
            )

        # A question was detected deterministically.
        # The semantic validator may not silently omit it.
        if not questions:
            return self._invalid(
                (
                    "Semantic validator returned no "
                    "questions for a response that "
                    "contains a question."
                )
            )

        unsupported_issues: list[str] = []

        for index, item in enumerate(
            questions,
            start=1,
        ):

            if not isinstance(
                item,
                dict,
            ):
                return self._invalid(
                    (
                        f"Question {index} was not "
                        "an object."
                    )
                )

            question_quote = item.get(
                "question_quote"
            )

            question_type = item.get(
                "question_type"
            )

            status = item.get(
                "status"
            )

            evidence_quote = item.get(
                "evidence_quote"
            )

            issue = item.get(
                "issue"
            )

            if not all(
                isinstance(value, str)
                for value in (
                    question_quote,
                    question_type,
                    status,
                    evidence_quote,
                    issue,
                )
            ):
                return self._invalid(
                    (
                        f"Question {index} has "
                        "malformed fields."
                    )
                )

            question_quote = (
                question_quote.strip()
            )

            evidence_quote = (
                evidence_quote.strip()
            )

            issue = issue.strip()

            if (
                not question_quote
                or
                not self._quote_exists(
                    quote=question_quote,
                    source=response_text,
                )
            ):
                return self._invalid(
                    (
                        f"Question {index} quote was "
                        "not found in the Tutor "
                        "response."
                    )
                )

            if question_type == "reflective":

                if (
                    status != "reflective"
                    or evidence_quote
                    or issue
                ):
                    return self._invalid(
                        (
                            f"Reflective question "
                            f"{index} violated the "
                            "reflective-question "
                            "contract."
                        )
                    )

                continue

            if question_type != "factual":
                return self._invalid(
                    (
                        f"Question {index} has an "
                        "unknown question type."
                    )
                )

            if status == "answerable":

                if not evidence_quote:
                    return self._invalid(
                        (
                            f"Answerable factual "
                            f"question {index} has "
                            "no source evidence."
                        )
                    )

                if not self._quote_exists(
                    quote=evidence_quote,
                    source=knowledge_text,
                ):
                    return self._invalid(
                        (
                            f"Evidence for question "
                            f"{index} was not found "
                            "in course knowledge."
                        )
                    )

                if issue:
                    return self._invalid(
                        (
                            f"Answerable factual "
                            f"question {index} "
                            "unexpectedly contains "
                            "an issue."
                        )
                    )

                precision_issue = (
                    self._answerability_precision_issue(
                        question=question_quote,
                        evidence=evidence_quote,
                    )
                )

                if precision_issue is not None:
                    unsupported_issues.append(
                        precision_issue
                    )
                    continue

                continue

            if status == "unsupported":

                if evidence_quote:
                    return self._invalid(
                        (
                            f"Unsupported factual "
                            f"question {index} "
                            "unexpectedly contains "
                            "source evidence."
                        )
                    )

                if not issue:
                    return self._invalid(
                        (
                            f"Unsupported factual "
                            f"question {index} has "
                            "no issue description."
                        )
                    )

                unsupported_issues.append(
                    issue
                )

                continue

            return self._invalid(
                (
                    f"Question {index} has an "
                    "unknown status."
                )
            )

        if unsupported_issues:
            return GuidingQuestionGroundingResult(
                status="unsupported",
                reason=(
                    "One or more factual guiding "
                    "questions require information "
                    "that is not supported by "
                    "course knowledge."
                ),
                issues=tuple(
                    dict.fromkeys(
                        unsupported_issues
                    )
                ),
            )

        return GuidingQuestionGroundingResult(
            status="supported",
            reason=(
                "All factual guiding questions "
                "are answerable from course "
                "knowledge, and reflective "
                "questions require no unsupported "
                "facts."
            ),
            issues=(),
        )


    def _answerability_precision_issue(
        self,
        *,
        question: str,
        evidence: str,
    ) -> str | None:

        question_norm = self._normalize(
            question
        ).lower()

        evidence_norm = self._normalize(
            evidence
        ).lower()

        asks_exact_value = any(
            hint in question_norm
            for hint
            in self._EXACT_VALUE_QUESTION_HINTS
        )

        asks_range = any(
            hint in question_norm
            for hint
            in self._RANGE_QUESTION_HINTS
        )

        asks_threshold = any(
            hint in question_norm
            for hint
            in self._THRESHOLD_QUESTION_HINTS
        )

        asks_efficiency = any(
            hint in question_norm
            for hint
            in self._EFFICIENCY_QUESTION_HINTS
        )

        numeric_values = re.findall(
            r"(?<![\w.])[+-]?\d+(?:\.\d+)?",
            evidence_norm,
        )

        # -------------------------------------------------
        # Efficiency / optimum questions
        #
        # Evidence about a relationship or operating
        # mechanism cannot establish an efficiency or
        # optimum criterion unless that criterion itself
        # appears in the cited evidence.
        # -------------------------------------------------

        if asks_efficiency:

            has_efficiency_evidence = any(
                hint in evidence_norm
                for hint
                in self._EFFICIENCY_EVIDENCE_HINTS
            )

            if not has_efficiency_evidence:
                return (
                    "The guiding question requests "
                    "an efficiency or optimum "
                    "criterion, but the cited course "
                    "evidence does not state such a "
                    "criterion."
                )

        # -------------------------------------------------
        # Exact-value questions
        #
        # Related qualitative evidence cannot answer
        # a request for an exact technical value.
        # -------------------------------------------------

        if (
            asks_exact_value
            and
            not numeric_values
        ):
            return (
                "The guiding question requests an "
                "exact quantitative value, but the "
                "cited course evidence contains no "
                "explicit numerical value."
            )

        # -------------------------------------------------
        # Numerical range questions
        #
        # A numerical range requires explicit bounds.
        # One related value or qualitative relationship
        # is not sufficient.
        # -------------------------------------------------

        if (
            asks_range
            and
            len(numeric_values) < 2
        ):
            return (
                "The guiding question requests a "
                "numerical range, but the cited "
                "course evidence does not provide "
                "explicit numerical bounds."
            )

        # -------------------------------------------------
        # Threshold questions
        #
        # Expressions such as 'how much is enough' or
        # 'มากพอแค่ไหน' request a magnitude/threshold.
        # A qualitative B-E relationship alone cannot
        # establish that threshold.
        # -------------------------------------------------

        if (
            asks_threshold
            and
            not numeric_values
        ):
            return (
                "The guiding question requests a "
                "quantitative threshold, but the "
                "cited course evidence provides no "
                "explicit threshold value."
            )


        return None

    def _invalid(
        self,
        issue: str,
    ) -> GuidingQuestionGroundingResult:

        return GuidingQuestionGroundingResult(
            status="invalid",
            reason=(
                "Guiding-question grounding "
                "validation failed closed."
            ),
            issues=(
                issue,
            ),
        )

    def _may_contain_question(
        self,
        text: str,
    ) -> bool:

        lowered = text.lower()

        # English interrogative words such as "which"
        # must occur at the beginning of a question-like
        # sentence/segment. A plain substring match would
        # incorrectly classify relative clauses such as:
        #
        # "... emitter current which is in turn ..."
        #
        # as questions.
        english_question_starters = (
            "what ",
            "why ",
            "how ",
            "which ",
        )

        for marker in self._QUESTION_HINTS:

            marker_lower = marker.lower()

            if marker_lower in english_question_starters:

                # Question begins the response.
                if (
                    lowered
                    .lstrip(" \t-*•")
                    .startswith(marker_lower)
                ):
                    return True

                # Question begins a later sentence or line.
                #
                # Comma is intentionally excluded because
                # ", which ..." commonly introduces a
                # relative clause rather than a question.
                for delimiter in (
                    "\n",
                    ". ",
                    "! ",
                    "? ",
                    ": ",
                    "; ",
                ):
                    segments = lowered.split(
                        delimiter
                    )

                    if any(
                        segment
                        .lstrip(" \t-*•")
                        .startswith(marker_lower)
                        for segment in segments[1:]
                    ):
                        return True

                continue

            # Preserve existing behavior for punctuation
            # and the existing Thai guiding-question hints.
            if marker_lower in lowered:
                return True

        return False

    def _quote_exists(
        self,
        quote: str,
        source: str,
    ) -> bool:

        quote_norm = self._normalize(
            quote
        )

        source_norm = self._normalize(
            source
        )

        return (
            bool(quote_norm)
            and
            quote_norm in source_norm
        )

    def _normalize(
        self,
        text: str,
    ) -> str:

        return " ".join(
            text
            .replace("\u00a0", " ")
            .replace("\u202f", " ")
            .replace("？", "?")
            .split()
        ).lower()