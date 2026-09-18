import json

from app.ai import (
    chat_with_structured_ai,
)

from app.services.knowledge_models import (
    KnowledgeResult,
)

from app.services.relevance_models import (
    RelevanceResult,
)


RELEVANCE_PROMPT = """
You are a relevance classifier for a course-based AI tutoring system.

Your task is to determine whether the RETRIEVED COURSE KNOWLEDGE
contains information that is meaningfully relevant to the learner's query.

Important:

- The learner query and course material may be in different languages.
- Judge semantic relevance, not exact word matching.
- Do not answer the learner's question.
- Do not evaluate whether the learner is correct.
- Only determine whether the retrieved course material is useful
  for answering or scaffolding the query.

Return the result using the required structured format.

Confidence must be between 0.0 and 1.0.
"""

RELEVANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "is_relevant": {
            "type": "boolean",
        },
        "confidence": {
            "type": "number",
        },
        "reason": {
            "type": "string",
        },
    },
    "required": [
        "is_relevant",
        "confidence",
        "reason",
    ],
    "additionalProperties": False,
}

class RelevanceGate:
    """
    ตรวจ semantic relevance ระหว่าง
    retrieval query กับ course knowledge
    """

    def evaluate(
        self,
        query: str,
        knowledge_result: KnowledgeResult,
    ) -> RelevanceResult:

        if not knowledge_result.has_context:
            return RelevanceResult(
                is_relevant=False,
                confidence=1.0,
                reason="No retrieved course context.",
            )

        messages = [
            {
                "role": "system",
                "content": RELEVANCE_PROMPT,
            },
            {
                "role": "user",
                "content": f"""
LEARNER / RETRIEVAL QUERY:

{query}

RETRIEVED COURSE KNOWLEDGE:

{knowledge_result.context}

Determine whether the retrieved course knowledge
is relevant to the query.
""",
            },
        ]

        raw_result = chat_with_structured_ai(
            messages=messages,
            schema_name="knowledge_relevance",
            schema=RELEVANCE_SCHEMA,
            task_name="relevance_gate",
        )

        try:
            data = json.loads(
                raw_result.strip()
            )

        except json.JSONDecodeError:
            return RelevanceResult(
                is_relevant=False,
                confidence=0.0,
                reason=(
                    "Relevance classifier "
                    "returned invalid JSON."
                ),
            )

        is_relevant = data.get(
            "is_relevant",
            False,
        )

        if not isinstance(
            is_relevant,
            bool,
        ):
            is_relevant = False

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

        return RelevanceResult(
            is_relevant=is_relevant,
            confidence=confidence,
            reason=reason,
        )