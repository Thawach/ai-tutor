import json

from app.ai import (
    chat_with_structured_ai,
)

from app.services.evidence_safe_translation_models import (
    EvidenceSafeTranslationResult,
)


EVIDENCE_SAFE_TRANSLATION_PROMPT = """
You are a translation-only component in a
course-grounded AI tutoring system.

Your task is to translate VERIFIED EVIDENCE into the
requested target language.

STRICT TRANSLATION BOUNDARY:

- Translate only the supplied VERIFIED EVIDENCE.
- Preserve the factual meaning exactly.
- Do not summarize.
- Do not explain.
- Do not answer the learner independently.
- Do not add technical knowledge.
- Do not add examples.
- Do not add numerical values.
- Do not add qualifiers.
- Do not add causal relationships.
- Do not strengthen relationships.
- Do not merge separate source relationships into a
  stronger combined relationship.
- Do not infer missing subjects, controllers, targets,
  variables, equations, quantities, or mechanisms.

RELATIONSHIP PRECISION:

- "related to" must remain equivalent to
  "related to".
- It must NOT become "controls", "determines",
  "causes", or "is proportional to".
- "function of" must not become an exact numerical or
  proportional relationship.
- A generic control statement must not be translated
  into a specific controller-target relationship that
  the source does not explicitly state.

TECHNICAL PRECISION:

Preserve technical symbols, abbreviations, terminal
names, and variable names when present, for example:

- Base
- Collector
- Emitter
- B-E
- B-C
- IC
- IE
- VBE

Technical English may remain in parentheses when that
helps preserve meaning.

SOURCE BOUNDARY:

The VERIFIED EVIDENCE is the complete factual boundary.

Do not use outside knowledge.

Do not repair or reconstruct missing information in
the evidence.

Do not use the learner question, selector reasoning,
or prior Tutor response to introduce facts.

OUTPUT:

Return only a faithful translation of the supplied
VERIFIED EVIDENCE in the requested target language.
""".strip()


EVIDENCE_SAFE_TRANSLATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "translated_text": {
            "type": "string",
        },
    },
    "required": [
        "translated_text",
    ],
}


class EvidenceSafeTranslationService:

    _NO_TRANSLATION_LANGUAGES = frozenset(
        {
            "en",
            "english",
            "en-us",
            "en-gb",
        }
    )

    def prepare_source(
        self,
        evidence_text: str,
    ) -> EvidenceSafeTranslationResult:

        if not isinstance(
            evidence_text,
            str,
        ):
            raise TypeError(
                "evidence_text must be str."
            )

        source_text = evidence_text.strip()

        if not source_text:
            return EvidenceSafeTranslationResult(
                status="invalid",
                translated_text="",
                source_text="",
                reason=(
                    "Verified evidence text "
                    "was empty."
                ),
                issues=(
                    "No evidence available "
                    "for translation.",
                ),
            )

        return EvidenceSafeTranslationResult(
            status="not_required",
            translated_text="",
            source_text=source_text,
            reason=(
                "Verified evidence is ready "
                "for bounded translation."
            ),
            issues=(),
        )

    def translate(
        self,
        evidence_text: str,
        target_language: str,
    ) -> EvidenceSafeTranslationResult:

        prepared = self.prepare_source(
            evidence_text
        )

        if prepared.status == "invalid":
            return prepared

        if not isinstance(
            target_language,
            str,
        ):
            raise TypeError(
                "target_language must be str."
            )

        normalized_language = (
            target_language
            .strip()
            .lower()
        )

        if not normalized_language:
            return EvidenceSafeTranslationResult(
                status="invalid",
                translated_text="",
                source_text=prepared.source_text,
                reason=(
                    "Target language was empty."
                ),
                issues=(
                    "No target language was "
                    "available for translation.",
                ),
            )

        if (
            normalized_language
            in self._NO_TRANSLATION_LANGUAGES
        ):
            return EvidenceSafeTranslationResult(
                status="not_required",
                translated_text="",
                source_text=prepared.source_text,
                reason=(
                    "Target language does not "
                    "require translation."
                ),
                issues=(),
            )

        messages = [
            {
                "role": "system",
                "content": (
                    EVIDENCE_SAFE_TRANSLATION_PROMPT
                ),
            },
            {
                "role": "user",
                "content": f"""
TARGET LANGUAGE:

{normalized_language}


VERIFIED EVIDENCE:

{prepared.source_text}


Translate only the VERIFIED EVIDENCE above.
Do not add any factual content.
""".strip(),
            },
        ]

        try:
            raw_result = chat_with_structured_ai(
                messages=messages,
                schema_name=(
                    "evidence_safe_translation"
                ),
                schema=(
                    EVIDENCE_SAFE_TRANSLATION_SCHEMA
                ),
                task_name=(
                    "evidence_safe_translation"
                ),
            )

        except Exception as exc:
            return EvidenceSafeTranslationResult(
                status="invalid",
                translated_text="",
                source_text=prepared.source_text,
                reason=(
                    "Evidence-safe translation "
                    "provider failed."
                ),
                issues=(
                    (
                        "Translation provider "
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
            return EvidenceSafeTranslationResult(
                status="invalid",
                translated_text="",
                source_text=prepared.source_text,
                reason=(
                    "Evidence-safe translator "
                    "returned invalid JSON."
                ),
                issues=(
                    "Invalid translation output.",
                ),
            )

        translated_text = data.get(
            "translated_text"
        )

        if not isinstance(
            translated_text,
            str,
        ):
            return EvidenceSafeTranslationResult(
                status="invalid",
                translated_text="",
                source_text=prepared.source_text,
                reason=(
                    "Evidence-safe translator "
                    "returned an invalid "
                    "translation structure."
                ),
                issues=(
                    "translated_text was not "
                    "a string.",
                ),
            )

        translated_text = (
            translated_text.strip()
        )

        if not translated_text:
            return EvidenceSafeTranslationResult(
                status="invalid",
                translated_text="",
                source_text=prepared.source_text,
                reason=(
                    "Evidence-safe translator "
                    "returned empty text."
                ),
                issues=(
                    "Translated evidence was empty.",
                ),
            )

        return EvidenceSafeTranslationResult(
            status="translated",
            translated_text=translated_text,
            source_text=prepared.source_text,
            reason=(
                "Verified evidence was translated "
                "within the bounded translation "
                "contract."
            ),
            issues=(),
        )